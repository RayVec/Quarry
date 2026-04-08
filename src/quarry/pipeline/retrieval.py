from __future__ import annotations

import asyncio
from collections.abc import Sequence

from quarry.adapters.interfaces import Reranker, Retriever
from quarry.domain.models import CitationIndexEntry, FacetRetrievalDiagnostic, QueryType, RetrieverDiagnostic, RetrievalFilters, RetrievedPassage
from quarry.logging_utils import elapsed_ms, logger_with_trace, timed


logger = logger_with_trace(__name__)


def reciprocal_rank_fusion(result_sets: Sequence[Sequence[RetrievedPassage]], *, rrf_k: int) -> list[RetrievedPassage]:
    merged: dict[str, RetrievedPassage] = {}
    fused_scores: dict[str, float] = {}
    for result_set in result_sets:
        for index, item in enumerate(result_set, start=1):
            fused_scores[item.chunk.chunk_id] = fused_scores.get(item.chunk.chunk_id, 0.0) + 1.0 / (rrf_k + index)
            if item.chunk.chunk_id not in merged or item.score > merged[item.chunk.chunk_id].score:
                merged[item.chunk.chunk_id] = item
    fused = [merged[chunk_id].model_copy(update={"score": score}) for chunk_id, score in fused_scores.items()]
    fused.sort(key=lambda item: item.score, reverse=True)
    return [item.model_copy(update={"rank": index + 1}) for index, item in enumerate(fused)]


class HybridRetriever:
    SINGLE_HOP_SPARSE_TOP_K = 12
    SINGLE_HOP_DENSE_TOP_K = 12
    SINGLE_HOP_RERANK_TOP_K = 8
    MULTIHOP_FOLLOWUP_RERANK_TOP_K = 10

    def __init__(
        self,
        *,
        sparse_retriever: Retriever,
        dense_retriever: Retriever,
        reranker: Reranker,
        sparse_top_k: int,
        dense_top_k: int,
        rerank_top_k: int,
        rrf_k: int,
        multihop_anchor_pool_size: int = 40,
        multihop_rerank_budget: int = 20,
    ) -> None:
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.reranker = reranker
        self.sparse_top_k = sparse_top_k
        self.dense_top_k = dense_top_k
        self.rerank_top_k = rerank_top_k
        self.multihop_anchor_pool_size = multihop_anchor_pool_size
        self.multihop_rerank_budget = multihop_rerank_budget
        self.rrf_k = rrf_k

    async def retrieve(
        self,
        *,
        original_query: str,
        facets: list[str],
        query_type: QueryType | str | None = None,
    ) -> tuple[list[RetrievedPassage], list[FacetRetrievalDiagnostic]]:
        retrieval_start = timed()
        sparse_top_k, dense_top_k, rerank_top_k = self._resolve_limits(query_type)
        is_multi_hop = self._is_multi_hop(query_type)
        logger.info(
            "hybrid retrieval started",
            extra={
                "original_query": original_query,
                "facets": facets,
                "facet_count": len(facets),
                "query_type": query_type.value if isinstance(query_type, QueryType) else query_type,
                "sparse_top_k": sparse_top_k,
                "dense_top_k": dense_top_k,
                "rerank_top_k": rerank_top_k,
                "console_visible": False,
            },
        )
        per_facet = await asyncio.gather(
            *(self._retrieve_facet(facet, sparse_top_k=sparse_top_k, dense_top_k=dense_top_k) for facet in facets)
        )
        merged: dict[str, RetrievedPassage] = {}
        diagnostics: list[FacetRetrievalDiagnostic] = []
        for passages, diagnostic in per_facet:
            diagnostics.append(diagnostic)
            for passage in passages:
                existing = merged.get(passage.chunk.chunk_id)
                if existing is None:
                    merged[passage.chunk.chunk_id] = passage.model_copy(update={"source_facets": list(dict.fromkeys(passage.source_facets or [passage.source_facet]))})
                    continue
                merged_facets = list(dict.fromkeys([*(existing.source_facets or [existing.source_facet]), *(passage.source_facets or [passage.source_facet])]))
                if passage.score > existing.score:
                    merged[passage.chunk.chunk_id] = passage.model_copy(update={"source_facets": merged_facets})
                else:
                    merged[passage.chunk.chunk_id] = existing.model_copy(update={"source_facets": merged_facets})
        rerank_candidates = list(merged.values())
        rerank_candidates.sort(key=lambda item: item.score, reverse=True)
        if is_multi_hop and len(rerank_candidates) > self.multihop_anchor_pool_size:
            rerank_candidates = rerank_candidates[: self.multihop_anchor_pool_size]
        reranked = await self.reranker.rerank(original_query, rerank_candidates)
        logger.info(
            "hybrid retrieval complete",
            extra={
                "merged_candidate_count": len(merged),
                "rerank_candidate_count": len(rerank_candidates),
                "reranked_count": len(reranked),
                "top_chunk_ids": [passage.chunk.chunk_id for passage in reranked[:5]],
                "latency_ms": elapsed_ms(retrieval_start),
                "console_visible": False,
            },
        )
        return reranked[:rerank_top_k], diagnostics

    async def retrieve_followup(
        self,
        *,
        original_query: str,
        facet: str,
        query_type: QueryType | str | None = None,
    ) -> tuple[list[RetrievedPassage], FacetRetrievalDiagnostic]:
        sparse_top_k, dense_top_k, _ = self._resolve_limits(query_type)
        fused, diagnostic = await self._retrieve_facet(
            facet,
            sparse_top_k=sparse_top_k,
            dense_top_k=dense_top_k,
        )
        reranked = await self.reranker.rerank(original_query, fused)
        return reranked[: self.MULTIHOP_FOLLOWUP_RERANK_TOP_K], diagnostic.model_copy(
            update={"reranked_count": min(len(reranked), self.MULTIHOP_FOLLOWUP_RERANK_TOP_K)}
        )

    async def scoped_retrieve(
        self,
        *,
        query: str,
        source_facet: str,
        filters: RetrievalFilters,
        top_k: int,
    ) -> tuple[list[RetrievedPassage], FacetRetrievalDiagnostic]:
        scoped_start = timed()
        logger.info(
            "scoped retrieval started",
            extra={
                "query": query,
                "source_facet": source_facet,
                "filters": filters.model_dump() if filters else None,
                "top_k": top_k,
                "console_visible": False,
            },
        )
        sparse_results, sparse_meta = await self._safe_search(
            self.sparse_retriever,
            query,
            top_k=top_k,
            source_facet=source_facet,
            filters=filters,
            retriever_name="sparse",
        )
        dense_results, dense_meta = await self._safe_search(
            self.dense_retriever,
            query,
            top_k=top_k,
            source_facet=source_facet,
            filters=filters,
            retriever_name="dense",
        )
        fused = reciprocal_rank_fusion([sparse_results, dense_results], rrf_k=self.rrf_k)
        reranked = await self.reranker.rerank(query, fused)
        diagnostic = FacetRetrievalDiagnostic(
            facet=source_facet,
            sparse=_to_retriever_diagnostic("sparse", sparse_meta, len(sparse_results)),
            dense=_to_retriever_diagnostic("dense", dense_meta, len(dense_results)),
            fused_count=len(fused),
            reranked_count=min(len(reranked), top_k),
            top_score_gap=_score_gap(reranked),
            top_rerank_score=reranked[0].score if reranked else None,
            degraded_mode=bool((sparse_meta or {}).get("fallback_used") or (dense_meta or {}).get("fallback_used")),
        )
        logger.info(
            "scoped retrieval complete",
            extra={
                "source_facet": source_facet,
                "result_count": len(reranked[:top_k]),
                "diagnostic": diagnostic.model_dump(),
                "latency_ms": elapsed_ms(scoped_start),
                "console_visible": False,
            },
        )
        return reranked[:top_k], diagnostic

    async def _retrieve_facet(
        self,
        facet: str,
        *,
        sparse_top_k: int,
        dense_top_k: int,
    ) -> tuple[list[RetrievedPassage], FacetRetrievalDiagnostic]:
        facet_start = timed()
        logger.info("facet retrieval started", extra={"facet": facet, "console_visible": False})
        sparse_task = self._safe_search(
            self.sparse_retriever,
            facet,
            top_k=sparse_top_k,
            source_facet=facet,
            retriever_name="sparse",
        )
        dense_task = self._safe_search(
            self.dense_retriever,
            facet,
            top_k=dense_top_k,
            source_facet=facet,
            retriever_name="dense",
        )
        (sparse_results, sparse_meta), (dense_results, dense_meta) = await asyncio.gather(sparse_task, dense_task)
        sparse_results = [
            passage.model_copy(update={"source_facets": [facet]})
            for passage in sparse_results
        ]
        dense_results = [
            passage.model_copy(update={"source_facets": [facet]})
            for passage in dense_results
        ]
        fused = reciprocal_rank_fusion([sparse_results, dense_results], rrf_k=self.rrf_k)
        diagnostic = FacetRetrievalDiagnostic(
            facet=facet,
            sparse=_to_retriever_diagnostic("sparse", sparse_meta, len(sparse_results)),
            dense=_to_retriever_diagnostic("dense", dense_meta, len(dense_results)),
            fused_count=len(fused),
            reranked_count=len(fused),
            top_score_gap=_score_gap(fused),
            top_rerank_score=fused[0].score if fused else None,
            degraded_mode=bool((sparse_meta or {}).get("fallback_used") or (dense_meta or {}).get("fallback_used")),
        )
        logger.info(
            "facet retrieval complete",
            extra={
                "facet": facet,
                "sparse_count": len(sparse_results),
                "dense_count": len(dense_results),
                "fused_count": len(fused),
                "top_chunk_ids": [passage.chunk.chunk_id for passage in fused[:5]],
                "diagnostic": diagnostic.model_dump(),
                "latency_ms": elapsed_ms(facet_start),
                "console_visible": False,
            },
        )
        return fused, diagnostic

    def _resolve_limits(self, query_type: QueryType | str | None) -> tuple[int, int, int]:
        raw_query_type = query_type.value if isinstance(query_type, QueryType) else query_type
        if raw_query_type == QueryType.SINGLE_HOP.value:
            return (
                min(self.sparse_top_k, self.SINGLE_HOP_SPARSE_TOP_K),
                min(self.dense_top_k, self.SINGLE_HOP_DENSE_TOP_K),
                min(self.rerank_top_k, self.SINGLE_HOP_RERANK_TOP_K),
            )
        return (
            self.sparse_top_k,
            self.dense_top_k,
            min(self.multihop_rerank_budget, self.rerank_top_k),
        )

    def _is_multi_hop(self, query_type: QueryType | str | None) -> bool:
        raw_query_type = query_type.value if isinstance(query_type, QueryType) else query_type
        return raw_query_type == QueryType.MULTI_HOP.value

    async def _safe_search(
        self,
        retriever: Retriever,
        query: str,
        *,
        top_k: int,
        source_facet: str,
        retriever_name: str,
        filters: RetrievalFilters | None = None,
    ) -> tuple[list[RetrievedPassage], dict[str, object] | None]:
        search_start = timed()
        try:
            results, meta = await retriever.search(
                query,
                top_k=top_k,
                source_facet=source_facet,
                filters=filters,
            )
            logger.info(
                "retriever search complete",
                extra={
                    "retriever_name": retriever_name,
                    "source_facet": source_facet,
                    "result_count": len(results),
                    "meta": meta,
                    "latency_ms": elapsed_ms(search_start),
                    "console_visible": False,
                },
            )
            return results, meta
        except Exception as exc:
            logger.warning(
                "retriever search failed",
                extra={
                    "retriever_name": retriever_name,
                    "source_facet": source_facet,
                    "error": str(exc),
                    "latency_ms": elapsed_ms(search_start),
                    "console_visible": False,
                },
            )
            return [], {
                "retriever": retriever_name,
                "result_count": 0,
                "fallback_used": True,
                "error": str(exc),
            }


def build_citation_index(passages: Sequence[RetrievedPassage], *, starting_id: int = 1, ambiguity_gap_threshold: float = 0.05) -> list[CitationIndexEntry]:
    citations: list[CitationIndexEntry] = []
    gap = _score_gap(passages)
    for offset, passage in enumerate(passages):
        chunk = passage.chunk
        top_result_is_ambiguous = offset == 0 and gap is not None and gap < ambiguity_gap_threshold
        citations.append(
            CitationIndexEntry(
                citation_id=starting_id + offset,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                section_heading=chunk.section_heading,
                section_path=chunk.section_path,
                page_number=chunk.page_start,
                page_end=chunk.page_end,
                retrieval_score=passage.score,
                source_facet=passage.source_facet,
                source_facets=passage.source_facets or [passage.source_facet],
                ambiguity_review_required=top_result_is_ambiguous,
                ambiguity_gap=gap if offset == 0 else None,
                retrieval_scores={passage.retriever: passage.score},
            )
        )
    return citations


def _to_retriever_diagnostic(name: str, payload: dict[str, object] | None, result_count: int) -> RetrieverDiagnostic:
    payload = payload or {}
    return RetrieverDiagnostic(
        retriever=name,
        result_count=int(payload.get("result_count", result_count)),
        latency_ms=float(payload["latency_ms"]) if payload.get("latency_ms") is not None else None,
        fallback_used=bool(payload.get("fallback_used", False)),
        error=str(payload["error"]) if payload.get("error") else None,
        provider=str(payload["provider"]) if payload.get("provider") else None,
    )


def _score_gap(passages: Sequence[RetrievedPassage]) -> float | None:
    if len(passages) < 2:
        return None
    return round(passages[0].score - passages[1].score, 6)
