import asyncio

from quarry.domain.models import ChunkObject, RetrievedPassage
from quarry.pipeline.retrieval import HybridRetriever, build_citation_index, reciprocal_rank_fusion


def build_chunk(chunk_id: str, text: str) -> ChunkObject:
    return ChunkObject(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_title="Doc 1",
        text=text,
        section_heading="Heading",
        section_path="Chapter 1 > Heading",
        page_start=1,
        page_end=1,
    )


def test_reciprocal_rank_fusion_rewards_consensus() -> None:
    chunk_a = build_chunk("a", "modular schedule improvements")
    chunk_b = build_chunk("b", "procurement delays")
    chunk_c = build_chunk("c", "safety benefits")

    sparse = [
        RetrievedPassage(chunk=chunk_a, score=3.0, source_facet="facet", rank=1, retriever="sparse"),
        RetrievedPassage(chunk=chunk_b, score=2.0, source_facet="facet", rank=2, retriever="sparse"),
    ]
    dense = [
        RetrievedPassage(chunk=chunk_c, score=0.95, source_facet="facet", rank=1, retriever="dense"),
        RetrievedPassage(chunk=chunk_a, score=0.90, source_facet="facet", rank=2, retriever="dense"),
    ]

    fused = reciprocal_rank_fusion([sparse, dense], rrf_k=60)

    assert fused[0].chunk.chunk_id == "a"


class FailingRetriever:
    async def search(self, *args, **kwargs):
        raise RuntimeError("boom")


class StaticRetriever:
    def __init__(self, result: RetrievedPassage) -> None:
        self.result = result
        self.top_ks: list[int] = []

    async def search(self, *args, **kwargs):
        self.top_ks.append(kwargs["top_k"])
        return [self.result], {"retriever": "static", "result_count": 1, "fallback_used": False}


class IdentityReranker:
    async def rerank(self, query, candidates):
        return list(candidates)


def test_hybrid_retriever_survives_single_retriever_failure() -> None:
    chunk = build_chunk("a", "modular schedule improvements")
    static = RetrievedPassage(chunk=chunk, score=1.0, source_facet="facet", rank=1, retriever="dense")
    retriever = HybridRetriever(
        sparse_retriever=FailingRetriever(),
        dense_retriever=StaticRetriever(static),
        reranker=IdentityReranker(),
        sparse_top_k=30,
        dense_top_k=30,
        rerank_top_k=20,
        rrf_k=60,
    )

    passages, diagnostics = asyncio.run(retriever.retrieve(original_query="query", facets=["facet"]))

    assert len(passages) == 1
    assert passages[0].chunk.chunk_id == "a"
    assert diagnostics[0].sparse.error == "boom"


def test_hybrid_retriever_uses_reduced_single_hop_limits() -> None:
    chunk = build_chunk("a", "modular schedule improvements")
    sparse = StaticRetriever(RetrievedPassage(chunk=chunk, score=1.0, source_facet="facet", rank=1, retriever="sparse"))
    dense = StaticRetriever(RetrievedPassage(chunk=chunk, score=1.0, source_facet="facet", rank=1, retriever="dense"))
    retriever = HybridRetriever(
        sparse_retriever=sparse,
        dense_retriever=dense,
        reranker=IdentityReranker(),
        sparse_top_k=30,
        dense_top_k=30,
        rerank_top_k=20,
        rrf_k=60,
    )

    passages, _diagnostics = asyncio.run(
        retriever.retrieve(original_query="What is PDRI maturity?", facets=["What is PDRI maturity?"], query_type="single_hop")
    )

    assert len(passages) == 1
    assert sparse.top_ks == [12]
    assert dense.top_ks == [12]


def test_build_citation_index_only_marks_top_result_with_global_ambiguity() -> None:
    chunk_a = build_chunk("a", "modular schedule improvements")
    chunk_b = build_chunk("b", "procurement delays")
    chunk_c = build_chunk("c", "safety benefits")

    passages = [
        RetrievedPassage(chunk=chunk_a, score=0.99, source_facet="facet", rank=1, retriever="reranked"),
        RetrievedPassage(chunk=chunk_b, score=0.985, source_facet="facet", rank=2, retriever="reranked"),
        RetrievedPassage(chunk=chunk_c, score=0.91, source_facet="facet", rank=3, retriever="reranked"),
    ]

    citations = build_citation_index(passages, ambiguity_gap_threshold=0.05)

    assert citations[0].ambiguity_review_required is True
    assert citations[0].ambiguity_gap == 0.005
    assert citations[1].ambiguity_review_required is False
    assert citations[1].ambiguity_gap is None
    assert citations[2].ambiguity_review_required is False
    assert citations[2].ambiguity_gap is None
