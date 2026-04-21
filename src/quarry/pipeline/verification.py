from __future__ import annotations

from collections import OrderedDict

from quarry.adapters.interfaces import ChunkStore, NLIClient
from quarry.domain.models import ConfidenceLabel, CoverageCheckResult, CitationIndexEntry, ParsedSentence, ReviewWarning, ScoredReference, SentenceStatus, SentenceType, VerificationResult
from quarry.logging_utils import logger_with_trace


logger = logger_with_trace(__name__)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).strip()


class VerificationService:
    CONFIDENCE_CACHE_MAX_ENTRIES = 4096
    DEFAULT_MIN_QUOTE_WORDS = 10

    def __init__(self, *, chunk_store: ChunkStore, nli_client: NLIClient) -> None:
        self.chunk_store = chunk_store
        self.nli_client = nli_client
        self._confidence_cache: OrderedDict[tuple[str, str], ScoredReference] = OrderedDict()
        self.quote_lookup_metrics: dict[str, float] = {
            "scoped_lookups": 0.0,
            "full_corpus_fallbacks": 0.0,
            "total_lookups": 0.0,
            "matched_lookups": 0.0,
            "total_candidates_checked": 0.0,
            "avg_candidates_checked": 0.0,
            "quote_match_rate": 0.0,
        }

    def _record_quote_lookup(self, *, scoped: bool, used_fallback: bool, matched: bool, candidates_checked: int) -> None:
        if scoped:
            self.quote_lookup_metrics["scoped_lookups"] += 1.0
        if used_fallback:
            self.quote_lookup_metrics["full_corpus_fallbacks"] += 1.0
        self.quote_lookup_metrics["total_lookups"] += 1.0
        if matched:
            self.quote_lookup_metrics["matched_lookups"] += 1.0
        self.quote_lookup_metrics["total_candidates_checked"] += float(candidates_checked)

        total_lookups = max(self.quote_lookup_metrics["total_lookups"], 1.0)
        self.quote_lookup_metrics["avg_candidates_checked"] = (
            self.quote_lookup_metrics["total_candidates_checked"] / total_lookups
        )
        self.quote_lookup_metrics["quote_match_rate"] = (
            self.quote_lookup_metrics["matched_lookups"] / total_lookups
        )

    def verify_exact_matches(
        self,
        parsed_sentences: list[ParsedSentence],
        citation_index: list[CitationIndexEntry],
        *,
        sentence_indices: set[int] | None = None,
    ) -> VerificationResult:
        targets = parsed_sentences if sentence_indices is None else [sentence for sentence in parsed_sentences if sentence.sentence_index in sentence_indices]
        logger.info(
            "exact match verification started",
            extra={
                "sentence_count": len(parsed_sentences),
                "target_sentence_count": len(targets),
                "citation_count": len(citation_index),
                "console_visible": False,
            },
        )
        chunk_ids_in_citations = [citation.chunk_id for citation in citation_index]
        citation_by_chunk_id = {citation.chunk_id: citation for citation in citation_index}
        next_citation_id = max([citation.citation_id for citation in citation_index], default=0) + 1

        for sentence in targets:
            if sentence.status == SentenceStatus.NO_REF:
                continue
            if sentence.sentence_type == SentenceType.STRUCTURE:
                sentence.status = SentenceStatus.UNCHECKED
                continue

            verified_count = 0
            failed_count = 0
            for reference in sentence.references:
                reference.matched_chunk_id = None
                reference.verified = False
                reference.confidence_score = None
                reference.confidence_label = None
                reference.confidence_unknown = False
                reference.citation_id = None

                min_quote_words = reference.minimum_quote_words or self.DEFAULT_MIN_QUOTE_WORDS
                if len(normalize_whitespace(reference.reference_quote).split()) < min_quote_words:
                    failed_count += 1
                    continue

                used_fallback = False
                chunk = self.chunk_store.find_chunk_by_quote(reference.reference_quote, chunk_ids=chunk_ids_in_citations)
                if chunk is None:
                    used_fallback = True
                    chunk = self.chunk_store.find_chunk_by_quote(reference.reference_quote)
                    if chunk is not None and chunk.chunk_id not in citation_by_chunk_id:
                        citation = CitationIndexEntry(
                            citation_id=next_citation_id,
                            chunk_id=chunk.chunk_id,
                            text=chunk.text,
                            document_id=chunk.document_id,
                            document_title=chunk.document_title,
                            section_heading=chunk.section_heading,
                            section_path=chunk.section_path,
                            page_number=chunk.page_start,
                            page_end=chunk.page_end,
                            retrieval_score=0.0,
                            source_facet="quote_discovery",
                            source_facets=["quote_discovery"],
                        )
                        citation_index.append(citation)
                        citation_by_chunk_id[chunk.chunk_id] = citation
                        chunk_ids_in_citations.append(chunk.chunk_id)
                        next_citation_id += 1
                        logger.info(
                            "quote discovery added citation",
                            extra={
                                "chunk_id": chunk.chunk_id,
                                "document_id": chunk.document_id,
                                "section_path": chunk.section_path,
                                "citation_id": citation.citation_id,
                                "console_visible": False,
                            },
                        )
                scoped_candidate_count = len(chunk_ids_in_citations)
                total_candidate_count = self.chunk_store.chunk_count() if used_fallback else scoped_candidate_count
                self._record_quote_lookup(
                    scoped=True,
                    used_fallback=used_fallback,
                    matched=chunk is not None,
                    candidates_checked=total_candidate_count,
                )

                if chunk is None:
                    failed_count += 1
                    continue

                citation = citation_by_chunk_id[chunk.chunk_id]
                reference.matched_chunk_id = chunk.chunk_id
                reference.verified = True
                reference.citation_id = citation.citation_id
                reference.document_id = citation.document_id
                reference.document_title = citation.document_title
                reference.section_heading = citation.section_heading
                reference.section_path = citation.section_path
                reference.page_number = citation.page_number
                reference.replacement_pending = citation.replacement_pending
                verified_count += 1

            if sentence.sentence_type == SentenceType.CLAIM:
                sentence.status = SentenceStatus.UNCHECKED if verified_count == len(sentence.references) and verified_count > 0 else SentenceStatus.UNGROUNDED
            elif sentence.sentence_type == SentenceType.SYNTHESIS:
                if verified_count == 0:
                    sentence.status = SentenceStatus.UNGROUNDED
                elif failed_count > 0:
                    sentence.status = SentenceStatus.PARTIALLY_VERIFIED
                else:
                    sentence.status = SentenceStatus.UNCHECKED
            logger.info(
                "sentence exact match result",
                extra={
                    "sentence_index": sentence.sentence_index,
                    "sentence_type": sentence.sentence_type.value,
                    "verified_count": verified_count,
                    "failed_count": failed_count,
                    "status": sentence.status.value,
                    "console_visible": False,
                },
            )

        logger.info(
            "exact match verification finished",
            extra={
                "sentence_status_summary": _sentence_status_summary(parsed_sentences),
                "citation_count": len(citation_index),
                "quote_lookup_metrics": self.quote_lookup_metrics,
            },
        )
        return VerificationResult(parsed_sentences=parsed_sentences, citation_index=citation_index)

    def check_facet_coverage(
        self,
        *,
        facets: list[str],
        parsed_sentences: list[ParsedSentence],
        citation_index: list[CitationIndexEntry],
    ) -> CoverageCheckResult:
        facet_chunk_ids: dict[str, set[str]] = {facet: set() for facet in facets}
        for citation in citation_index:
            for facet in citation.source_facets or [citation.source_facet]:
                if facet in facet_chunk_ids:
                    facet_chunk_ids[facet].add(citation.chunk_id)

        resolved_chunk_ids: set[str] = set()
        for sentence in parsed_sentences:
            for reference in sentence.references:
                if reference.verified and reference.matched_chunk_id:
                    resolved_chunk_ids.add(reference.matched_chunk_id)

        covered: list[str] = []
        gap: list[str] = []
        for facet in facets:
            candidate_ids = facet_chunk_ids.get(facet, set())
            if candidate_ids and resolved_chunk_ids.intersection(candidate_ids):
                covered.append(facet)
            else:
                gap.append(facet)
        return CoverageCheckResult(
            covered_facets=covered,
            gap_facets=gap,
            trigger_followup=bool(gap),
        )

    async def score_confidence(self, parsed_sentences: list[ParsedSentence]) -> list[ParsedSentence]:
        return await self.score_confidence_for_sentences(parsed_sentences)

    async def score_confidence_for_sentences(
        self,
        parsed_sentences: list[ParsedSentence],
        *,
        sentence_indices: set[int] | None = None,
    ) -> list[ParsedSentence]:
        logger.info(
            "confidence scoring started",
            extra={
                "sentence_count": len(parsed_sentences),
                "target_sentence_count": (
                    len(sentence_indices) if sentence_indices is not None else len(parsed_sentences)
                ),
                "console_visible": False,
            },
        )
        targets = (
            parsed_sentences
            if sentence_indices is None
            else [sentence for sentence in parsed_sentences if sentence.sentence_index in sentence_indices]
        )
        for sentence in targets:
            if sentence.status in {SentenceStatus.NO_REF, SentenceStatus.UNGROUNDED}:
                continue
            if sentence.sentence_type == SentenceType.STRUCTURE:
                sentence.status = SentenceStatus.UNCHECKED
                continue

            verified_references = [reference for reference in sentence.references if reference.verified and reference.matched_chunk_id]
            if not verified_references:
                sentence.status = SentenceStatus.UNGROUNDED
                continue

            scored_payloads: list[tuple] = []
            uncached_keys: list[tuple[str, str]] = []
            uncached_texts: list[str] = []
            for reference in verified_references:
                chunk = self.chunk_store.get_chunk(reference.matched_chunk_id)
                if chunk is not None:
                    cache_key = (normalize_whitespace(sentence.sentence_text), chunk.chunk_id)
                    scored_payloads.append((reference, cache_key, chunk.text))
                    cached_score = self._confidence_cache_get(cache_key)
                    if cached_score is None:
                        uncached_keys.append(cache_key)
                        uncached_texts.append(chunk.text)

            if uncached_texts:
                scores = await self.nli_client.score(sentence.sentence_text, uncached_texts)
                for cache_key, score in zip(uncached_keys, scores):
                    self._confidence_cache_put(cache_key, score)

            for reference, cache_key, _chunk_text in scored_payloads:
                score = self._confidence_cache_get(cache_key)
                if score is None:
                    continue
                reference.confidence_score = score.score
                reference.confidence_label = score.label
                reference.confidence_unknown = score.label is None
                if score.label is None and ReviewWarning.CONFIDENCE_UNKNOWN not in sentence.warnings:
                    sentence.warnings.append(ReviewWarning.CONFIDENCE_UNKNOWN)

            sentence.status = self._derive_final_status(sentence)
            logger.info(
                "sentence confidence result",
                extra={
                    "sentence_index": sentence.sentence_index,
                    "sentence_type": sentence.sentence_type.value,
                    "verified_reference_count": len(verified_references),
                    "status": sentence.status.value,
                    "confidence_labels": [reference.confidence_label.value if reference.confidence_label else None for reference in verified_references],
                    "console_visible": False,
                },
            )
        logger.info("confidence scoring finished", extra={"sentence_status_summary": _sentence_status_summary(parsed_sentences)})
        return parsed_sentences

    def _confidence_cache_get(self, key: tuple[str, str]) -> ScoredReference | None:
        cached = self._confidence_cache.get(key)
        if cached is None:
            return None
        self._confidence_cache.move_to_end(key)
        return cached.model_copy()

    def _confidence_cache_put(self, key: tuple[str, str], value: ScoredReference) -> None:
        self._confidence_cache[key] = value.model_copy()
        self._confidence_cache.move_to_end(key)
        while len(self._confidence_cache) > self.CONFIDENCE_CACHE_MAX_ENTRIES:
            self._confidence_cache.popitem(last=False)

    def _derive_final_status(self, sentence: ParsedSentence) -> SentenceStatus:
        if sentence.status == SentenceStatus.NO_REF:
            return SentenceStatus.NO_REF
        if sentence.sentence_type == SentenceType.STRUCTURE:
            return SentenceStatus.UNCHECKED

        verified_references = [reference for reference in sentence.references if reference.verified]
        if not verified_references:
            return SentenceStatus.UNGROUNDED

        if sentence.sentence_type == SentenceType.CLAIM:
            if len(verified_references) != len(sentence.references):
                return SentenceStatus.UNGROUNDED
            labels = {reference.confidence_label for reference in verified_references}
            if labels == {None}:
                return SentenceStatus.VERIFIED
            if labels <= {ConfidenceLabel.SUPPORTED, None}:
                return SentenceStatus.VERIFIED
            return SentenceStatus.PARTIALLY_VERIFIED

        failed_refs = [reference for reference in sentence.references if not reference.verified]
        if failed_refs:
            return SentenceStatus.PARTIALLY_VERIFIED
        if any(reference.confidence_label not in {None, ConfidenceLabel.SUPPORTED} for reference in verified_references):
            return SentenceStatus.PARTIALLY_VERIFIED
        return SentenceStatus.VERIFIED


def _sentence_status_summary(parsed_sentences: list[ParsedSentence]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for sentence in parsed_sentences:
        summary[sentence.status.value] = summary.get(sentence.status.value, 0) + 1
    return summary
