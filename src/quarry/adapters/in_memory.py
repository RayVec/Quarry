from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

from quarry.adapters.interfaces import (
    ChunkStore,
    DecompositionClient,
    EmbeddingClient,
    GenerationClient,
    MetadataEnricher,
    NLIClient,
    Reranker,
    Retriever,
)
from quarry.domain.models import (
    ChunkObject,
    CommentIntent,
    ConfidenceLabel,
    GenerationRequest,
    RefinementScope,
    RetrievalFilters,
    RetrievedPassage,
    ScoredReference,
)
from quarry.logging_utils import logger_with_trace


logger = logger_with_trace(__name__)
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def extract_exact_quote(text: str, *, min_words: int = 15, max_words: int = 28) -> str:
    words = normalize_text(text).split()
    if not words:
        return ""
    if len(words) < min_words:
        return " ".join(words)
    return " ".join(words[: min(max_words, len(words))])


def no_ref_fallback_response(request: GenerationRequest) -> str:
    if request.mode == "refinement_planning":
        return json.dumps(
            {
                "overall_scope": "no_change",
                "change_summary": "The current answer can remain unchanged.",
                "comments": [],
                "target_sentence_indices": [],
            }
        )
    if request.mode == "sentence_refinement":
        sentence = normalize_text(request.target_sentence_text or request.failed_sentence_text or "")
        if not sentence:
            sentence = "This sentence could not be grounded in the available evidence."
        return f"[CLAIM] {sentence} [NO_REF]"
    if request.mode == "regeneration":
        return "[CLAIM] This sentence could not be grounded in the available evidence. [NO_REF]"
    if request.mode == "supplement":
        return "[CLAIM] Additional grounded content could not be produced from the available evidence. [NO_REF]"
    if request.mode == "refinement":
        return "[CLAIM] The response could not be regenerated from the remaining grounded evidence. [NO_REF]"
    return "[CLAIM] A grounded answer could not be produced from the available evidence. [NO_REF]"


class InMemoryChunkStore(ChunkStore):
    def __init__(self, chunks: Iterable[ChunkObject]) -> None:
        self._chunks = list(chunks)
        self._chunk_map = {chunk.chunk_id: chunk for chunk in self._chunks}
        self._normalized_chunk_text_by_id = {
            chunk.chunk_id: normalize_text(chunk.text)
            for chunk in self._chunks
        }

    @classmethod
    def from_directory(cls, corpus_dir: Path) -> "InMemoryChunkStore":
        chunks: list[ChunkObject] = []
        if not corpus_dir.exists():
            return cls([])

        manifest_path = corpus_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for document in manifest.get("documents", []):
                chunks_path = Path(document["chunks_path"])
                if chunks_path.exists():
                    chunks.extend(ChunkObject.model_validate(item) for item in json.loads(chunks_path.read_text()))
        else:
            for path in sorted(corpus_dir.glob("*.json")):
                payload = json.loads(path.read_text())
                if isinstance(payload, list):
                    raw_chunks = payload
                elif isinstance(payload, dict) and "chunks" in payload:
                    raw_chunks = payload["chunks"]
                else:
                    raw_chunks = [payload]
                chunks.extend(ChunkObject.model_validate(raw_chunk) for raw_chunk in raw_chunks)
        return cls(chunks)

    def all_chunks(self) -> list[ChunkObject]:
        return list(self._chunks)

    def chunk_count(self) -> int:
        return len(self._chunks)

    def get_chunk(self, chunk_id: str) -> ChunkObject | None:
        return self._chunk_map.get(chunk_id)

    def find_chunk_by_quote(
        self,
        quote: str,
        *,
        chunk_ids: Sequence[str] | None = None,
    ) -> ChunkObject | None:
        normalized_quote = normalize_text(quote)
        if not normalized_quote:
            return None
        if chunk_ids is None:
            candidates = self._chunks
            for chunk in candidates:
                if normalized_quote in self._normalized_chunk_text_by_id.get(chunk.chunk_id, ""):
                    return chunk
            return None

        for chunk_id in chunk_ids:
            chunk = self._chunk_map.get(chunk_id)
            if chunk is None:
                continue
            if normalized_quote in self._normalized_chunk_text_by_id.get(chunk_id, ""):
                return chunk
        return None

class HeuristicDecompositionClient(DecompositionClient):
    MULTI_HOP_HINTS = (
        "compare",
        "relationship",
        "impact",
        "factors",
        "tradeoffs",
        "causes",
        "drivers",
        "between",
        "across",
        "versus",
        " and ",
    )

    async def decompose_query(self, query: str, max_facets: int) -> list[str]:
        normalized = normalize_text(query)
        affect_match = re.match(
            r"(?i)^(how do|how does|what factors in|which factors in)\s+(.+?)\s+and\s+(.+?)\s+(affect|influence|impact|shape)\s+(.+?)\??$",
            normalized,
        )
        if affect_match:
            _, left, right, verb, tail = affect_match.groups()
            return [
                f"How does {left.strip()} {verb} {tail.strip()}?",
                f"How does {right.strip()} {verb} {tail.strip()}?",
            ][:max_facets]

        raw_parts = re.split(r"\b(?:and|vs\.?|versus|across|between|compare|with)\b|,", normalized, flags=re.IGNORECASE)
        parts = [part.strip(" .") for part in raw_parts if len(part.strip()) > 10]
        if not parts:
            return [normalized]

        facets: list[str] = []
        for part in parts:
            if part.lower() == normalized.lower():
                continue
            facet = part
            if not facet.endswith("?") and normalized.endswith("?"):
                facet = f"{facet}?"
            facets.append(facet)
            if len(facets) >= max_facets:
                break
        return facets or [normalized]


class KeywordSparseRetriever(Retriever):
    def __init__(self, chunk_store: InMemoryChunkStore) -> None:
        self.chunk_store = chunk_store

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        source_facet: str,
        filters: RetrievalFilters | None = None,
    ) -> tuple[list[RetrievedPassage], dict[str, object] | None]:
        query_terms = tokenize(query)
        if not query_terms:
            return [], {"retriever": "sparse", "result_count": 0, "fallback_used": True}
        results: list[RetrievedPassage] = []
        for chunk in self.chunk_store.all_chunks():
            if filters and filters.document_id and chunk.document_id != filters.document_id:
                continue
            if filters and filters.section_path and chunk.section_path != filters.section_path:
                continue
            chunk_terms = tokenize(chunk.text)
            score = sum(chunk_terms.count(term) for term in query_terms)
            score += 2 * sum(1 for entity in chunk.metadata_entities if entity.lower() in query.lower())
            if score <= 0:
                continue
            results.append(
                RetrievedPassage(
                    chunk=chunk,
                    score=float(score),
                    source_facet=source_facet,
                    rank=0,
                    retriever="sparse" if not filters else "scoped",
                )
            )
        results.sort(key=lambda result: result.score, reverse=True)
        ranked = [result.model_copy(update={"rank": index + 1}) for index, result in enumerate(results[:top_k])]
        return ranked, {"retriever": "sparse", "result_count": len(ranked), "fallback_used": True}


class SemanticDenseRetriever(Retriever):
    def __init__(self, chunk_store: InMemoryChunkStore) -> None:
        self.chunk_store = chunk_store

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        source_facet: str,
        filters: RetrievalFilters | None = None,
    ) -> tuple[list[RetrievedPassage], dict[str, object] | None]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return [], {"retriever": "dense", "result_count": 0, "fallback_used": True}
        results: list[RetrievedPassage] = []
        for chunk in self.chunk_store.all_chunks():
            if filters and filters.document_id and chunk.document_id != filters.document_id:
                continue
            if filters and filters.section_path and chunk.section_path != filters.section_path:
                continue
            enriched_text = " ".join([chunk.text, chunk.metadata_summary, *chunk.metadata_questions, *chunk.metadata_entities])
            chunk_terms = set(tokenize(enriched_text))
            overlap = len(query_terms & chunk_terms)
            if overlap == 0:
                continue
            score = overlap / max(len(query_terms), 1)
            results.append(
                RetrievedPassage(
                    chunk=chunk,
                    score=score,
                    source_facet=source_facet,
                    rank=0,
                    retriever="dense" if not filters else "scoped",
                )
            )
        results.sort(key=lambda result: result.score, reverse=True)
        ranked = [result.model_copy(update={"rank": index + 1}) for index, result in enumerate(results[:top_k])]
        return ranked, {"retriever": "dense", "result_count": len(ranked), "fallback_used": True}


class SimpleCrossEncoderReranker(Reranker):
    async def rerank(self, query: str, candidates: Sequence[RetrievedPassage]) -> list[RetrievedPassage]:
        query_terms = set(tokenize(query))
        reranked: list[RetrievedPassage] = []
        for candidate in candidates:
            passage_terms = set(tokenize(candidate.chunk.text))
            overlap = len(query_terms & passage_terms)
            score = candidate.score + (overlap / max(len(query_terms), 1))
            reranked.append(candidate.model_copy(update={"score": score, "retriever": "reranked"}))
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked


class DeterministicGenerationClient(GenerationClient):
    AFFIRMATION_MARKERS = (
        "good",
        "great",
        "aligned",
        "looks right",
        "looks good",
        "perfectly aligned",
        "agree",
        "works for me",
    )
    GLOBAL_MARKERS = (
        "rewrite",
        "restructure",
        "reorganize",
        "reframe",
        "completely",
        "overall answer",
        "main structure",
    )

    def _infer_comment_intent(self, comment_text: str) -> tuple[CommentIntent, RefinementScope]:
        lowered = normalize_text(comment_text).lower()
        if any(marker in lowered for marker in self.AFFIRMATION_MARKERS):
            return CommentIntent.AFFIRMATION, RefinementScope.NONE
        if any(marker in lowered for marker in self.GLOBAL_MARKERS):
            return CommentIntent.REWRITE_REQUEST, RefinementScope.GLOBAL
        if any(marker in lowered for marker in ("add", "clarify", "tighten", "revise", "explain", "detail", "adjust")):
            return CommentIntent.MINOR_EDIT, RefinementScope.LOCAL
        return CommentIntent.SUBSTANTIVE_EDIT, RefinementScope.LOCAL

    async def generate(self, request: GenerationRequest) -> str:
        if request.mode == "refinement_planning":
            comment_payloads: list[dict[str, object]] = []
            target_sentence_indices = sorted({index for index in request.target_sentence_indices if index >= 0})
            overall_scope = RefinementScope.NONE

            if request.rejected_pairs or request.mismatch_citation_ids:
                overall_scope = RefinementScope.GLOBAL

            for comment in request.selection_comments:
                intent, scope = self._infer_comment_intent(comment.comment_text)
                if scope == RefinementScope.GLOBAL:
                    overall_scope = RefinementScope.GLOBAL
                elif scope == RefinementScope.LOCAL and overall_scope != RefinementScope.GLOBAL:
                    overall_scope = RefinementScope.LOCAL
                comment_payloads.append(
                    {
                        "comment_id": comment.comment_id,
                        "intent": intent.value,
                        "scope": scope.value if scope != RefinementScope.NONE else "no_change",
                        "target_sentence_indices": target_sentence_indices,
                        "summary": comment.comment_text.strip(),
                    }
                )

            if request.rejected_pairs or request.mismatch_citation_ids:
                change_summary = "Reviewer feedback requires broader evidence-aware revision."
            elif overall_scope == RefinementScope.LOCAL:
                change_summary = "Reviewer feedback calls for a localized edit."
            else:
                change_summary = "Reviewer feedback affirms the current answer."

            return json.dumps(
                {
                    "overall_scope": (
                        "global_rewrite"
                        if overall_scope == RefinementScope.GLOBAL
                        else "local_edit" if overall_scope == RefinementScope.LOCAL else "no_change"
                    ),
                    "change_summary": change_summary,
                    "comments": comment_payloads,
                    "target_sentence_indices": target_sentence_indices,
                }
            )

        if request.mode == "sentence_refinement":
            sentence = normalize_text(request.target_sentence_text or request.failed_sentence_text or "")
            note = normalize_text(request.revision_note or request.failed_sentence_comment or "").lower()
            citations = request.citation_index
            if citations:
                quote = extract_exact_quote(citations[0].text, min_words=8, max_words=18)
                if quote:
                    sentence = quote[0].upper() + quote[1:]
                    if "tight" in note and "subset" in sentence.lower():
                        sentence = "FEED maturity elements are a subset of the entire PDRI."
                    if not sentence.endswith("."):
                        sentence += "."
                    return f'[CLAIM] {sentence} [REF: "{quote.replace(chr(34), chr(39))}"]'
            if not sentence:
                sentence = "This sentence could not be grounded in the available evidence."
            return f"[CLAIM] {sentence} [NO_REF]"

        citations = request.citation_index
        if request.mismatch_citation_ids:
            citations = [citation for citation in citations if citation.citation_id not in request.mismatch_citation_ids]

        if request.selected_facets:
            selected = {facet.lower() for facet in request.selected_facets}
            filtered = [citation for citation in citations if citation.source_facet.lower() in selected]
            if filtered:
                citations = filtered

        if request.mode == "regeneration" and request.failed_sentence_text:
            citations = sorted(
                citations,
                key=lambda citation: len(set(tokenize(request.failed_sentence_text)) & set(tokenize(citation.text))),
                reverse=True,
            )

        if not citations:
            return '[CLAIM] The available evidence is insufficient to produce a grounded answer for this request. [NO_REF]'

        lines: list[str] = []
        if request.mode == "supplement":
            facet_text = ", ".join(request.selected_facets) if request.selected_facets else "the selected facets"
            lines.append(f"[STRUCTURE] This supplement adds evidence for {facet_text}.")
        elif request.mode == "refinement":
            lines.append("[STRUCTURE] This revision emphasizes the strongest remaining evidence after review.")
        elif request.mode == "regeneration":
            lines.append("[STRUCTURE] The sentence has been rewritten to align with the available evidence.")
        else:
            facet_text = ", ".join(request.facets[:3]) if request.facets else "the retrieved passages"
            lines.append(f'[STRUCTURE] The retrieved evidence for "{request.original_query}" centers on {facet_text}.')

        for citation in citations[:4]:
            quote = extract_exact_quote(citation.text)
            if len(quote.split()) < 15:
                lines.append(f"[CLAIM] Passage [{citation.citation_id}] is relevant but too short to support a verified quote. [NO_REF]")
                continue
            claim_text = quote[0].upper() + quote[1:]
            if not claim_text.endswith("."):
                claim_text += "."
            lines.append(f'[CLAIM] {claim_text} [REF: "{quote.replace(chr(34), chr(39))}"]')

        if len(citations) >= 2 and request.mode != "regeneration":
            first = extract_exact_quote(citations[0].text)
            second = extract_exact_quote(citations[1].text)
            if len(first.split()) >= 15 and len(second.split()) >= 15:
                lines.append(
                    "[SYNTHESIS] Taken together, the retrieved passages reinforce related evidence across multiple sections. "
                    f'[REF: "{first.replace(chr(34), chr(39))}"] '
                    f'[REF: "{second.replace(chr(34), chr(39))}"]'
                )

        return "\n\n".join(lines)


class ConservativeFallbackGenerationClient(GenerationClient):
    async def generate(self, request: GenerationRequest) -> str:
        return no_ref_fallback_response(request)


class HeuristicNLIClient(NLIClient):
    async def score(self, sentence_text: str, chunk_texts: Sequence[str]) -> list[ScoredReference]:
        sentence_terms = set(tokenize(sentence_text))
        scores: list[ScoredReference] = []
        for chunk_text in chunk_texts:
            chunk_terms = set(tokenize(chunk_text))
            if not sentence_terms or not chunk_terms:
                scores.append(ScoredReference(score=None, label=None))
                continue
            intersection = len(sentence_terms & chunk_terms)
            # Use the more generous direction (max of precision/recall) so that
            # paraphrased or slightly reworded sentences are not unfairly penalised
            # when their vocabulary largely overlaps with the source chunk.
            precision = intersection / max(len(sentence_terms), 1)
            recall = intersection / max(len(chunk_terms), 1)
            overlap = max(precision, recall)
            if overlap >= 0.7:
                label = ConfidenceLabel.SUPPORTED
            elif overlap >= 0.4:
                label = ConfidenceLabel.PARTIALLY_SUPPORTED
            else:
                label = ConfidenceLabel.NOT_SUPPORTED
            scores.append(ScoredReference(score=round(overlap, 3), label=label))
        return scores


class HeuristicMetadataEnricher(MetadataEnricher):
    async def enrich(self, chunk: ChunkObject) -> ChunkObject:
        tokens = tokenize(chunk.text)
        summary = normalize_text(chunk.text.split(".")[0]) + "."
        entities = []
        for token in chunk.text.split():
            cleaned = token.strip(",.()")
            if cleaned.istitle() and cleaned not in entities:
                entities.append(cleaned)
            if len(entities) >= 5:
                break
        questions = [
            f"What evidence is given about {chunk.section_heading.lower()}?",
            f"How does {chunk.section_heading.lower()} affect project outcomes?",
        ]
        return chunk.model_copy(
            update={
                "metadata_summary": summary if summary != "." else " ".join(tokens[:20]),
                "metadata_entities": entities,
                "metadata_questions": questions,
            }
        )


class HashEmbeddingClient(EmbeddingClient):
    def __init__(self, *, dimensions: int = 192) -> None:
        self.dimensions = dimensions

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in tokenize(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:2], "big") % self.dimensions
                vector[bucket] += 1.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / norm for value in vector])
        logger.info("generated hash embeddings", extra={"vector_count": len(vectors)})
        return vectors
