from __future__ import annotations

from quarry.adapters.in_memory import extract_exact_quote
from quarry.adapters.interfaces import GenerationClient
from quarry.domain.models import CitationIndexEntry, GenerationRequest, ParsedSentence, SentenceType


class AnswerGenerator:
    def __init__(self, client: GenerationClient) -> None:
        self.client = client

    async def generate(self, request: GenerationRequest) -> str:
        return await self.client.generate(request)


class SentenceRegenerator:
    def regenerate(
        self,
        sentence: ParsedSentence,
        citation_index: list[CitationIndexEntry],
        *,
        failed_regeneration_response: str | None = None,
    ) -> GenerationRequest:
        return GenerationRequest(
            original_query=sentence.sentence_text,
            facets=[],
            citation_index=self._rank_citations(sentence, citation_index),
            mode="regeneration",
            failed_sentence_text=sentence.sentence_text,
            failed_regeneration_response=failed_regeneration_response,
        )

    def deterministic_rewrite(self, sentence: ParsedSentence, citation_index: list[CitationIndexEntry]) -> str:
        ranked_citations = self._rank_citations(sentence, citation_index)
        if not ranked_citations:
            return f"[{sentence.sentence_type.value.upper()}] {sentence.sentence_text} [NO_REF]"

        if sentence.sentence_type == SentenceType.SYNTHESIS:
            selected = [citation for citation in ranked_citations[:2] if len(extract_exact_quote(citation.text).split()) >= 15]
            if len(selected) < 2:
                return f"[SYNTHESIS] {sentence.sentence_text} [NO_REF]"
            first_quote = extract_exact_quote(selected[0].text).replace('"', "'")
            second_quote = extract_exact_quote(selected[1].text).replace('"', "'")
            return (
                "[SYNTHESIS] Taken together, the available passages support a cross-section reading of the evidence. "
                f'[REF: "{first_quote}"] '
                f'[REF: "{second_quote}"]'
            )

        best = ranked_citations[0]
        quote = extract_exact_quote(best.text).replace('"', "'")
        if len(quote.split()) < 15:
            return f"[CLAIM] {sentence.sentence_text} [NO_REF]"
        claim_text = quote[0].upper() + quote[1:]
        if not claim_text.endswith("."):
            claim_text += "."
        return f'[CLAIM] {claim_text} [REF: "{quote}"]'

    def _rank_citations(self, sentence: ParsedSentence, citation_index: list[CitationIndexEntry]) -> list[CitationIndexEntry]:
        return sorted(
            citation_index,
            key=lambda citation: self._overlap_score(sentence.sentence_text, citation.text),
            reverse=True,
        )

    def _overlap_score(self, sentence_text: str, chunk_text: str) -> int:
        sentence_terms = {token.lower() for token in sentence_text.split()}
        chunk_terms = {token.lower() for token in chunk_text.split()}
        return len(sentence_terms & chunk_terms)
