from __future__ import annotations

import re

from quarry.adapters.interfaces import GenerationClient
from quarry.domain.models import CitationIndexEntry, GenerationRequest, ParsedSentence, SentenceType

INLINE_REF_PATTERN = re.compile(r"\[(?:REF:[^\]]+|NO_REF)\]")


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

    def deterministic_rewrite(self, sentence: ParsedSentence, _citation_index: list[CitationIndexEntry]) -> str:
        fallback_text = self._clean_sentence_text(sentence.sentence_text)
        if not fallback_text:
            fallback_text = "This sentence could not be grounded in the available evidence."
        return f"[{sentence.sentence_type.value.upper()}] {fallback_text} [NO_REF]"

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

    def _clean_sentence_text(self, text: str) -> str:
        cleaned = INLINE_REF_PATTERN.sub("", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
