from __future__ import annotations

import re

from quarry.domain.models import ParsedSentence, Reference, ReviewWarning, SentenceStatus, SentenceType


TAG_PATTERN = re.compile(r"\[(CLAIM|SYNTHESIS|STRUCTURE)\]")
REF_PATTERN = re.compile(r"\[REF:\s*(?:\"([^\"]+)\"|'([^']+)')\s*\]")
NO_REF_PATTERN = re.compile(r"\[NO_REF\]")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%?\b")
PROPER_NOUN_PATTERN = re.compile(r"\b[A-Z][a-z]{2,}\b")
DOMAIN_TERMS = {"schedule", "cost", "risk", "safety", "procurement", "phase", "modular", "project", "construction"}


def parse_generated_response(raw_response: str) -> list[ParsedSentence]:
    tagged_blocks = _split_tagged_blocks(raw_response)
    if not tagged_blocks:
        return _fallback_parse(raw_response)

    parsed_sentences: list[ParsedSentence] = []
    for index, block in enumerate(tagged_blocks):
        tag_match = TAG_PATTERN.search(block)
        if not tag_match:
            continue
        tag = tag_match.group(1)
        refs = [match[0] or match[1] for match in REF_PATTERN.findall(block)]
        no_ref = bool(NO_REF_PATTERN.search(block))

        cleaned_text = TAG_PATTERN.sub("", block, count=1)
        cleaned_text = REF_PATTERN.sub("", cleaned_text)
        cleaned_text = NO_REF_PATTERN.sub("", cleaned_text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        sentence_type = _normalize_sentence_type(SentenceType(tag.lower()), len(refs))
        warnings: list[ReviewWarning] = []
        if sentence_type == SentenceType.STRUCTURE and _has_factual_tokens(cleaned_text):
            warnings.append(ReviewWarning.STRUCTURAL_FACT)
        if len(refs) > 5:
            warnings.append(ReviewWarning.OVER_CITED)

        parsed_sentences.append(
            ParsedSentence(
                sentence_index=index,
                sentence_text=cleaned_text,
                sentence_type=sentence_type,
                references=[Reference(reference_quote=quote) for quote in refs],
                status=SentenceStatus.NO_REF if no_ref else SentenceStatus.UNCHECKED,
                warnings=warnings,
                raw_text=block.strip(),
            )
        )
    return parsed_sentences


def render_parsed_sentences(parsed_sentences: list[ParsedSentence]) -> str:
    rendered: list[str] = []
    for sentence in parsed_sentences:
        tag = f"[{sentence.sentence_type.value.upper()}]"
        body = f"{tag} {sentence.sentence_text}".strip()
        if sentence.status == SentenceStatus.NO_REF:
            body = f"{body} [NO_REF]"
        else:
            for reference in sentence.references:
                body = f'{body} [REF: "{reference.reference_quote.replace(chr(34), chr(39))}"]'
        rendered.append(body.strip())
    return "\n\n".join(rendered)


def _split_tagged_blocks(raw_response: str) -> list[str]:
    matches = list(TAG_PATTERN.finditer(raw_response))
    if not matches:
        return []
    blocks: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_response)
        blocks.append(raw_response[match.start() : end].strip())
    return blocks


def _fallback_parse(raw_response: str) -> list[ParsedSentence]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", raw_response.strip()) if sentence.strip()]
    parsed: list[ParsedSentence] = []
    for index, sentence in enumerate(sentences):
        refs = [match[0] or match[1] for match in REF_PATTERN.findall(sentence)]
        sentence_type = SentenceType.CLAIM if refs else SentenceType.STRUCTURE
        warnings: list[ReviewWarning] = []
        if sentence_type == SentenceType.STRUCTURE and _has_factual_tokens(sentence):
            warnings.append(ReviewWarning.STRUCTURAL_FACT)
        if len(refs) > 5:
            warnings.append(ReviewWarning.OVER_CITED)
        parsed.append(
            ParsedSentence(
                sentence_index=index,
                sentence_text=REF_PATTERN.sub("", sentence).strip(),
                sentence_type=sentence_type,
                references=[Reference(reference_quote=quote) for quote in refs],
                status=SentenceStatus.UNCHECKED,
                warnings=warnings,
                raw_text=sentence,
            )
        )
    return parsed


def _normalize_sentence_type(initial: SentenceType, ref_count: int) -> SentenceType:
    if ref_count == 0:
        return SentenceType.STRUCTURE if initial == SentenceType.STRUCTURE else SentenceType.CLAIM
    if ref_count == 1:
        return SentenceType.CLAIM
    return SentenceType.SYNTHESIS


def _has_factual_tokens(sentence_text: str) -> bool:
    if NUMBER_PATTERN.search(sentence_text):
        return True
    tokens = sentence_text.split()
    if any(term in sentence_text.lower() for term in DOMAIN_TERMS):
        return True
    proper_nouns = PROPER_NOUN_PATTERN.findall(sentence_text)
    return len(proper_nouns) > 1 or (len(tokens) > 1 and any(token in proper_nouns for token in tokens[1:]))
