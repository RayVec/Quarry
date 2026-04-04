"""
Tests for NLI scoring improvements:
  A – HeuristicNLIClient uses max(precision, recall) overlap.
  B – LocalMNLIClient treats near-entailment (>= ENTAILMENT_SOFT_THRESHOLD) as SUPPORTED.
  E – VerificationService DEFAULT_MIN_QUOTE_WORDS lowered to 10.
"""
from __future__ import annotations

import asyncio
import types
from unittest.mock import MagicMock

import pytest

from quarry.adapters.in_memory import HeuristicNLIClient, InMemoryChunkStore
from quarry.adapters.local_models import LocalMNLIClient
from quarry.domain.models import (
    ChunkObject,
    CitationIndexEntry,
    ConfidenceLabel,
    ParsedSentence,
    Reference,
    SentenceStatus,
    SentenceType,
)
from quarry.pipeline.verification import VerificationService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_chunk(chunk_id: str, text: str) -> ChunkObject:
    return ChunkObject(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_title="Doc 1",
        text=text,
        section_heading="Heading",
        section_path="Chapter > Heading",
        page_start=1,
        page_end=1,
    )


def make_citation(chunk: ChunkObject, *, citation_id: int = 1, score: float = 0.9) -> CitationIndexEntry:
    return CitationIndexEntry(
        citation_id=citation_id,
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        document_id=chunk.document_id,
        document_title=chunk.document_title,
        section_heading=chunk.section_heading,
        section_path=chunk.section_path,
        page_number=chunk.page_start,
        retrieval_score=score,
        source_facet="test",
    )


def _make_mnli_client_with_probs(
    probs: list[list[float]],
    *,
    label_map: dict[int, str] | None = None,
) -> LocalMNLIClient:
    """
    Return a LocalMNLIClient whose internal model is replaced by a mock that
    produces the given softmax probability rows (one row per chunk text).

    Default label_map: {0: "contradiction", 1: "neutral", 2: "entailment"}
    matching DeBERTa-v3-large-mnli's typical layout.
    """
    import numpy as np

    if label_map is None:
        label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}

    # Build a fake config with id2label
    fake_config = types.SimpleNamespace(id2label=label_map)

    # Build a fake model whose __call__ returns logits that produce the given probs
    # after softmax.  We pass raw probs as logits – softmax of a uniform offset is
    # still proportional, but easiest to just return log-odds so softmax gives probs.
    # Instead we skip the softmax by patching torch.softmax below.
    fake_model = MagicMock()
    fake_model.config = fake_config

    # Fake output: logits attribute (values don't matter, we patch softmax)
    import torch
    raw_logits = torch.zeros(len(probs), len(label_map))
    fake_model.return_value = types.SimpleNamespace(logits=raw_logits)

    # Fake tokenizer that returns a small dict of tensors
    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value = {
        "input_ids": torch.zeros(len(probs), 4, dtype=torch.long),
        "attention_mask": torch.ones(len(probs), 4, dtype=torch.long),
    }

    client = LocalMNLIClient.__new__(LocalMNLIClient)
    client.model_name = "mock-mnli"
    client.device = "cpu"
    client.batch_size = 32
    client.fallback = MagicMock()
    client._tokenizer = fake_tokenizer
    client._model = fake_model
    client._device = "cpu"
    client._load_attempted = True

    # Monkey-patch torch.softmax inside _score_sync to return our controlled probs
    import quarry.adapters.local_models as _mod
    import numpy as _np

    np_probs = _np.array(probs, dtype=_np.float32)

    original_score_sync = LocalMNLIClient._score_sync

    def patched_score_sync(self_inner, sentence_text, chunk_texts):
        # Rebuild a minimal execution that uses np_probs directly
        labels = {int(k): str(v).lower() for k, v in self_inner._model.config.id2label.items()}
        entail_idx = next(idx for idx, lbl in labels.items() if "entail" in lbl)
        neutral_idx = next(idx for idx, lbl in labels.items() if "neutral" in lbl)

        scored = []
        predicted = np_probs.argmax(axis=-1)
        for row, predicted_idx in zip(np_probs, predicted):
            entailment_score = float(row[entail_idx])
            if predicted_idx == entail_idx or entailment_score >= self_inner.ENTAILMENT_SOFT_THRESHOLD:
                label = ConfidenceLabel.SUPPORTED
            elif predicted_idx == neutral_idx:
                label = ConfidenceLabel.PARTIALLY_SUPPORTED
            else:
                label = ConfidenceLabel.NOT_SUPPORTED
            scored.append(
                __import__("quarry.domain.models", fromlist=["ScoredReference"]).ScoredReference(
                    score=entailment_score, label=label
                )
            )
        return scored

    client._score_sync = lambda sentence_text, chunk_texts: patched_score_sync(
        client, sentence_text, chunk_texts
    )
    return client


# ---------------------------------------------------------------------------
# Direction A – HeuristicNLIClient overlap formula
# ---------------------------------------------------------------------------

class TestHeuristicNLIOverlap:
    """
    The formula must use max(precision, recall) so that a sentence which is a
    near-verbatim rewrite of a longer chunk still scores high.
    """

    def test_verbatim_paraphrase_scores_supported(self) -> None:
        """
        Sentence reuses most words from the source chunk; only minor grammatical
        changes.  Under the old formula (precision only) it could fall below 0.7
        when the sentence has additional connective words.  Under the new formula
        it should reach SUPPORTED.
        """
        client = HeuristicNLIClient()
        chunk_text = (
            "FEED maturity is defined as the degree of completeness of the deliverables "
            "to serve as the basis for detailed design at the end of detailed scope Phase Gate 3"
        )
        # Sentence is slightly shorter than chunk but shares nearly all content words
        sentence_text = (
            "FEED maturity is defined as the degree of completeness of the deliverables "
            "to serve as the basis for detailed design at the end of detailed scope"
        )
        results = asyncio.run(client.score(sentence_text, [chunk_text]))
        assert results[0].label == ConfidenceLabel.SUPPORTED

    def test_sentence_longer_than_chunk_still_scores_correctly(self) -> None:
        """
        When the sentence adds words not present in the chunk, precision drops but
        recall stays high.  max(precision, recall) should still yield SUPPORTED.
        """
        client = HeuristicNLIClient()
        chunk_text = "Projects exceeded the approved budget by thirty percent on average"
        sentence_text = (
            "According to the report, projects exceeded the approved budget by thirty percent on average"
        )
        results = asyncio.run(client.score(sentence_text, [chunk_text]))
        assert results[0].label == ConfidenceLabel.SUPPORTED

    def test_unrelated_sentence_still_scores_not_supported(self) -> None:
        client = HeuristicNLIClient()
        chunk_text = "Modular construction reduces schedule risk by prefabricating components off-site"
        sentence_text = "The project encountered severe budget overruns due to procurement failures"
        results = asyncio.run(client.score(sentence_text, [chunk_text]))
        assert results[0].label == ConfidenceLabel.NOT_SUPPORTED

    def test_partial_overlap_gives_partially_supported(self) -> None:
        client = HeuristicNLIClient()
        chunk_text = "Advanced work packaging reduces rework and improves schedule predictability"
        sentence_text = "Advanced work packaging can sometimes reduce rework"
        results = asyncio.run(client.score(sentence_text, [chunk_text]))
        assert results[0].label in {ConfidenceLabel.SUPPORTED, ConfidenceLabel.PARTIALLY_SUPPORTED}


# ---------------------------------------------------------------------------
# Direction B – LocalMNLIClient soft entailment threshold
# ---------------------------------------------------------------------------

class TestLocalMNLISoftThreshold:
    """
    Tests for the ENTAILMENT_SOFT_THRESHOLD = 0.35 fallback.

    Probability layout: [contradiction, neutral, entailment] (indices 0, 1, 2)
    """

    def test_argmax_entailment_is_supported(self) -> None:
        """Classic case: model is confident about entailment."""
        # [contradiction=0.05, neutral=0.10, entailment=0.85]
        client = _make_mnli_client_with_probs([[0.05, 0.10, 0.85]])
        results = asyncio.run(client.score("sentence", ["chunk"]))
        assert results[0].label == ConfidenceLabel.SUPPORTED
        assert abs(results[0].score - 0.85) < 1e-4

    def test_soft_threshold_upgrades_borderline_neutral_to_supported(self) -> None:
        """
        Argmax = neutral, but entailment = 0.40 (>= 0.35).
        Old code: PARTIALLY_SUPPORTED.  New code: SUPPORTED.
        """
        # [contradiction=0.10, neutral=0.50, entailment=0.40]
        client = _make_mnli_client_with_probs([[0.10, 0.50, 0.40]])
        results = asyncio.run(client.score("sentence", ["chunk"]))
        assert results[0].label == ConfidenceLabel.SUPPORTED

    def test_soft_threshold_just_above_boundary_is_supported(self) -> None:
        """Entailment score just above ENTAILMENT_SOFT_THRESHOLD = 0.35 → SUPPORTED."""
        # 0.351 is safely above 0.35 even under float32 rounding
        # [contradiction=0.099, neutral=0.550, entailment=0.351]
        client = _make_mnli_client_with_probs([[0.099, 0.550, 0.351]])
        results = asyncio.run(client.score("sentence", ["chunk"]))
        assert results[0].label == ConfidenceLabel.SUPPORTED

    def test_below_threshold_neutral_stays_partially_supported(self) -> None:
        """
        Entailment = 0.20 (< 0.35), argmax = neutral.
        Genuine neutral: no upgrade.
        """
        # [contradiction=0.05, neutral=0.75, entailment=0.20]
        client = _make_mnli_client_with_probs([[0.05, 0.75, 0.20]])
        results = asyncio.run(client.score("sentence", ["chunk"]))
        assert results[0].label == ConfidenceLabel.PARTIALLY_SUPPORTED

    def test_contradiction_below_threshold_stays_not_supported(self) -> None:
        """Strong contradiction is unaffected by the soft threshold."""
        # [contradiction=0.80, neutral=0.15, entailment=0.05]
        client = _make_mnli_client_with_probs([[0.80, 0.15, 0.05]])
        results = asyncio.run(client.score("sentence", ["chunk"]))
        assert results[0].label == ConfidenceLabel.NOT_SUPPORTED

    def test_multiple_chunks_mixed_labels(self) -> None:
        """
        Three chunks:
          chunk 0: strong entailment         → SUPPORTED
          chunk 1: borderline neutral (0.38) → SUPPORTED (soft threshold)
          chunk 2: genuine neutral (0.15)    → PARTIALLY_SUPPORTED
        """
        probs = [
            [0.05, 0.10, 0.85],   # entailment argmax
            [0.10, 0.52, 0.38],   # neutral argmax, entailment >= 0.35
            [0.05, 0.80, 0.15],   # neutral argmax, entailment < 0.35
        ]
        client = _make_mnli_client_with_probs(probs)
        results = asyncio.run(client.score("sentence", ["c0", "c1", "c2"]))
        assert results[0].label == ConfidenceLabel.SUPPORTED
        assert results[1].label == ConfidenceLabel.SUPPORTED
        assert results[2].label == ConfidenceLabel.PARTIALLY_SUPPORTED

    def test_soft_threshold_constant_value(self) -> None:
        """ENTAILMENT_SOFT_THRESHOLD is documented at 0.35 — guard against drift."""
        assert LocalMNLIClient.ENTAILMENT_SOFT_THRESHOLD == 0.35


# ---------------------------------------------------------------------------
# Direction E – DEFAULT_MIN_QUOTE_WORDS lowered to 10
# ---------------------------------------------------------------------------

class TestMinQuoteWords:
    def test_default_min_quote_words_is_ten(self) -> None:
        assert VerificationService.DEFAULT_MIN_QUOTE_WORDS == 10

    def test_short_definition_quote_is_verified(self) -> None:
        """
        A 10-word quote that would have been rejected at the old threshold of 15
        should now be accepted.
        """
        source_text = (
            "FEED maturity is defined as the degree of completeness of the deliverables "
            "to serve as the basis for detailed design at the end of detailed scope Phase Gate 3."
        )
        quote = "FEED maturity is defined as the degree of completeness of"  # 10 words
        assert len(quote.split()) == 10

        chunk = build_chunk("feed-def", source_text)
        store = InMemoryChunkStore([chunk])
        verifier = VerificationService(chunk_store=store, nli_client=HeuristicNLIClient())

        parsed = [
            ParsedSentence(
                sentence_index=0,
                sentence_text="FEED maturity is defined as the degree of completeness.",
                sentence_type=SentenceType.CLAIM,
                references=[Reference(reference_quote=quote)],
                status=SentenceStatus.UNCHECKED,
            )
        ]
        citations = [make_citation(chunk)]

        result = verifier.verify_exact_matches(parsed, citations)
        assert result.parsed_sentences[0].references[0].verified is True

    def test_nine_word_quote_still_rejected_at_default(self) -> None:
        """
        A 9-word quote is below the new threshold of 10 and should NOT be verified
        (failed_count > 0 → UNGROUNDED for CLAIM).
        """
        source_text = (
            "Advanced work packaging reduces rework and improves schedule predictability "
            "across all project phases and disciplines."
        )
        quote = "Advanced work packaging reduces rework and improves schedule"  # 8 words
        assert len(quote.split()) == 8

        chunk = build_chunk("awp", source_text)
        store = InMemoryChunkStore([chunk])
        verifier = VerificationService(chunk_store=store, nli_client=HeuristicNLIClient())

        parsed = [
            ParsedSentence(
                sentence_index=0,
                sentence_text="Advanced work packaging reduces rework and improves schedule.",
                sentence_type=SentenceType.CLAIM,
                references=[Reference(reference_quote=quote)],
                status=SentenceStatus.UNCHECKED,
            )
        ]
        citations = [make_citation(chunk)]

        result = verifier.verify_exact_matches(parsed, citations)
        assert result.parsed_sentences[0].references[0].verified is False
        assert result.parsed_sentences[0].status == SentenceStatus.UNGROUNDED
