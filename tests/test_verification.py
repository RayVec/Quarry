import asyncio

from quarry.adapters.in_memory import HeuristicNLIClient, InMemoryChunkStore
from quarry.domain.models import ChunkObject, CitationIndexEntry, ConfidenceLabel, ParsedSentence, Reference, ScoredReference, SentenceStatus, SentenceType
from quarry.pipeline.verification import VerificationService


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


def test_verifier_discovers_quote_outside_initial_citation_index() -> None:
    chunk_a = build_chunk(
        "a",
        "Prefabricated modular approaches led to a 23 percent decrease in overall schedule duration across Phase III projects and improved coordination discipline.",
    )
    chunk_b = build_chunk(
        "b",
        "Projects that locked procurement packages later than the sixty percent design milestone experienced repeated site disruptions because equipment lead times no longer aligned with installation windows.",
    )
    store = InMemoryChunkStore([chunk_a, chunk_b])
    verifier = VerificationService(chunk_store=store, nli_client=HeuristicNLIClient())

    parsed = [
        ParsedSentence(
            sentence_index=0,
            sentence_text="Late procurement planning created disruption risk.",
            sentence_type=SentenceType.CLAIM,
            references=[
                Reference(
                    reference_quote="Projects that locked procurement packages later than the sixty percent design milestone experienced repeated site disruptions because equipment lead times no longer aligned with installation windows."
                )
            ],
            status=SentenceStatus.UNCHECKED,
        )
    ]
    citations = [
        CitationIndexEntry(
            citation_id=1,
            chunk_id="a",
            text=chunk_a.text,
            document_id=chunk_a.document_id,
            document_title=chunk_a.document_title,
            section_heading=chunk_a.section_heading,
            section_path=chunk_a.section_path,
            page_number=chunk_a.page_start,
            retrieval_score=0.8,
            source_facet="modular schedule",
        )
    ]

    verified = verifier.verify_exact_matches(parsed, citations)
    scored = asyncio.run(verifier.score_confidence(verified.parsed_sentences))

    assert len(verified.citation_index) == 2
    assert scored[0].references[0].verified is True
    assert scored[0].references[0].matched_chunk_id == "b"


def test_verifier_can_reverify_only_changed_sentence() -> None:
    chunk_a = build_chunk(
        "a",
        "Prefabricated modular approaches led to a 23 percent decrease in overall schedule duration across Phase III projects and improved coordination discipline.",
    )
    chunk_b = build_chunk(
        "b",
        "Projects that locked procurement packages later than the sixty percent design milestone experienced repeated site disruptions because equipment lead times no longer aligned with installation windows.",
    )
    store = InMemoryChunkStore([chunk_a, chunk_b])
    verifier = VerificationService(chunk_store=store, nli_client=HeuristicNLIClient())
    citations = [
        CitationIndexEntry(
            citation_id=1,
            chunk_id="a",
            text=chunk_a.text,
            document_id=chunk_a.document_id,
            document_title=chunk_a.document_title,
            section_heading=chunk_a.section_heading,
            section_path=chunk_a.section_path,
            page_number=chunk_a.page_start,
            retrieval_score=0.8,
            source_facet="modular schedule",
        ),
        CitationIndexEntry(
            citation_id=2,
            chunk_id="b",
            text=chunk_b.text,
            document_id=chunk_b.document_id,
            document_title=chunk_b.document_title,
            section_heading=chunk_b.section_heading,
            section_path=chunk_b.section_path,
            page_number=chunk_b.page_start,
            retrieval_score=0.7,
            source_facet="procurement risk",
        ),
    ]
    parsed = [
        ParsedSentence(
            sentence_index=0,
            sentence_text="Modular approaches improved coordination discipline.",
            sentence_type=SentenceType.CLAIM,
            references=[
                Reference(
                    reference_quote=chunk_a.text,
                    matched_chunk_id="a",
                    verified=True,
                    citation_id=1,
                )
            ],
            status=SentenceStatus.VERIFIED,
        ),
        ParsedSentence(
            sentence_index=1,
            sentence_text="Late procurement planning created disruption risk.",
            sentence_type=SentenceType.CLAIM,
            references=[Reference(reference_quote=chunk_b.text)],
            status=SentenceStatus.UNGROUNDED,
        ),
    ]

    verified = verifier.verify_exact_matches(parsed, citations, sentence_indices={1})

    assert verified.parsed_sentences[0].status == SentenceStatus.VERIFIED
    assert verified.parsed_sentences[0].references[0].matched_chunk_id == "a"
    assert verified.parsed_sentences[1].status == SentenceStatus.UNCHECKED
    assert verified.parsed_sentences[1].references[0].matched_chunk_id == "b"


class CountingNLIClient:
    def __init__(self) -> None:
        self.calls = 0

    async def score(self, sentence_text: str, chunk_texts: list[str]):
        self.calls += 1
        return [ScoredReference(score=0.99, label=ConfidenceLabel.SUPPORTED) for _ in chunk_texts]


def test_verifier_caches_confidence_scores_for_unchanged_pairs() -> None:
    chunk = build_chunk(
        "a",
        "Prefabricated modular approaches led to a 23 percent decrease in overall schedule duration across Phase III projects and improved coordination discipline.",
    )
    store = InMemoryChunkStore([chunk])
    nli = CountingNLIClient()
    verifier = VerificationService(chunk_store=store, nli_client=nli)
    parsed = [
        ParsedSentence(
            sentence_index=0,
            sentence_text="Modular approaches improved coordination discipline.",
            sentence_type=SentenceType.CLAIM,
            references=[
                Reference(
                    reference_quote=chunk.text,
                    matched_chunk_id="a",
                    verified=True,
                    citation_id=1,
                )
            ],
            status=SentenceStatus.UNCHECKED,
        )
    ]

    first = asyncio.run(verifier.score_confidence(parsed))
    second = asyncio.run(verifier.score_confidence(first))

    assert first[0].status == SentenceStatus.VERIFIED
    assert second[0].status == SentenceStatus.VERIFIED
    assert nli.calls == 1
