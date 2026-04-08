import asyncio

from quarry.adapters.interfaces import DecompositionClient
from quarry.domain.models import CitationIndexEntry, GenerationRequest, ReviewComment, SentenceCitationPair
from quarry.pipeline.decomposition import QueryDecomposer
from quarry.prompts import SHARED_SYSTEM_PROMPT, decomposition_classification_prompt, decomposition_prompt, generation_prompt, repair_generation_prompt, with_shared_system_prompt


def _citation(citation_id: int, text: str, *, note: str | None = None) -> CitationIndexEntry:
    return CitationIndexEntry(
        citation_id=citation_id,
        chunk_id=f"chunk-{citation_id}",
        text=text,
        document_id="doc-1",
        document_title="Sample Report",
        section_heading="Executive Summary",
        section_path="1 Executive Summary",
        page_number=1,
        retrieval_score=0.91,
        source_facet="schedule",
        reviewer_note=note,
    )


def test_generation_prompt_places_context_and_task_before_citation_rules() -> None:
    request = GenerationRequest(
        original_query="What are the main schedule risks?",
        facets=["schedule drivers", "procurement delays"],
        citation_index=[_citation(1, "Procurement packages that were locked late repeatedly disrupted installation windows.")],
    )

    prompt = generation_prompt(request)

    assert "## Query" in prompt
    assert "## Information Facets" in prompt
    assert "- schedule drivers" in prompt
    assert "- procurement delays" in prompt
    assert "## Source Passages" in prompt
    assert "## Your Task" in prompt
    assert "## Citation Format" in prompt
    assert prompt.index("## Source Passages") < prompt.index("## Your Task") < prompt.index("## Citation Format")
    assert "The most important finding or answer comes first" in prompt
    assert "For standard response generation, use 10 to 40 words" in prompt
    assert "Insert a [PARA] marker when the topic shifts" in prompt
    assert "[PARA] is formatting only" in prompt


def test_decomposition_prompt_uses_search_ready_facet_instructions() -> None:
    prompt = decomposition_prompt("How does Advanced Work Packaging affect project cost performance and safety outcomes?", 4)

    assert "You are decomposing a research query into focused sub-queries" in prompt
    assert 'Each sub-query (called a "facet") should be:' in prompt
    assert "A complete, self-contained question that could be typed into" in prompt
    assert "- Produce 2 to 4 facets" in prompt
    assert '"How does Advanced Work Packaging affect project cost performance?"' in prompt
    assert '{"facets": ["sub-query 1", "sub-query 2", ...]}' in prompt
    assert "The first character of your response must be '{' and there must be no prefix text." in prompt


def test_shared_system_prompt_covers_domain_format_grounding_and_verbatim_rules() -> None:
    wrapped = with_shared_system_prompt('Return JSON only: {"ok": true}')

    assert "You are QUARRY, a research assistant that helps domain experts" in SHARED_SYSTEM_PROMPT
    assert "technical construction industry" in SHARED_SYSTEM_PROMPT
    assert "return only valid JSON" in SHARED_SYSTEM_PROMPT
    assert "You never invent information." in SHARED_SYSTEM_PROMPT
    assert "When quoting from source passages, copy the text exactly as it" in SHARED_SYSTEM_PROMPT
    assert wrapped.startswith(SHARED_SYSTEM_PROMPT)


def test_generation_prompt_adds_supplement_instruction_without_repeating_structure_rules_first() -> None:
    request = GenerationRequest(
        original_query="What are the main schedule risks?",
        facets=["schedule drivers", "procurement delays"],
        citation_index=[_citation(1, "Procurement packages that were locked late repeatedly disrupted installation windows.")],
        mode="supplement",
        existing_response="The current answer focuses on procurement and misses labor planning.",
        selected_facets=["labor planning"],
    )

    prompt = generation_prompt(request)

    assert "## Existing Response Context" in prompt
    assert "## Additional Instruction" in prompt
    assert "- labor planning" in prompt
    assert "Write only the additional content addressing those facets." in prompt
    assert prompt.index("## Existing Response Context") < prompt.index("## Your Task")


def test_generation_prompt_includes_refinement_feedback_sections() -> None:
    request = GenerationRequest(
        original_query="What are the main schedule risks?",
        facets=["schedule drivers"],
        citation_index=[
            _citation(
                1,
                "Procurement packages that were locked late repeatedly disrupted installation windows.",
                note="The reviewer thinks this passage overstates the effect size.",
            )
        ],
        mode="refinement",
        mismatch_citation_ids=[1],
        disagreement_notes=["This claim may not apply to retrofit projects."],
        disagreement_contexts=["Sentence 2: A later section says retrofit schedules were more variable than new-build schedules."],
    )

    prompt = generation_prompt(request)

    assert "## Reviewer Feedback" in prompt
    assert "Citation [1] in section '1 Executive Summary' was flagged" in prompt
    assert "Disagreement note: This claim may not apply to retrofit projects." in prompt
    assert "Contradicting evidence: Sentence 2: A later section says retrofit schedules were more variable than new-build schedules." in prompt
    assert "Avoid reliance on flagged passages." in prompt
    assert "Reviewer approval is a positive signal for sentence-citation pairs." in prompt


def test_generation_prompt_includes_reviewer_approved_pairs() -> None:
    request = GenerationRequest(
        original_query="What are the main schedule risks?",
        facets=["schedule drivers"],
        citation_index=[_citation(2, "Schedule risk remained elevated when procurement lagged design completion milestones.")],
        mode="refinement",
        approved_pairs=[SentenceCitationPair(sentence_index=1, citation_id=2)],
    )

    prompt = generation_prompt(request)

    assert "Reviewer approvals (positive signal, not proof):" in prompt
    assert "- Reviewer approved sentence-citation pair (sentence_index=1, citation_id=2)." in prompt


def test_generation_prompt_includes_selection_comments() -> None:
    request = GenerationRequest(
        original_query="What are the main schedule risks?",
        facets=["schedule drivers"],
        citation_index=[_citation(1, "Procurement packages that were locked late repeatedly disrupted installation windows.")],
        mode="refinement",
        selection_comments=[
            ReviewComment(
                comment_id="c1",
                text_selection="threshold should be under five million dollars",
                char_start=12,
                char_end=54,
                comment_text="This number is incorrect; use 10M not 5M.",
                sentence_index=3,
                sentence_type="claim",
                sentence_text="Old sentence",
            )
        ],
    )

    prompt = generation_prompt(request)

    assert 'Selected: "threshold should be under five million dollars"' in prompt
    assert "This number is incorrect; use 10M not 5M." in prompt
    assert "Treat each selection comment as a required edit." in prompt


def test_generation_prompt_includes_sentence_repair_mode() -> None:
    request = GenerationRequest(
        original_query="Repair this sentence",
        facets=[],
        citation_index=[_citation(1, "Factory fabrication reduced schedule variance by narrowing weather exposure during installation.")],
        mode="regeneration",
        failed_sentence_text="Factory fabrication always eliminates delay risk.",
    )

    prompt = generation_prompt(request)

    assert "## Sentence Repair" in prompt
    assert '"Factory fabrication always eliminates delay risk."' in prompt
    assert "Rewrite only this sentence." in prompt
    assert "shorter exact quotes of 8 to 15 words are allowed" in prompt
    assert "Write a clear, natural sentence" in prompt


def test_generation_prompt_includes_retry_guidance_for_failed_regeneration() -> None:
    request = GenerationRequest(
        original_query="Repair this sentence",
        facets=[],
        citation_index=[_citation(1, "Factory fabrication reduced schedule variance by narrowing weather exposure during installation.")],
        mode="regeneration",
        failed_sentence_text="Factory fabrication always eliminates delay risk.",
        failed_regeneration_response="Factory fabrication greatly reduces weather-driven delay exposure.",
    )

    prompt = generation_prompt(request)

    assert "## Retry Guidance" in prompt
    assert 'Your previous rewrite was: "Factory fabrication greatly reduces weather-driven delay exposure."' in prompt
    assert "This still could not be verified." in prompt
    assert "respond with [NO_REF]" in prompt


def test_repair_generation_prompt_wraps_prior_attempt_into_new_structure() -> None:
    request = GenerationRequest(
        original_query="What are the main schedule risks?",
        facets=["schedule drivers"],
        citation_index=[_citation(1, "Procurement packages that were locked late repeatedly disrupted installation windows.")],
    )

    prompt = repair_generation_prompt(request, "bad output")

    assert "## Previous Attempt" in prompt
    assert "bad output" in prompt
    assert "## Repair Instruction" in prompt


class _NoClassificationClient(DecompositionClient):
    def __init__(self, *, facets: list[str] | None = None) -> None:
        self.decompose_called = False
        self._facets = facets or []

    async def decompose_query(self, query: str, max_facets: int) -> list[str]:
        self.decompose_called = True
        return list(self._facets[:max_facets])


def test_query_decomposer_uses_heuristic_first_for_obvious_single_hop_queries() -> None:
    client = _NoClassificationClient()
    decomposer = QueryDecomposer(client, max_facets=4)

    result = asyncio.run(decomposer.decompose("What is PDRI maturity?"))

    assert result.query_type.value == "single_hop"
    assert result.facets == ["What is PDRI maturity?"]
    assert client.decompose_called is False


def test_query_decomposer_uses_heuristic_first_for_obvious_multi_hop_queries() -> None:
    client = _NoClassificationClient(
        facets=[
            "What are the key risk factors for schedule delays in CII Phase III projects?",
            "What mitigation strategies are recommended for schedule delays in CII Phase III projects?",
        ]
    )
    decomposer = QueryDecomposer(client, max_facets=4)

    result = asyncio.run(
        decomposer.decompose("What are the key risk factors and recommended mitigation strategies for schedule delays in CII Phase III projects?")
    )

    assert result.query_type.value == "multi_hop"
    assert len(result.facets) == 2
    assert client.decompose_called is True


def test_query_decomposer_defaults_to_multi_hop_when_heuristic_inconclusive() -> None:
    """When heuristic cannot classify, decomposer defaults to multi_hop without calling classify model."""
    client = _NoClassificationClient(
        facets=[
            "Why were retrofit projects delayed?",
            "What evidence links retrofit conditions to those delays?",
        ],
    )
    decomposer = QueryDecomposer(client, max_facets=4)

    result = asyncio.run(decomposer.decompose("Why were retrofit projects delayed?"))

    assert result.query_type.value == "multi_hop"
    assert client.decompose_called is True
