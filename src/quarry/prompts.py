from __future__ import annotations

import json

from quarry.domain.models import ChunkObject, GenerationRequest


SHARED_SYSTEM_PROMPT = (
    "You are QUARRY, a research assistant that helps domain experts\n"
    "locate and verify information from technical construction industry\n"
    "reports.\n\n"
    "You always respond in the exact format requested. When asked for\n"
    "JSON, return only valid JSON with no preamble, no markdown fences,\n"
    "and no commentary. When asked for tagged sentences, use only the\n"
    "tags specified ([CLAIM], [SYNTHESIS], [STRUCTURE]) with no\n"
    "additional markup.\n\n"
    "You never invent information. Every factual statement you make\n"
    "must be traceable to text provided in the prompt. If you cannot\n"
    "find supporting evidence, say so rather than fabricating content.\n\n"
    "When quoting from source passages, copy the text exactly as it\n"
    "appears. Do not paraphrase, truncate, rearrange, or correct\n"
    "spelling or grammar in quoted material."
)


def decomposition_classification_prompt(query: str) -> str:
    return (
        "Classify this research query into one of three types.\n\n"
        "single_hop: The query asks for one specific fact, definition,\n"
        "statistic, or piece of information that would likely be found\n"
        "in a single section of a document.\n\n"
        "multi_hop: The query asks about multiple topics, requests a\n"
        "comparison, spans different aspects of a subject, or would\n"
        "require gathering information from several document sections\n"
        "to answer fully.\n\n"
        "clarification_required: The query is too vague, ambiguous, or\n"
        "incomplete to classify. Examples: a single word with no context,\n"
        "a pronoun with no referent, or a question so broad that any\n"
        "document section could be relevant.\n\n"
        "Examples:\n"
        '- "What is the definition of Advanced Work Packaging?" → single_hop\n'
        '- "What are the key risk factors and recommended mitigation\n'
        '  strategies for schedule delays?" → multi_hop\n'
        '- "costs" → clarification_required\n'
        '- "What did they recommend?" → clarification_required\n'
        '- "How does modular construction affect both project timelines\n'
        '  and labor costs?" → multi_hop\n'
        '- "What was the average cost overrun reported in Phase III?" → single_hop\n\n'
        f"Query: {query}"
        "\n\nRespond with JSON only:\n"
        '{"query_type": "single_hop" | "multi_hop" | "clarification_required"}'
    )


def decomposition_prompt(query: str, max_facets: int) -> str:
    return (
        "You are decomposing a research query into focused sub-queries\n"
        "that will be used to search a corpus of technical construction\n"
        "reports.\n\n"
        'Each sub-query (called a "facet") should be:\n'
        "- A complete, self-contained question that could be typed into\n"
        "  a search engine and return useful results on its own\n"
        "- Focused on one specific aspect of the original query\n"
        "- Specific enough to target a particular section or topic in\n"
        "  a document, not so broad that any section could match\n\n"
        "Rules:\n"
        f"- Produce 2 to {max_facets} facets\n"
        "- Every named entity in the original query (project names,\n"
        "  organizations, standards, specific metrics) must appear in\n"
        "  at least one facet\n"
        "- Do not combine two distinct topics into one facet\n"
        "- Do not produce facets that overlap heavily with each other\n\n"
        "Example:\n"
        'Query: "What are the key risk factors and recommended mitigation\n'
        'strategies for schedule delays in CII Phase III projects?"\n'
        "Facets:\n"
        '- "What are the key risk factors for schedule delays in CII Phase III projects?"\n'
        '- "What mitigation strategies are recommended for schedule delays in CII Phase III projects?"\n\n'
        "Example:\n"
        'Query: "How does Advanced Work Packaging affect project cost\n'
        'performance and safety outcomes?"\n'
        "Facets:\n"
        '- "How does Advanced Work Packaging affect project cost performance?"\n'
        '- "How does Advanced Work Packaging affect safety outcomes?"\n\n'
        f"Query: {query}\n\n"
        "Respond with JSON only:\n"
        '{"facets": ["sub-query 1", "sub-query 2", ...]}'
    )


def metadata_enrichment_prompt(chunk: ChunkObject) -> str:
    return (
        "You are enriching a retrieval chunk.\n"
        "Return JSON with keys summary, entities, questions.\n"
        "summary must be one sentence.\n"
        "entities must be a short list of important terms.\n"
        "questions must be 2 to 3 user-facing questions.\n"
        "Output must be valid JSON only.\n"
        "The first character of your response must be '{' and there must be no prefix text.\n"
        f"Section heading: {chunk.section_heading}\n"
        f"Text: {chunk.text}"
    )


def generation_prompt(request: GenerationRequest) -> str:
    sections: list[str] = [
        (
            "You are a research assistant helping a domain expert locate and verify\n"
            "information from technical reports. Your job is to answer the query\n"
            "using only the evidence provided in the passages below."
        ),
        f"## Query\n{request.original_query}",
        "## Information Facets\n"
        "The query has been decomposed into these aspects:\n"
        f"{_format_facets(request.selected_facets or request.facets)}",
        "## Source Passages\n"
        f"{_format_passages(request)}",
    ]

    if request.existing_response:
        sections.append(f"## Existing Response Context\n{request.existing_response}")

    reviewer_feedback = _format_reviewer_feedback(request)
    if reviewer_feedback:
        sections.append("## Reviewer Feedback\nThe expert reviewed the previous response and flagged the following:\n\n" + reviewer_feedback)

    if request.repair_prior_response:
        sections.append(
            "## Previous Attempt\n"
            "The previous output was malformed or unusable. Repair it while keeping the content grounded in the passages below.\n\n"
            f"{request.repair_prior_response.strip()}"
        )

    sections.append(
        "## Your Task\n\n"
        "Read the passages carefully. Write a clear, well-organized response\n"
        "that addresses the query by synthesizing evidence from the passages.\n\n"
        "Organize your response so that:\n"
        "- The most important finding or answer comes first\n"
        "- Related points are grouped together rather than scattered\n"
        "- Transitions between topics are natural\n"
        "- If the query has multiple facets, address each one, but weave them\n"
        "  into a coherent narrative rather than treating them as separate lists\n\n"
        "When writing, think about what a domain expert would want to read:\n"
        "direct answers supported by evidence, with enough context to understand\n"
        "the significance, but no unnecessary filler."
    )

    sections.append(
        "## Citation Format\n\n"
        "Group related sentences into paragraphs. Insert a [PARA] marker when the topic shifts\n"
        "or when transitioning between distinct aspects of the answer.\n"
        "[PARA] is formatting only: it is not a sentence tag and must not include references.\n\n"
        "After writing each sentence, tag it and cite your evidence using\n"
        "exactly this format:\n\n"
        "CLAIM sentences state a single fact drawn from one passage.\n"
        "Format: [CLAIM] Your sentence here. [REF: \"exact quote from passage\"]\n"
        "The quote must be copied verbatim from a passage.\n"
        "For standard response generation, use 10 to 40 words with no\n"
        "paraphrasing or truncation.\n\n"
        "SYNTHESIS sentences connect findings across multiple passages.\n"
        "Format: [SYNTHESIS] Your sentence here. [REF: \"quote from passage A\"] [REF: \"quote from passage B\"]\n"
        "Each quote must be verbatim from a different passage.\n\n"
        "STRUCTURE sentences are transitions, framing, or introductions\n"
        "that make no factual claim.\n"
        "Format: [STRUCTURE] Your sentence here.\n"
        "No references needed.\n\n"
        "If no passage supports a claim, use [NO_REF] instead of inventing\n"
        "a quote. Never fabricate a reference."
    )

    mode_instruction = _format_mode_instruction(request)
    if mode_instruction:
        sections.append(mode_instruction)

    return "\n\n".join(section.strip() for section in sections if section.strip())


def repair_generation_prompt(request: GenerationRequest, raw_response: str) -> str:
    repaired_request = request.model_copy(update={"repair_prior_response": raw_response.strip()})
    return (
        generation_prompt(repaired_request)
        + "\n\n## Repair Instruction\n"
        "Return a corrected response that strictly follows the task and citation rules above.\n"
        "Do not add commentary, markdown fences, or explanations."
    )


def parse_json_response(raw_text: str) -> dict[str, object]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_candidates = [index for index in (text.find("{"), text.find("[")) if index >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        end = max(text.rfind("}"), text.rfind("]"))
        if end <= start:
            raise
        return json.loads(text[start : end + 1])


def _format_citation_line(citation, mismatch_ids: list[int]) -> str:
    prefix: list[str] = []
    if citation.citation_id in mismatch_ids:
        prefix.append(f"[MISMATCH: {citation.reviewer_note or 'reviewer flagged this passage'}]")
    if citation.replacement_pending:
        prefix.append("[REPLACED_PENDING_REGENERATION]")
    if citation.reviewer_note and citation.citation_id not in mismatch_ids:
        prefix.append(f"[NOTE: {citation.reviewer_note}]")
    prefix_text = " ".join(prefix)
    return f"{prefix_text} [{citation.citation_id}] ({citation.section_path}) {citation.text}".strip()


def with_shared_system_prompt(prompt: str) -> str:
    return f"{SHARED_SYSTEM_PROMPT}\n\n{prompt.strip()}"


def _format_facets(facets: list[str]) -> str:
    cleaned = [facet.strip() for facet in facets if facet.strip()]
    if not cleaned:
        return "- No explicit facets were provided for this request."
    return "\n".join(f"- {facet}" for facet in cleaned)


def _format_passages(request: GenerationRequest) -> str:
    passages = [_format_citation_line(citation, request.mismatch_citation_ids) for citation in request.citation_index]
    return "\n".join(passages) if passages else "(No passages were provided.)"


def _format_reviewer_feedback(request: GenerationRequest) -> str:
    feedback_lines: list[str] = []
    by_id = {citation.citation_id: citation for citation in request.citation_index}

    for citation_id in request.mismatch_citation_ids:
        citation = by_id.get(citation_id)
        if citation is None:
            feedback_lines.append(f"- Citation [{citation_id}] was flagged as a mismatch.")
            continue
        note = citation.reviewer_note or "Reviewer flagged this passage as a mismatch."
        feedback_lines.append(
            f"- Citation [{citation_id}] in section '{citation.section_path}' was flagged: {note}"
        )

    for note in request.disagreement_notes:
        if note.strip():
            feedback_lines.append(f"- Disagreement note: {note.strip()}")

    for context in request.disagreement_contexts:
        if context.strip():
            feedback_lines.append(f"- Contradicting evidence: {context.strip()}")

    for index, comment in enumerate(request.selection_comments, start=1):
        selection = comment.text_selection.strip()
        note = comment.comment_text.strip()
        if selection and note and not comment.resolved:
            feedback_lines.append(f'{index}. Selected: "{selection}"')
            feedback_lines.append(f"   Comment: {note}")

    return "\n".join(feedback_lines)


def _format_mode_instruction(request: GenerationRequest) -> str:
    if request.mode == "supplement":
        selected = request.selected_facets or request.facets
        return (
            "## Additional Instruction\n"
            "The expert has already received the response shown above.\n"
            "They feel that these facets need more coverage:\n"
            f"{_format_facets(selected)}\n\n"
            "Write only the additional content addressing those facets. Do not\n"
            "repeat anything from the existing response. Continue the narrative\n"
            "naturally as if appending a new section."
        )

    if request.mode == "refinement":
        selection_comment_instruction = ""
        if request.selection_comments:
            selection_comment_instruction = (
                "\nThe reviewer has highlighted sections of the response and left comments.\n"
                "Treat each selection comment as a required edit.\n"
                "Use the selected text span directly to locate what to revise.\n"
                "You may adjust surrounding wording only when needed for consistency."
            )
        return (
            "## Additional Instruction\n"
            "Regenerate the full response from the current answer context and passages.\n"
            "Avoid reliance on flagged passages.\n"
            "Where the reviewer disagreed with a claim, present both the original\n"
            "evidence and any contradicting evidence. If evidence is insufficient\n"
            "after removing flagged passages, say so rather than citing unsupported\n"
            "sources."
            f"{selection_comment_instruction}"
        )

    if request.mode == "regeneration" and request.failed_sentence_text:
        comment_instruction = ""
        if request.failed_sentence_comment:
            comment_instruction = (
                "\n\n## Reviewer Comment\n"
                f"{request.failed_sentence_comment.strip()}\n"
                "Address this comment while rewriting the sentence."
            )
        retry_guidance = ""
        if request.failed_regeneration_response:
            retry_guidance = (
                "\n\n## Retry Guidance\n"
                f'Your previous rewrite was: "{request.failed_regeneration_response}"\n'
                "This still could not be verified. Either find a different passage to cite,\n"
                "or respond with [NO_REF] if no passage supports this claim."
            )
        return (
            "## Sentence Repair\n"
            "The following sentence failed citation verification:\n"
            f"\"{request.failed_sentence_text}\"\n\n"
            "Rewrite only this sentence. Use evidence from the passages above.\n"
            "Write a clear, natural sentence rather than copying raw bullet points,\n"
            "headings, or chunk openings.\n"
            "For sentence repair, shorter exact quotes of 8 to 15 words are allowed\n"
            "when they provide a clean anchor for the sentence.\n"
            "Include a valid [REF: \"exact quote\"] or mark it [NO_REF] if no\n"
            "passage supports the claim."
            f"{comment_instruction}"
            f"{retry_guidance}"
        )

    return ""
