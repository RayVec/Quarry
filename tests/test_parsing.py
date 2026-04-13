from quarry.domain.models import SentenceStatus, SentenceType
from quarry.pipeline.parsing import has_multiple_natural_sentences, parse_generated_response, render_parsed_sentences


def test_parse_generated_response_reclassifies_and_flags_structure() -> None:
    raw_response = """
    [STRUCTURE] Phase III projects saw 23 percent schedule reductions.

    [CLAIM] Multiple passages support this conclusion.
    [REF: "Prefabricated modular approaches led to a 23 percent decrease in overall schedule duration across Phase III projects, especially when design coordination was completed before fabrication mobilization began."]
    [REF: "Labor costs were reduced by 15 percent when off-site fabrication was used for structural components, but those savings depended on stable procurement sequencing and vendor availability."]
    """

    parsed = parse_generated_response(raw_response)

    assert len(parsed) == 2
    assert parsed[0].sentence_type == SentenceType.STRUCTURE
    assert parsed[0].structural_warning is True
    assert parsed[1].sentence_type == SentenceType.SYNTHESIS
    assert parsed[1].status == SentenceStatus.UNCHECKED


def test_parse_generated_response_assigns_paragraph_indices_from_para_markers() -> None:
    raw_response = """
    [CLAIM] The detailed scope phase produced engineering outputs and deliverables.
    [REF: "The detailed scope phase produced engineering outputs and deliverables for front-end design."]

    [PARA]

    [CLAIM] Those outputs became foundational reading material for subsequent planning.
    [REF: "Those engineering outputs became foundational reading material for subsequent planning decisions."]
    """

    parsed = parse_generated_response(raw_response)

    assert len(parsed) == 2
    assert parsed[0].paragraph_index == 0
    assert parsed[1].paragraph_index == 1

    rendered = render_parsed_sentences(parsed)
    assert "[PARA]" in rendered


def test_has_multiple_natural_sentences_detects_combined_claim_block() -> None:
    assert has_multiple_natural_sentences(
        "FEED maturity is defined in the report. It is also described in relation to the broader PDRI."
    ) is True
    assert has_multiple_natural_sentences(
        "Using this output, executive leadership (e.g., project sponsor, executive steering committees) can better assess where and how to commit limited resources."
    ) is False
