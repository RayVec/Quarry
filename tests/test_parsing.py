from quarry.domain.models import SentenceStatus, SentenceType
from quarry.pipeline.parsing import parse_generated_response


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

