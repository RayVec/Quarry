from quarry.ingest.parsers import _build_mlx_document_from_blocks, parse_text_document


def test_parse_text_document_preserves_filename_stem_in_document_title() -> None:
    parsed_document = parse_text_document(
        "/tmp/314_2.pdf",
        "Executive Summary\nEvidence about schedule outcomes.",
        parser_name="basic_text",
    )

    assert parsed_document.document_title == "314_2"


def test_build_mlx_document_from_blocks_preserves_filename_stem_in_document_title() -> None:
    parsed_document = _build_mlx_document_from_blocks(
        "/tmp/314_2.pdf",
        [
            (
                1,
                [
                    {"text": "Executive Summary", "block_type": "heading", "section_depth": 0},
                    {"text": "Evidence about schedule outcomes.", "block_type": "paragraph"},
                ],
            )
        ],
        parser_name="qwen3_vl_mlx",
        parser_provenance=["mlx-community/Qwen3-VL-4B-Instruct-4bit"],
    )

    assert parsed_document.document_title == "314_2"
