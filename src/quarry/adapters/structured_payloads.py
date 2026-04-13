from __future__ import annotations

from quarry.domain.models import ChunkObject


def extract_query_facets(payload: dict[str, object], *, query: str, max_facets: int) -> list[str]:
    facets = [str(item) for item in payload.get("facets", []) if str(item).strip()]
    return facets[:max_facets] or [query]


def apply_metadata_enrichment(chunk: ChunkObject, payload: dict[str, object]) -> ChunkObject:
    return chunk.model_copy(
        update={
            "metadata_summary": str(payload.get("summary", chunk.metadata_summary)),
            "metadata_entities": [str(item) for item in payload.get("entities", [])],
            "metadata_questions": [str(item) for item in payload.get("questions", [])],
        }
    )