from __future__ import annotations

import re

from quarry.adapters.interfaces import DecompositionClient
from quarry.domain.models import DecompositionResult, QueryType
from quarry.logging_utils import logger_with_trace


ENTITY_PATTERN = re.compile(r"\b(?:[A-Z]{2,}(?:[-/][A-Z0-9]+)*|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|\d+(?:\.\d+)?%?)\b")
WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'/-]*")
SINGLE_HOP_PREFIXES = (
    "what is ",
    "what was ",
    "what are ",
    "define ",
    "definition of ",
    "what is the definition of ",
    "how many ",
    "how much ",
    "what percentage ",
    "what proportion ",
    "what level ",
    "what maturity ",
    "what was the average ",
    "what is the average ",
    "what was the median ",
    "what is the median ",
    "who is ",
    "who was ",
    "when was ",
    "where is ",
    "where was ",
    "which standard ",
    "which phase ",
)
MULTI_HOP_PHRASES = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    "between",
    "across",
    "both ",
    "tradeoffs",
    "trade-offs",
    "relationship between",
    "risk factors and recommended mitigation strategies",
    "causes and mitigation",
)
MULTI_HOP_OPENERS = (
    "what are ",
    "how does ",
    "how do ",
    "which factors ",
    "what factors ",
    "what are the key ",
)
METRIC_HINTS = ("average", "median", "mean", "rate", "percentage", "proportion", "cost overrun", "maturity")


logger = logger_with_trace(__name__)


class QueryDecomposer:
    def __init__(self, client: DecompositionClient, *, max_facets: int) -> None:
        self.client = client
        self.max_facets = max_facets

    async def decompose(self, query: str) -> DecompositionResult:
        normalized_query = " ".join(query.split()).strip()
        
        # Use heuristic-only classification
        raw_query_type, classification_source = self._heuristic_classify_query(normalized_query)
        
        # If heuristic cannot determine, default to multi_hop
        if raw_query_type is None:
            logger.info(
                "heuristic uncertain, defaulting to multi_hop",
                extra={
                    "query_preview": self._preview_query(normalized_query),
                    "classification_source": "heuristic_default_multi_hop",
                },
            )
            raw_query_type = "multi_hop"
            classification_source = "heuristic_default_multi_hop"
        
        # Validate query_type
        if raw_query_type in QueryType._value2member_map_:
            query_type = QueryType(raw_query_type)
        else:
            query_type = QueryType.SINGLE_HOP
            classification_source = f"{classification_source}_fallback_single_hop"
        
        self._log_classification(
            normalized_query,
            query_type=query_type.value,
            source=classification_source,
        )
        
        # Single hop: return original query
        if query_type == QueryType.SINGLE_HOP:
            return DecompositionResult(query_type=query_type, facets=[normalized_query])
        
        # Multi hop: decompose into facets using MLX
        facets = await self.client.decompose_query(normalized_query, self.max_facets)
        if not facets:
            facets = [normalized_query]
            query_type = QueryType.SINGLE_HOP
        
        facets = self._validate_entities(normalized_query, facets)[: self.max_facets]
        deduped: list[str] = []
        seen: set[str] = set()
        for facet in facets:
            key = facet.lower()
            if key not in seen:
                deduped.append(facet)
                seen.add(key)
        return DecompositionResult(query_type=query_type, facets=deduped or [normalized_query])

    def _validate_entities(self, original_query: str, facets: list[str]) -> list[str]:
        entities = [entity.strip() for entity in ENTITY_PATTERN.findall(original_query)]
        validated = list(facets)
        for entity in entities:
            if not any(entity.lower() in facet.lower() for facet in validated):
                validated.append(f"{original_query} focusing on {entity}")
        return validated

    def _heuristic_classify_query(self, query: str) -> tuple[str | None, str]:
        lowered = query.lower().strip()
        tokens = WORD_PATTERN.findall(lowered)

        # No longer use clarification - only classify as single or multi hop
        if self._looks_multi_hop(lowered):
            return "multi_hop", "heuristic_multi_hop"

        if self._looks_single_hop(lowered):
            return "single_hop", "heuristic_single_hop"

        return None, "heuristic"

    def _looks_multi_hop(self, lowered: str) -> bool:
        if any(phrase in lowered for phrase in MULTI_HOP_PHRASES):
            return True
        if lowered.count(",") > 0 and any(lowered.startswith(prefix) for prefix in MULTI_HOP_OPENERS):
            return True
        if " and " in lowered and any(lowered.startswith(prefix) for prefix in MULTI_HOP_OPENERS):
            return True
        return False

    def _looks_single_hop(self, lowered: str) -> bool:
        if any(lowered.startswith(prefix) for prefix in SINGLE_HOP_PREFIXES):
            return True
        if any(hint in lowered for hint in METRIC_HINTS) and " and " not in lowered and "," not in lowered:
            return True
        return False

    def _log_classification(
        self,
        query: str,
        *,
        query_type: str,
        source: str,
    ) -> None:
        logger.info(
            "query classification resolved",
            extra={
                "query_preview": self._preview_query(query),
                "query_type": query_type,
                "classification_source": source,
            },
        )

    def _preview_query(self, query: str, *, limit: int = 120) -> str:
        if len(query) <= limit:
            return query
        return f"{query[: limit - 1]}…"
