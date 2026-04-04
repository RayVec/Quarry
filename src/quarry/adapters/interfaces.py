from __future__ import annotations

from typing import Protocol, Sequence

from quarry.domain.models import (
    ChunkObject,
    GenerationRequest,
    ParsedDocument,
    RetrievalFilters,
    RetrievedPassage,
    ScoredReference,
)


class ChunkStore(Protocol):
    def all_chunks(self) -> list[ChunkObject]:
        ...

    def get_chunk(self, chunk_id: str) -> ChunkObject | None:
        ...

    def find_chunk_by_quote(
        self,
        quote: str,
        *,
        chunk_ids: Sequence[str] | None = None,
    ) -> ChunkObject | None:
        ...


class DecompositionClient(Protocol):
    async def classify_query(self, query: str) -> str:
        ...

    async def decompose_query(self, query: str, max_facets: int) -> list[str]:
        ...


class Retriever(Protocol):
    async def search(
        self,
        query: str,
        *,
        top_k: int,
        source_facet: str,
        filters: RetrievalFilters | None = None,
    ) -> tuple[list[RetrievedPassage], dict[str, object] | None]:
        ...


class Reranker(Protocol):
    async def rerank(self, query: str, candidates: Sequence[RetrievedPassage]) -> list[RetrievedPassage]:
        ...


class GenerationClient(Protocol):
    async def generate(self, request: GenerationRequest) -> str:
        ...


class NLIClient(Protocol):
    async def score(
        self,
        sentence_text: str,
        chunk_texts: Sequence[str],
    ) -> list[ScoredReference]:
        ...


class ParserAdapter(Protocol):
    parser_name: str

    def parse(self, source_path: str) -> ParsedDocument:
        ...


class MetadataEnricher(Protocol):
    async def enrich(self, chunk: ChunkObject) -> ChunkObject:
        ...


class EmbeddingClient(Protocol):
    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...
