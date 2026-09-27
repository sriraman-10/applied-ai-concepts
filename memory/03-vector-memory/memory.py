"""Persistent semantic-memory manager backed by ChromaDB."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import chromadb
from chromadb.config import Settings as ChromaSettings

from embeddings import EmbeddingProvider
from models import (
    AddMemoryResult,
    JSONScalar,
    JSONValue,
    MemoryStats,
    MemoryType,
    SearchResult,
    VectorMemory,
)

logger = logging.getLogger(__name__)
_RESERVED_METADATA = {
    "memory_type",
    "source",
    "importance",
    "created_at",
    "updated_at",
    "content_hash",
    "metadata_json",
}
_SOURCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/ -]{0,99}$")


class MemoryValidationError(ValueError):
    """Raised when application input violates the memory contract."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalise_content(content: str) -> str:
    return " ".join(content.casefold().split())


def _content_hash(content: str) -> str:
    return hashlib.sha256(_normalise_content(content).encode("utf-8")).hexdigest()


def _coerce_type(value: MemoryType | str) -> MemoryType:
    try:
        return value if isinstance(value, MemoryType) else MemoryType(value)
    except ValueError as error:
        allowed = ", ".join(item.value for item in MemoryType)
        raise MemoryValidationError(f"memory_type must be one of: {allowed}") from error


def _validate_content(content: str) -> str:
    cleaned = content.strip()
    if not cleaned:
        raise MemoryValidationError("content cannot be empty")
    if len(cleaned) > 20_000:
        raise MemoryValidationError("content cannot exceed 20,000 characters")
    return cleaned


def _validate_source(source: str) -> str:
    cleaned = source.strip()
    if not _SOURCE_RE.fullmatch(cleaned):
        raise MemoryValidationError("source must be 1-100 safe printable characters")
    return cleaned


def _validate_importance(importance: int) -> int:
    if (
        isinstance(importance, bool)
        or not isinstance(importance, int)
        or not 1 <= importance <= 5
    ):
        raise MemoryValidationError("importance must be an integer from 1 to 5")
    return importance


def _validate_metadata(
    metadata: Mapping[str, JSONValue] | None,
) -> dict[str, JSONValue]:
    result = dict(metadata or {})
    if len(result) > 50:
        raise MemoryValidationError("metadata cannot contain more than 50 keys")
    for key in result:
        if not isinstance(key, str) or not key or len(key) > 100:
            raise MemoryValidationError(
                "metadata keys must be non-empty strings up to 100 characters"
            )
        if key in _RESERVED_METADATA:
            raise MemoryValidationError(f"metadata key {key!r} is reserved")
    try:
        encoded = json.dumps(
            result, ensure_ascii=False, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise MemoryValidationError(
            f"metadata must be valid finite JSON: {error}"
        ) from error
    if len(encoded) > 20_000:
        raise MemoryValidationError("metadata cannot exceed 20,000 encoded characters")
    return result


def _finite_vector(vector: Any) -> list[float]:
    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError) as error:
        raise MemoryValidationError("embedding must contain only numbers") from error
    if not values or not all(math.isfinite(value) for value in values):
        raise MemoryValidationError("embedding must be non-empty and finite")
    return values


class VectorMemoryManager:
    """Application-facing vector memory; Chroma details stay behind this class."""

    def __init__(
        self,
        path: Path | str,
        collection_name: str,
        embedding_provider: EmbeddingProvider,
        *,
        max_top_k: int = 25,
    ) -> None:
        if max_top_k <= 0:
            raise ValueError("max_top_k must be positive")
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
        self.max_top_k = max_top_k
        self._client = chromadb.PersistentClient(
            path=str(self.path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._open_collection()

    def _open_collection(self):
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _storage_metadata(memory: VectorMemory) -> dict[str, str | int | float | bool]:
        return {
            "memory_type": memory.memory_type.value,
            "source": memory.source,
            "importance": memory.importance,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "content_hash": _content_hash(memory.content),
            "metadata_json": json.dumps(
                memory.metadata, ensure_ascii=False, sort_keys=True
            ),
            **{
                f"user_{key}": value
                for key, value in memory.metadata.items()
                if isinstance(value, (str, int, float, bool)) and value is not None
            },
        }

    @staticmethod
    def _to_memory(
        memory_id: str, content: str, metadata: Mapping[str, Any]
    ) -> VectorMemory:
        return VectorMemory(
            id=memory_id,
            content=content,
            memory_type=MemoryType(str(metadata["memory_type"])),
            source=str(metadata["source"]),
            created_at=datetime.fromisoformat(str(metadata["created_at"])),
            updated_at=datetime.fromisoformat(str(metadata["updated_at"])),
            metadata=json.loads(str(metadata.get("metadata_json", "{}"))),
            importance=int(metadata["importance"]),
        )

    def _get_by_hash(self, digest: str) -> VectorMemory | None:
        result = self._collection.get(
            where={"content_hash": digest},
            include=["documents", "metadatas"],
            limit=1,
        )
        ids = result.get("ids") or []
        if not ids:
            return None
        return self._to_memory(
            ids[0],
            (result.get("documents") or [""])[0],
            (result.get("metadatas") or [{}])[0],
        )

    async def add_memory(
        self,
        content: str,
        *,
        memory_type: MemoryType | str = MemoryType.OBSERVATION,
        source: str = "cli",
        metadata: Mapping[str, JSONValue] | None = None,
        importance: int = 3,
    ) -> AddMemoryResult:
        clean_content = _validate_content(content)
        clean_type = _coerce_type(memory_type)
        clean_source = _validate_source(source)
        clean_metadata = _validate_metadata(metadata)
        clean_importance = _validate_importance(importance)
        digest = _content_hash(clean_content)
        existing = self._get_by_hash(digest)
        if existing is not None:
            logger.info("memory_add outcome=duplicate memory_id=%s", existing.id)
            return AddMemoryResult(existing, created=False)
        embedding = _finite_vector(await self.embedding_provider.embed(clean_content))
        now = _utc_now()
        memory = VectorMemory(
            id=str(uuid.uuid4()),
            content=clean_content,
            memory_type=clean_type,
            source=clean_source,
            created_at=now,
            updated_at=now,
            metadata=clean_metadata,
            importance=clean_importance,
        )
        self._collection.add(
            ids=[memory.id],
            documents=[memory.content],
            embeddings=[embedding],
            metadatas=[self._storage_metadata(memory)],
        )
        logger.info(
            "memory_add outcome=created memory_id=%s type=%s",
            memory.id,
            memory.memory_type.value,
        )
        return AddMemoryResult(memory, created=True)

    def get_memory(self, memory_id: str) -> VectorMemory | None:
        result = self._collection.get(
            ids=[memory_id], include=["documents", "metadatas"]
        )
        ids = result.get("ids") or []
        if not ids:
            return None
        return self._to_memory(
            ids[0],
            (result.get("documents") or [""])[0],
            (result.get("metadatas") or [{}])[0],
        )

    def list_memories(
        self, *, memory_type: MemoryType | str | None = None, limit: int = 100
    ) -> list[VectorMemory]:
        if limit <= 0 or limit > 500:
            raise MemoryValidationError("limit must be from 1 to 500")
        where = (
            {"memory_type": _coerce_type(memory_type).value}
            if memory_type is not None
            else None
        )
        kwargs: dict[str, Any] = {"include": ["documents", "metadatas"], "limit": limit}
        if where is not None:
            kwargs["where"] = where
        result = self._collection.get(**kwargs)
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        memories = [
            self._to_memory(i, d, m) for i, d, m in zip(ids, documents, metadatas)
        ]
        return sorted(memories, key=lambda item: item.created_at, reverse=True)

    @staticmethod
    def _where_filter(
        filters: Mapping[str, JSONScalar] | None,
    ) -> dict[str, Any] | None:
        if not filters:
            return None
        clauses: list[dict[str, Any]] = []
        for key, value in filters.items():
            if key == "memory_type":
                value = _coerce_type(str(value)).value
                field = key
            elif key == "source":
                field = key
            elif key == "importance":
                value = _validate_importance(value)  # type: ignore[arg-type]
                field = key
            else:
                if not isinstance(key, str) or not key or len(key) > 100:
                    raise MemoryValidationError(
                        "filter keys must be non-empty strings up to 100 characters"
                    )
                if value is None or isinstance(value, (list, dict)):
                    raise MemoryValidationError(
                        "metadata filters require a scalar non-null value"
                    )
                field = f"user_{key}"
            clauses.append({field: value})
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: Mapping[str, JSONScalar] | None = None,
        min_similarity: float | None = None,
    ) -> list[SearchResult]:
        clean_query = _validate_content(query)
        if top_k <= 0 or top_k > self.max_top_k:
            raise MemoryValidationError(f"top_k must be from 1 to {self.max_top_k}")
        if min_similarity is not None and not -1.0 <= min_similarity <= 1.0:
            raise MemoryValidationError("min_similarity must be between -1 and 1")
        count = self.count()
        if count == 0:
            return []
        started = time.perf_counter()
        embedding = _finite_vector(await self.embedding_provider.embed(clean_query))
        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": min(top_k, count),
            "include": ["documents", "metadatas", "distances"],
        }
        where = self._where_filter(filters)
        if where is not None:
            kwargs["where"] = where
        result = self._collection.query(**kwargs)
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        matches: list[SearchResult] = []
        for memory_id, content, metadata, distance_value in zip(
            ids, documents, metadatas, distances
        ):
            distance = float(distance_value)
            score = max(-1.0, min(1.0, 1.0 - distance))
            if min_similarity is None or score >= min_similarity:
                matches.append(
                    SearchResult(
                        self._to_memory(memory_id, content, metadata), distance, score
                    )
                )
        logger.info(
            "memory_search result_count=%d top_k=%d latency_ms=%.2f",
            len(matches),
            top_k,
            (time.perf_counter() - started) * 1000,
        )
        return matches

    async def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        memory_type: MemoryType | str | None = None,
        source: str | None = None,
        metadata: Mapping[str, JSONValue] | None = None,
        importance: int | None = None,
    ) -> VectorMemory | None:
        current = self.get_memory(memory_id)
        if current is None:
            return None
        new_content = (
            _validate_content(content) if content is not None else current.content
        )
        duplicate = self._get_by_hash(_content_hash(new_content))
        if duplicate is not None and duplicate.id != memory_id:
            raise MemoryValidationError(
                "another memory already has the same normalized content"
            )
        updated = VectorMemory(
            id=current.id,
            content=new_content,
            memory_type=(
                _coerce_type(memory_type)
                if memory_type is not None
                else current.memory_type
            ),
            source=_validate_source(source) if source is not None else current.source,
            created_at=current.created_at,
            updated_at=_utc_now(),
            metadata=(
                _validate_metadata(metadata)
                if metadata is not None
                else current.metadata
            ),
            importance=(
                _validate_importance(importance)
                if importance is not None
                else current.importance
            ),
        )
        kwargs: dict[str, Any] = {
            "ids": [memory_id],
            "metadatas": [self._storage_metadata(updated)],
        }
        if new_content != current.content:
            kwargs["documents"] = [updated.content]
            kwargs["embeddings"] = [
                _finite_vector(await self.embedding_provider.embed(new_content))
            ]
        self._collection.update(**kwargs)
        logger.info("memory_update outcome=updated memory_id=%s", memory_id)
        return updated

    def delete_memory(self, memory_id: str) -> bool:
        if self.get_memory(memory_id) is None:
            logger.info("memory_delete outcome=not_found memory_id=%s", memory_id)
            return False
        self._collection.delete(ids=[memory_id])
        logger.info("memory_delete outcome=deleted memory_id=%s", memory_id)
        return True

    def count(self) -> int:
        return self._collection.count()

    def stats(self) -> MemoryStats:
        return MemoryStats(count=self.count(), collection=self.collection_name)

    def clear(self) -> int:
        count = self.count()
        self._client.delete_collection(self.collection_name)
        self._collection = self._open_collection()
        logger.info("memory_clear deleted_count=%d", count)
        return count
