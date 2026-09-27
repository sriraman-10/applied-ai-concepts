"""Typed domain models for semantic vector memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class MemoryType(str, Enum):
    OBSERVATION = "observation"
    FACT = "fact"
    PREFERENCE = "preference"
    INCIDENT = "incident"
    LEARNING = "learning"


@dataclass(frozen=True, slots=True)
class VectorMemory:
    id: str
    content: str
    memory_type: MemoryType
    source: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, JSONValue]
    importance: int


@dataclass(frozen=True, slots=True)
class AddMemoryResult:
    memory: VectorMemory
    created: bool


@dataclass(frozen=True, slots=True)
class SearchResult:
    memory: VectorMemory
    distance: float
    similarity_score: float


@dataclass(frozen=True, slots=True)
class MemoryStats:
    count: int
    collection: str
