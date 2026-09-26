"""Typed models for external key-value memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class ValueType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    LIST = "list"
    OBJECT = "object"
    NULL = "null"


@dataclass(frozen=True, slots=True)
class MemoryKey:
    namespace: str
    entity_id: str
    key: str


@dataclass(frozen=True, slots=True)
class KVMemory:
    id: str
    namespace: str
    entity_id: str
    key: str
    value: JSONValue
    value_type: ValueType
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None
    metadata: dict[str, JSONValue] | None


@dataclass(frozen=True, slots=True)
class MemoryStats:
    active_entries: int
    expired_entries: int
    namespaces: int
    entities: int
