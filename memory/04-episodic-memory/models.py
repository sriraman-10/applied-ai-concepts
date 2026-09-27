"""Typed models for complete support-case episodes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class EpisodeStatus(str, Enum):
    OPEN = "open"
    COMPLETED = "completed"


class EpisodeOutcome(str, Enum):
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass(frozen=True, slots=True)
class EpisodeAction:
    sequence: int
    action_type: str
    tool_name: str | None
    input_summary: str | None
    result_summary: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class EpisodeObservation:
    sequence: int
    source: str
    content: str
    important: bool
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Episode:
    id: str
    episode_key: str
    customer_id: str | None
    issue_type: str
    description: str
    episode_summary: str | None
    actions: tuple[EpisodeAction, ...]
    observations: tuple[EpisodeObservation, ...]
    resolution: str | None
    outcome: EpisodeOutcome | None
    status: EpisodeStatus
    metadata: dict[str, JSONValue]
    created_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True, slots=True)
class EpisodeSearchResult:
    episode: Episode
    distance: float
    similarity_score: float


@dataclass(frozen=True, slots=True)
class EpisodeStats:
    total: int
    open: int
    completed: int
    resolved: int
    escalated: int
    failed: int
    abandoned: int
