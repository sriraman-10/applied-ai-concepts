"""Typed procedural-memory definitions and separate runtime state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class ProcedureStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class ExecutionStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    FAILED = "failed"


class StepStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ProcedureStep:
    sequence: int
    action: str
    tool: str | None = None
    required: bool = True


@dataclass(frozen=True, slots=True)
class SupportProcedure:
    id: str
    procedure_key: str
    version: int
    name: str
    description: str
    category: str
    procedure_summary: str
    status: ProcedureStatus
    steps: tuple[ProcedureStep, ...]
    preconditions: tuple[str, ...]
    completion_conditions: tuple[str, ...]
    escalation_rules: tuple[str, ...]
    metadata: dict[str, JSONValue]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ProcedureSearchResult:
    procedure: SupportProcedure
    distance: float
    similarity_score: float


@dataclass(frozen=True, slots=True)
class StepExecution:
    step_sequence: int
    action: str
    tool_name: str | None
    status: StepStatus
    result_summary: str | None
    started_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True, slots=True)
class ProcedureExecution:
    id: str
    procedure_id: str
    procedure_key: str
    procedure_version: int
    status: ExecutionStatus
    current_step: int
    started_at: datetime
    completed_at: datetime | None
    outcome: str | None
    metadata: dict[str, JSONValue]
    steps: tuple[StepExecution, ...]


@dataclass(frozen=True, slots=True)
class ProcedureStats:
    total: int
    draft: int
    active: int
    deprecated: int
    executions: int
    completed: int
    escalated: int
