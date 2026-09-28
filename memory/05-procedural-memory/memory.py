"""Application-owned procedural-memory lifecycle and validation."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from uuid import uuid4

from embeddings import EmbeddingProvider
from models import (
    JSONValue,
    ProcedureSearchResult,
    ProcedureStatus,
    ProcedureStep,
    SupportProcedure,
)
from store import PostgresProcedureStore, StoreConflictError

logger = logging.getLogger(__name__)


class ProcedureValidationError(ValueError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _text(name: str, value: str, maximum: int = 2_000) -> str:
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ProcedureValidationError(f"{name} cannot be empty")
    if len(cleaned) > maximum:
        raise ProcedureValidationError(f"{name} cannot exceed {maximum} characters")
    return cleaned


def _key(value: str) -> str:
    key = _text("procedure_key", value, 100).upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
        raise ProcedureValidationError(
            "procedure_key must use uppercase letters, numbers, and underscores"
        )
    return key


class ProceduralMemoryManager:
    def __init__(
        self,
        store: PostgresProcedureStore,
        embedding_provider: EmbeddingProvider,
        *,
        max_top_k: int = 25,
    ) -> None:
        self.store, self.embedding_provider, self.max_top_k = (
            store,
            embedding_provider,
            max_top_k,
        )

    @staticmethod
    def build_summary(
        name: str,
        description: str,
        category: str,
        steps: Sequence[ProcedureStep],
        escalation_rules: Sequence[str],
        issue_patterns: Sequence[str],
    ) -> str:
        actions = ", ".join(step.action.replace("_", " ") for step in steps)
        return " ".join(
            (
                f"{name}. {description}",
                f"Category: {category}.",
                "Use for: " + "; ".join(issue_patterns) + ".",
                f"Important actions: {actions}.",
                "Escalate for: " + "; ".join(escalation_rules) + ".",
            )
        )

    async def create_procedure(
        self,
        *,
        procedure_key: str,
        name: str,
        description: str,
        category: str,
        steps: Sequence[ProcedureStep | Mapping[str, object]],
        preconditions: Sequence[str] = (),
        completion_conditions: Sequence[str],
        escalation_rules: Sequence[str],
        issue_patterns: Sequence[str],
        metadata: Mapping[str, JSONValue] | None = None,
        version: int = 1,
        status: ProcedureStatus | str = ProcedureStatus.DRAFT,
    ) -> SupportProcedure:
        if version <= 0:
            raise ProcedureValidationError("version must be positive")
        normalized_steps = self._steps(steps)
        conditions = tuple(
            _text("completion condition", x, 300) for x in completion_conditions
        )
        escalations = tuple(_text("escalation rule", x, 300) for x in escalation_rules)
        patterns = tuple(_text("issue pattern", x, 300) for x in issue_patterns)
        if not conditions or not escalations or not patterns:
            raise ProcedureValidationError(
                "completion conditions, escalation rules, and issue patterns are required"
            )
        clean_name, clean_description, clean_category = (
            _text("name", name, 200),
            _text("description", description),
            _text("category", category, 100),
        )
        summary = self.build_summary(
            clean_name,
            clean_description,
            clean_category,
            normalized_steps,
            escalations,
            patterns,
        )
        embedding = await self.embedding_provider.embed(summary)
        now = _now()
        procedure = SupportProcedure(
            str(uuid4()),
            _key(procedure_key),
            version,
            clean_name,
            clean_description,
            clean_category,
            summary,
            ProcedureStatus(status),
            normalized_steps,
            tuple(_text("precondition", x, 300) for x in preconditions),
            conditions,
            escalations,
            dict(metadata or {}) | {"issue_patterns": list(patterns)},
            now,
            now,
        )
        try:
            created = await self.store.create(procedure, embedding)
        except StoreConflictError as error:
            raise ProcedureValidationError(str(error)) from error
        logger.info(
            "procedure_created key=%s version=%s status=%s",
            created.procedure_key,
            created.version,
            created.status.value,
        )
        return created

    async def create_new_version(
        self, procedure_key: str, **overrides: object
    ) -> SupportProcedure:
        key = _key(procedure_key)
        versions = await self.store.list(key=key)
        if not versions:
            raise ProcedureValidationError("procedure not found")
        source = versions[0]
        data = dict(
            procedure_key=key,
            version=await self.store.next_version(key),
            name=source.name,
            description=source.description,
            category=source.category,
            steps=source.steps,
            preconditions=source.preconditions,
            completion_conditions=source.completion_conditions,
            escalation_rules=source.escalation_rules,
            issue_patterns=source.metadata.get("issue_patterns", [source.description]),
            metadata=source.metadata,
            status=ProcedureStatus.DRAFT,
        )
        data.update(overrides)
        data["status"] = ProcedureStatus.DRAFT
        created = await self.create_procedure(**data)
        logger.info("procedure_version_created key=%s version=%s", key, created.version)
        return created

    def _steps(
        self, values: Sequence[ProcedureStep | Mapping[str, object]]
    ) -> tuple[ProcedureStep, ...]:
        steps = []
        for index, value in enumerate(values, 1):
            step = (
                value
                if isinstance(value, ProcedureStep)
                else ProcedureStep(
                    sequence=int(value.get("sequence", index)),
                    action=str(value.get("action", "")),
                    tool=str(value["tool"]) if value.get("tool") else None,
                    required=bool(value.get("required", True)),
                )
            )
            if step.sequence != index:
                raise ProcedureValidationError(
                    "steps must have consecutive sequence numbers starting at 1"
                )
            steps.append(
                ProcedureStep(
                    index,
                    _text("step action", step.action, 100),
                    step.tool,
                    step.required,
                )
            )
        if not steps:
            raise ProcedureValidationError("at least one procedure step is required")
        return tuple(steps)

    async def get_procedure(self, key: str, version: int) -> SupportProcedure | None:
        return await self.store.get(_key(key), version)

    async def get_active_version(self, key: str) -> SupportProcedure | None:
        return await self.store.get_active(_key(key))

    async def search_procedures(
        self,
        query: str,
        *,
        top_k: int = 3,
        category: str | None = None,
        min_similarity: float | None = None,
    ) -> list[ProcedureSearchResult]:
        query = _text("query", query)
        if not 1 <= top_k <= self.max_top_k:
            raise ProcedureValidationError(f"top_k must be from 1 to {self.max_top_k}")
        if min_similarity is not None and not -1 <= min_similarity <= 1:
            raise ProcedureValidationError("min_similarity must be between -1 and 1")
        results = await self.store.search(
            await self.embedding_provider.embed(query),
            top_k=top_k,
            category=category,
            min_similarity=min_similarity,
        )
        logger.info(
            "procedure_search result_count=%s category=%s",
            len(results),
            category or "all",
        )
        return results

    async def activate_procedure(self, key: str, version: int) -> SupportProcedure:
        result = await self._status(key, version, ProcedureStatus.ACTIVE)
        logger.info(
            "procedure_activated key=%s version=%s", result.procedure_key, version
        )
        return result

    async def deactivate_procedure(self, key: str, version: int) -> SupportProcedure:
        result = await self._status(key, version, ProcedureStatus.DRAFT)
        logger.info(
            "procedure_deactivated key=%s version=%s", result.procedure_key, version
        )
        return result

    async def deprecate_procedure(self, key: str, version: int) -> SupportProcedure:
        return await self._status(key, version, ProcedureStatus.DEPRECATED)

    async def _status(
        self, key: str, version: int, status: ProcedureStatus
    ) -> SupportProcedure:
        try:
            return await self.store.set_status(_key(key), version, status, _now())
        except StoreConflictError as error:
            raise ProcedureValidationError(str(error)) from error

    async def list_procedures(
        self, *, key: str | None = None, status: ProcedureStatus | None = None
    ) -> list[SupportProcedure]:
        return await self.store.list(key=_key(key) if key else None, status=status)

    async def delete_procedure(self, key: str, version: int) -> bool:
        return await self.store.delete(_key(key), version)

    async def count(self):
        return await self.store.stats()
