"""Application-owned episodic-memory lifecycle."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from embeddings import EmbeddingProvider
from models import (
    Episode,
    EpisodeAction,
    EpisodeObservation,
    EpisodeOutcome,
    EpisodeSearchResult,
    EpisodeStats,
    EpisodeStatus,
    JSONValue,
)
from store import EpisodeStore, StoreConflictError

logger = logging.getLogger(__name__)


class EpisodeValidationError(ValueError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _text(name: str, value: str, maximum: int) -> str:
    cleaned = " ".join(value.split())
    if not cleaned:
        raise EpisodeValidationError(f"{name} cannot be empty")
    if len(cleaned) > maximum:
        raise EpisodeValidationError(f"{name} cannot exceed {maximum} characters")
    return cleaned


def _optional_text(name: str, value: str | None, maximum: int) -> str | None:
    return None if value is None else _text(name, value, maximum)


def _metadata(value: Mapping[str, JSONValue] | None) -> dict[str, JSONValue]:
    result = dict(value or {})
    try:
        encoded = json.dumps(
            result, ensure_ascii=False, allow_nan=False, sort_keys=True
        )
    except (TypeError, ValueError) as error:
        raise EpisodeValidationError(
            "metadata must contain finite JSON values"
        ) from error
    if len(encoded) > 20_000:
        raise EpisodeValidationError("metadata cannot exceed 20,000 encoded characters")
    return result


def _outcome(value: EpisodeOutcome | str) -> EpisodeOutcome:
    try:
        return value if isinstance(value, EpisodeOutcome) else EpisodeOutcome(value)
    except ValueError as error:
        allowed = ", ".join(item.value for item in EpisodeOutcome)
        raise EpisodeValidationError(f"outcome must be one of: {allowed}") from error


def _finite_embedding(values: Sequence[float]) -> list[float]:
    vector = [float(value) for value in values]
    if not vector or not all(math.isfinite(value) for value in vector):
        raise EpisodeValidationError("embedding must be non-empty and finite")
    return vector


def build_episode_summary(
    episode: Episode, resolution: str, outcome: EpisodeOutcome
) -> str:
    """Build a compact search representation from observable episode data."""
    action_text = "; ".join(
        f"{action.sequence}. {action.action_type}"
        + (f" via {action.tool_name}" if action.tool_name else "")
        + (f" → {action.result_summary}" if action.result_summary else "")
        for action in episode.actions
    )
    important = [item for item in episode.observations if item.important]
    selected = important or list(episode.observations)
    observation_text = "; ".join(
        f"{item.sequence}. {item.source}: {item.content}" for item in selected
    )
    parts = [
        f"Issue: {episode.description}",
        f"Issue type: {episode.issue_type}",
        f"Actions: {action_text}",
        f"Key observations: {observation_text}",
        f"Resolution: {resolution}",
        f"Outcome: {outcome.value}",
    ]
    return _text("episode_summary", "\n".join(parts), 8_000)


class EpisodicMemoryManager:
    def __init__(
        self,
        store: EpisodeStore,
        embedding_provider: EmbeddingProvider,
        *,
        max_top_k: int = 25,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider
        self.max_top_k = max_top_k

    async def start_episode(
        self,
        description: str,
        *,
        issue_type: str,
        customer_id: str | None = None,
        episode_key: str | None = None,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> Episode:
        now = _now()
        episode = Episode(
            id=str(uuid4()),
            episode_key=_text(
                "episode_key", episode_key or await self.store.next_episode_key(), 100
            ),
            customer_id=_optional_text("customer_id", customer_id, 200),
            issue_type=_text("issue_type", issue_type, 100),
            description=_text("description", description, 5_000),
            episode_summary=None,
            actions=(),
            observations=(),
            resolution=None,
            outcome=None,
            status=EpisodeStatus.OPEN,
            metadata=_metadata(metadata),
            created_at=now,
            completed_at=None,
        )
        try:
            stored = await self.store.create(episode)
        except StoreConflictError as error:
            raise EpisodeValidationError(str(error)) from error
        logger.info(
            "episode_started episode_key=%s issue_type=%s",
            stored.episode_key,
            stored.issue_type,
        )
        return stored

    async def record_action(
        self,
        episode_key: str,
        *,
        action_type: str,
        tool_name: str | None = None,
        input_summary: str | None = None,
        result_summary: str | None = None,
    ) -> Episode:
        episode = await self._open_episode(episode_key)
        action = EpisodeAction(
            sequence=len(episode.actions) + 1,
            action_type=_text("action_type", action_type, 100),
            tool_name=_optional_text("tool_name", tool_name, 100),
            input_summary=_optional_text("input_summary", input_summary, 1_000),
            result_summary=_optional_text("result_summary", result_summary, 2_000),
            created_at=_now(),
        )
        try:
            result = await self.store.append_action(episode.episode_key, action)
        except StoreConflictError as error:
            raise EpisodeValidationError(str(error)) from error
        logger.info(
            "episode_action_recorded episode_key=%s sequence=%d action_type=%s",
            episode_key,
            action.sequence,
            action.action_type,
        )
        return result

    async def record_observation(
        self,
        episode_key: str,
        *,
        source: str,
        content: str,
        important: bool = True,
    ) -> Episode:
        episode = await self._open_episode(episode_key)
        observation = EpisodeObservation(
            sequence=len(episode.observations) + 1,
            source=_text("source", source, 100),
            content=_text("content", content, 2_000),
            important=bool(important),
            created_at=_now(),
        )
        try:
            result = await self.store.append_observation(
                episode.episode_key, observation
            )
        except StoreConflictError as error:
            raise EpisodeValidationError(str(error)) from error
        logger.info(
            "episode_observation_recorded episode_key=%s sequence=%d source=%s important=%s",
            episode_key,
            observation.sequence,
            observation.source,
            observation.important,
        )
        return result

    async def complete_episode(
        self,
        episode_key: str,
        *,
        resolution: str,
        outcome: EpisodeOutcome | str,
    ) -> Episode:
        episode = await self._open_episode(episode_key)
        if not episode.actions:
            raise EpisodeValidationError(
                "an episode needs at least one recorded action"
            )
        if not episode.observations:
            raise EpisodeValidationError(
                "an episode needs at least one recorded observation"
            )
        clean_resolution = _text("resolution", resolution, 5_000)
        clean_outcome = _outcome(outcome)
        summary = build_episode_summary(episode, clean_resolution, clean_outcome)
        embedding = _finite_embedding(await self.embedding_provider.embed(summary))
        logger.info(
            "episode_embedding_created episode_key=%s summary_length=%d",
            episode_key,
            len(summary),
        )
        try:
            result = await self.store.complete(
                episode_key,
                summary=summary,
                resolution=clean_resolution,
                outcome=clean_outcome,
                completed_at=_now(),
                embedding=embedding,
            )
        except StoreConflictError as error:
            raise EpisodeValidationError(str(error)) from error
        logger.info(
            "episode_completed episode_key=%s outcome=%s",
            episode_key,
            clean_outcome.value,
        )
        return result

    async def get_episode(self, episode_key: str) -> Episode | None:
        result = await self.store.get(_text("episode_key", episode_key, 100))
        logger.info(
            "episode_retrieved episode_key=%s found=%s", episode_key, result is not None
        )
        return result

    async def search_similar(
        self,
        query: str,
        *,
        top_k: int = 5,
        issue_type: str | None = None,
        customer_id: str | None = None,
        min_similarity: float | None = None,
        outcomes: tuple[EpisodeOutcome, ...] = (EpisodeOutcome.RESOLVED,),
    ) -> list[EpisodeSearchResult]:
        clean_query = _text("query", query, 5_000)
        if top_k <= 0 or top_k > self.max_top_k:
            raise EpisodeValidationError(f"top_k must be from 1 to {self.max_top_k}")
        if min_similarity is not None and not -1.0 <= min_similarity <= 1.0:
            raise EpisodeValidationError("min_similarity must be between -1 and 1")
        if not outcomes:
            raise EpisodeValidationError("at least one outcome filter is required")
        started = time.perf_counter()
        vector = _finite_embedding(await self.embedding_provider.embed(clean_query))
        results = await self.store.search(
            vector,
            top_k=top_k,
            issue_type=_optional_text("issue_type", issue_type, 100),
            customer_id=_optional_text("customer_id", customer_id, 200),
            outcomes=outcomes,
            min_similarity=min_similarity,
        )
        logger.info(
            "episode_search query_length=%d top_k=%d issue_type=%s customer_filter=%s outcomes=%s results_count=%d latency_ms=%.2f",
            len(clean_query),
            top_k,
            issue_type,
            customer_id is not None,
            ",".join(item.value for item in outcomes),
            len(results),
            (time.perf_counter() - started) * 1000,
        )
        return results

    async def list_episodes(
        self, *, status: EpisodeStatus | None = None, limit: int = 100
    ) -> list[Episode]:
        if limit <= 0 or limit > 500:
            raise EpisodeValidationError("limit must be from 1 to 500")
        return await self.store.list(status=status, limit=limit)

    async def delete_episode(self, episode_key: str) -> bool:
        clean_key = _text("episode_key", episode_key, 100)
        deleted = await self.store.delete(clean_key)
        logger.info("episode_deleted episode_key=%s deleted=%s", clean_key, deleted)
        return deleted

    async def stats(self) -> EpisodeStats:
        return await self.store.stats()

    async def _open_episode(self, episode_key: str) -> Episode:
        episode = await self.get_episode(episode_key)
        if episode is None:
            raise EpisodeValidationError("episode not found")
        if episode.status is not EpisodeStatus.OPEN:
            raise EpisodeValidationError("completed episodes cannot be changed")
        return episode
