"""Application-controlled execution of approved structured procedures."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from models import (
    ExecutionStatus,
    ProcedureExecution,
    ProcedureStatus,
    StepStatus,
    SupportProcedure,
)
from store import PostgresProcedureStore, StoreConflictError
from tools import SupportToolError, SupportTools

logger = logging.getLogger(__name__)
ToolObserver = Callable[[str, dict[str, Any], dict[str, Any]], None]


class ExecutionError(RuntimeError):
    pass


class ProcedureExecutor:
    def __init__(
        self,
        store: PostgresProcedureStore,
        tools: SupportTools,
        *,
        on_tool_event: ToolObserver | None = None,
    ) -> None:
        self.store, self.tools, self.on_tool_event = store, tools, on_tool_event

    async def execute(
        self, procedure: SupportProcedure, issue: str
    ) -> tuple[ProcedureExecution, list[dict[str, Any]]]:
        if procedure.status is not ProcedureStatus.ACTIVE:
            raise ExecutionError("only an approved active procedure can execute")
        transaction = re.search(r"\bTXN-[A-Z0-9-]+\b", issue.upper())
        execution = ProcedureExecution(
            str(uuid4()),
            procedure.id,
            procedure.procedure_key,
            procedure.version,
            ExecutionStatus.RUNNING,
            0,
            datetime.now(timezone.utc),
            None,
            None,
            {"channel": "cli"},
            (),
        )
        try:
            execution = await self.store.start_execution(execution)
        except StoreConflictError as error:
            raise ExecutionError(str(error)) from error
        logger.info(
            "execution_started id=%s procedure=%s version=%s",
            execution.id,
            procedure.procedure_key,
            procedure.version,
        )
        observations: list[dict[str, Any]] = []
        if transaction is None:
            result = self.tools.escalate_case("transaction identifier is required")
            return (
                await self._terminal(
                    execution.id, ExecutionStatus.ESCALATED, result["reason"]
                ),
                observations,
            )
        transaction_id = transaction.group(0)
        context: dict[str, dict[str, Any]] = {}
        for step in procedure.steps:
            started = datetime.now(timezone.utc)
            await self.store.start_step(
                execution.id, step.sequence, step.action, step.tool, started
            )
            logger.info(
                "step_started execution_id=%s step=%s action=%s",
                execution.id,
                step.sequence,
                step.action,
            )
            try:
                result = self._run_step(step.action, step.tool, transaction_id, context)
            except (SupportToolError, TypeError, KeyError) as error:
                await self.store.finish_step(
                    execution.id,
                    step.sequence,
                    StepStatus.FAILED,
                    str(error),
                    datetime.now(timezone.utc),
                )
                return (
                    await self._terminal(
                        execution.id, ExecutionStatus.ESCALATED, str(error)
                    ),
                    observations,
                )
            context[step.action] = result
            summary = json.dumps(result, ensure_ascii=False, sort_keys=True)
            await self.store.finish_step(
                execution.id,
                step.sequence,
                StepStatus.COMPLETED,
                summary,
                datetime.now(timezone.utc),
            )
            logger.info(
                "step_completed execution_id=%s step=%s", execution.id, step.sequence
            )
            observations.append(
                {
                    "step": step.sequence,
                    "action": step.action,
                    "tool": step.tool,
                    "result": result,
                }
            )
            if step.tool and self.on_tool_event:
                self.on_tool_event(
                    step.tool,
                    self._arguments(step.tool, transaction_id, context),
                    result,
                )
            if self._must_escalate(step.action, result):
                reason = "procedure escalation rule reached: inconsistent or unavailable support state"
                return (
                    await self._terminal(
                        execution.id, ExecutionStatus.ESCALATED, reason
                    ),
                    observations,
                )
        return (
            await self._terminal(
                execution.id,
                ExecutionStatus.COMPLETED,
                "procedure completion conditions satisfied",
            ),
            observations,
        )

    def _arguments(
        self, tool: str, transaction_id: str, context: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        if tool in {
            "check_transaction",
            "check_gateway",
            "check_refund",
            "issue_refund",
        }:
            return {"transaction_id": transaction_id}
        if tool == "notify_customer":
            return {"message": self._customer_message(transaction_id, context)}
        if tool == "escalate_case":
            return {"reason": "approved procedure escalation rule reached"}
        raise SupportToolError(f"tool is not approved for procedural execution: {tool}")

    def _run_step(
        self,
        action: str,
        tool: str | None,
        transaction_id: str,
        context: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if action == "identify_transaction":
            return {"transaction_id": transaction_id, "identified": True}
        if action == "prevent_duplicate_refund":
            refund = context.get("check_refund") or self.tools.check_refund(
                transaction_id
            )
            return {
                "safe_to_issue": refund.get("state") in {"none", "failed", "cancelled"},
                "refund_state": refund.get("state"),
            }
        if tool is None:
            return {"completed": True}
        if tool == "issue_refund":
            refund = context.get("check_refund")
            if refund is None:
                raise SupportToolError("check_refund must complete before issue_refund")
            if refund.get("state") not in {"none", "failed", "cancelled"}:
                return {"issued": False, "reason": "refund already exists", **refund}
        return self.tools.execute(tool, self._arguments(tool, transaction_id, context))

    @staticmethod
    def _customer_message(
        transaction_id: str, context: dict[str, dict[str, Any]]
    ) -> str:
        refund = context.get("issue_refund") or context.get("check_refund", {})
        state, eta = refund.get("state", "unknown"), refund.get("eta_business_days")
        suffix = (
            f" Expected processing time: {eta} business days."
            if eta is not None
            else " No verified ETA is available."
        )
        return f"Transaction {transaction_id}: refund status is {state}.{suffix}"

    @staticmethod
    def _must_escalate(action: str, result: dict[str, Any]) -> bool:
        return (action == "check_transaction" and result.get("state") == "unknown") or (
            action == "check_gateway" and result.get("result") == "unavailable"
        )

    async def _terminal(
        self, execution_id: str, status: ExecutionStatus, outcome: str
    ) -> ProcedureExecution:
        result = await self.store.finish_execution(
            execution_id, status, outcome, datetime.now(timezone.utc)
        )
        logger.info("execution_%s id=%s", status.value, execution_id)
        return result
