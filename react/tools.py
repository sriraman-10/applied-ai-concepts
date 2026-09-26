"""Deterministic mock observability tools for the incident scenario."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

ToolResult = dict[str, Any]


def get_service_metrics(service: str) -> ToolResult:
    """Return current latency, CPU, and database metrics for a service."""
    metrics = {
        "checkout-service": {
            "p95_latency_ms": 1850,
            "baseline_p95_latency_ms": 240,
            "cpu_percent": 42,
            "database_latency_ms": 18,
            "status": "degraded",
        },
        "payment-service": {
            "p95_latency_ms": 1720,
            "baseline_p95_latency_ms": 210,
            "cpu_percent": 39,
            "database_latency_ms": 21,
            "status": "degraded",
        },
        "fraud-service": {
            "p95_latency_ms": 1680,
            "baseline_p95_latency_ms": 120,
            "cpu_percent": 76,
            "database_latency_ms": 16,
            "status": "degraded",
        },
    }
    return metrics.get(service, {"error": f"Unknown service: {service}"})


def get_recent_deployments(service: str) -> ToolResult:
    """Return recent deployments for a service."""
    deployments = {
        "checkout-service": {"deployments_last_24h": [], "latest": None},
        "payment-service": {
            "deployments_last_24h": [],
            "latest": "2026-09-18T09:15:00Z",
        },
        "fraud-service": {
            "deployments_last_24h": [],
            "latest": "2026-09-12T14:30:00Z",
        },
    }
    return deployments.get(service, {"error": f"Unknown service: {service}"})


def get_logs(service: str, query: str) -> ToolResult:
    """Search deterministic mock logs for a service and query."""
    normalized_query = query.lower()
    if service == "payment-service" and any(
        term in normalized_query for term in ("timeout", "error", "fraud")
    ):
        return {
            "matches": 37,
            "entries": [
                {
                    "level": "ERROR",
                    "message": "fraud-service request timed out after 1500ms",
                    "dependency": "fraud-service",
                },
                {
                    "level": "WARN",
                    "message": "payment authorization delayed waiting for fraud check",
                    "dependency": "fraud-service",
                },
            ],
        }
    if service == "checkout-service":
        return {
            "matches": 6,
            "entries": [
                {
                    "level": "WARN",
                    "message": "checkout request waiting on payment-service",
                    "dependency": "payment-service",
                }
            ],
        }
    return {"matches": 0, "entries": []}


def get_dependency_health(service: str) -> ToolResult:
    """Return the health of a service's direct dependencies."""
    health = {
        "checkout-service": {
            "dependencies": [
                {"service": "inventory-service", "status": "healthy", "p95_latency_ms": 55},
                {"service": "payment-service", "status": "degraded", "p95_latency_ms": 1720},
            ]
        },
        "payment-service": {
            "dependencies": [
                {"service": "payment-db", "status": "healthy", "p95_latency_ms": 21},
                {"service": "fraud-service", "status": "degraded", "p95_latency_ms": 1680},
            ]
        },
        "fraud-service": {
            "dependencies": [
                {"service": "fraud-db", "status": "healthy", "p95_latency_ms": 16}
            ]
        },
    }
    return health.get(service, {"error": f"Unknown service: {service}"})


TOOL_FUNCTIONS: dict[str, Callable[..., ToolResult]] = {
    "get_service_metrics": get_service_metrics,
    "get_recent_deployments": get_recent_deployments,
    "get_logs": get_logs,
    "get_dependency_health": get_dependency_health,
}


def call_tool(name: str, arguments: dict[str, Any]) -> ToolResult:
    """Dispatch a named mock tool and return structured data."""
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return function(**arguments)
    except TypeError as error:
        return {"error": f"Invalid arguments for {name}: {error}"}


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Dispatch a tool and serialize its result for a model API."""
    return json.dumps(call_tool(name, arguments), sort_keys=True)
