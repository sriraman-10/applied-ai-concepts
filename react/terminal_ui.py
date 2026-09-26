"""Rich terminal output and evidence validation shared by both agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()


@dataclass(frozen=True)
class ToolEvent:
    """One observable tool call and its structured result."""

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass(frozen=True)
class EvidenceRequirement:
    label: str
    tool: str
    service: str
    validates: Callable[[dict[str, Any]], bool]


def _degraded_metrics(result: dict[str, Any]) -> bool:
    return (
        result.get("status") == "degraded"
        and isinstance(result.get("p95_latency_ms"), (int, float))
        and isinstance(result.get("baseline_p95_latency_ms"), (int, float))
        and result["p95_latency_ms"] > result["baseline_p95_latency_ms"]
    )


def _has_dependency(service: str) -> Callable[[dict[str, Any]], bool]:
    def validate(result: dict[str, Any]) -> bool:
        return any(
            item.get("service") == service and item.get("status") == "degraded"
            for item in result.get("dependencies", [])
        )

    return validate


def _payment_timeout(result: dict[str, Any]) -> bool:
    return any(
        entry.get("dependency") == "fraud-service"
        and "timed out" in str(entry.get("message", "")).lower()
        for entry in result.get("entries", [])
    )


REQUIRED_EVIDENCE = (
    EvidenceRequirement("degraded checkout-service metrics", "get_service_metrics", "checkout-service", _degraded_metrics),
    EvidenceRequirement("checkout-service deployment history", "get_recent_deployments", "checkout-service", lambda result: "deployments_last_24h" in result),
    EvidenceRequirement("degraded payment-service dependency", "get_dependency_health", "checkout-service", _has_dependency("payment-service")),
    EvidenceRequirement("degraded fraud-service dependency", "get_dependency_health", "payment-service", _has_dependency("fraud-service")),
    EvidenceRequirement("payment-service timeout logs naming fraud-service", "get_logs", "payment-service", _payment_timeout),
    EvidenceRequirement("degraded fraud-service metrics", "get_service_metrics", "fraud-service", _degraded_metrics),
)


def show_start(query: str, mode: str) -> None:
    console.print(
        Panel(
            f"[bold]{query}[/bold]\n[dim]{mode}[/dim]",
            title="[bold cyan]ReAct Incident Investigation[/bold cyan]",
            border_style="cyan",
        )
    )


def show_agent_phase() -> None:
    console.rule("[bold cyan]Phase 1 · Agent investigation[/bold cyan]")


def show_iteration(iteration: int) -> None:
    console.rule(f"[bold blue]Model turn {iteration}[/bold blue]")


def show_continuation(missing: list[str], round_number: int) -> None:
    details = "\n".join(f"• {item}" for item in missing)
    console.print(
        Panel(
            "The model returned an answer before the evidence contract was met.\n"
            f"Starting bounded continuation {round_number}.\n\nMissing:\n{details}",
            title="Evidence gate",
            border_style="yellow",
        )
    )


def show_error(message: str) -> None:
    console.print(Panel(message, title="Run failed", border_style="red"))


def _target(arguments: dict[str, Any]) -> str:
    service = str(arguments.get("service", "—"))
    query = arguments.get("query")
    return service if not query else f"{service}\n[dim]query: {query}[/dim]"


def _result_summary(result: dict[str, Any]) -> tuple[str, str]:
    if "error" in result:
        return str(result["error"]), "red"
    if "p95_latency_ms" in result:
        summary = (
            f"p95 [bold]{result['p95_latency_ms']} ms[/bold] "
            f"(baseline {result['baseline_p95_latency_ms']} ms)\n"
            f"CPU {result['cpu_percent']}% · DB {result['database_latency_ms']} ms"
        )
        return summary, "red" if result.get("status") == "degraded" else "green"
    if "dependencies" in result:
        dependencies = result["dependencies"]
        summary = "\n".join(
            f"{item['service']}: {item['p95_latency_ms']} ms "
            f"([{'red' if item['status'] == 'degraded' else 'green'}]{item['status']}[/])"
            for item in dependencies
        )
        color = "yellow" if any(item["status"] == "degraded" for item in dependencies) else "green"
        return summary, color
    if "entries" in result:
        entries = result["entries"]
        summary = f"{result.get('matches', len(entries))} matches"
        if entries:
            summary += "\n" + "\n".join(str(item["message"]) for item in entries)
        return summary, "yellow" if entries else "green"
    if "deployments_last_24h" in result:
        count = len(result["deployments_last_24h"])
        return f"{count} deployments in the last 24h", "green" if count == 0 else "yellow"
    return str(result), "white"


def show_tool_event(event: ToolEvent) -> None:
    summary, color = _result_summary(event.result)
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("Action", style="cyan", no_wrap=True)
    table.add_column("Target", style="white")
    table.add_column("Observation")
    table.add_row(event.name, _target(event.arguments), f"[{color}]{summary}[/{color}]")
    console.print(table)


def _matching_events(events: list[ToolEvent], requirement: EvidenceRequirement) -> list[ToolEvent]:
    return [
        event
        for event in events
        if event.name == requirement.tool
        and event.arguments.get("service") == requirement.service
    ]


def missing_evidence(events: list[ToolEvent]) -> list[str]:
    """Return requirements that no observed, successful result satisfies."""
    return [
        requirement.label
        for requirement in REQUIRED_EVIDENCE
        if not any(requirement.validates(event.result) for event in _matching_events(events, requirement))
    ]


def _find_event(events: list[ToolEvent], name: str, service: str) -> ToolEvent | None:
    return next(
        (
            event
            for event in reversed(events)
            if event.name == name and event.arguments.get("service") == service
        ),
        None,
    )


def evidence_rows(events: list[ToolEvent]) -> list[tuple[str, str]]:
    """Build concise rows solely from evidence observed during this run."""
    rows: list[tuple[str, str]] = []
    checkout = _find_event(events, "get_service_metrics", "checkout-service")
    deployments = _find_event(events, "get_recent_deployments", "checkout-service")
    checkout_dependencies = _find_event(events, "get_dependency_health", "checkout-service")
    payment_dependencies = _find_event(events, "get_dependency_health", "payment-service")
    payment_logs = _find_event(events, "get_logs", "payment-service")
    fraud = _find_event(events, "get_service_metrics", "fraud-service")

    if checkout and "error" not in checkout.result:
        data = checkout.result
        rows.append(("Checkout", f"p95 {data['p95_latency_ms']} ms vs {data['baseline_p95_latency_ms']} ms baseline; DB {data['database_latency_ms']} ms"))
    if deployments and "deployments_last_24h" in deployments.result:
        rows.append(("Deployments", f"{len(deployments.result['deployments_last_24h'])} checkout deployments in the last 24h"))
    if checkout_dependencies:
        payment = next((item for item in checkout_dependencies.result.get("dependencies", []) if item.get("service") == "payment-service"), None)
        if payment:
            rows.append(("Payment path", f"payment-service is {payment['status']} at {payment['p95_latency_ms']} ms"))
    if payment_dependencies:
        fraud_dependency = next((item for item in payment_dependencies.result.get("dependencies", []) if item.get("service") == "fraud-service"), None)
        if fraud_dependency:
            rows.append(("Fraud path", f"fraud-service is {fraud_dependency['status']} at {fraud_dependency['p95_latency_ms']} ms"))
    if payment_logs:
        entries = payment_logs.result.get("entries", [])
        if entries:
            rows.append(("Payment logs", str(entries[0]["message"])))
    if fraud and "error" not in fraud.result:
        data = fraud.result
        rows.append(("Fraud", f"p95 {data['p95_latency_ms']} ms vs {data['baseline_p95_latency_ms']} ms baseline; DB {data['database_latency_ms']} ms"))
    return rows


def evidence_verdict(events: list[ToolEvent]) -> tuple[str, str]:
    """Return the strongest conclusion justified by the observed results."""
    if missing_evidence(events):
        return (
            "INCONCLUSIVE",
            "The investigation stopped before collecting the complete evidence path required for a reliable root-cause verdict.",
        )
    return (
        "SUPPORTED",
        "fraud-service is the likely downstream bottleneck. Its elevated latency propagates through payment-service to checkout-service. The available evidence does not establish why fraud-service itself is slow.",
    )


def show_final(model_answer: str, events: list[ToolEvent]) -> None:
    console.rule("[bold magenta]Phase 2 · Evidence validation[/bold magenta]")

    table = Table(title="Evidence collected by the agent", header_style="bold magenta")
    table.add_column("Signal", style="cyan", no_wrap=True)
    table.add_column("Observed evidence")
    rows = evidence_rows(events)
    if rows:
        for signal, observation in rows:
            table.add_row(signal, observation)
    else:
        table.add_row("—", "No tool evidence was collected")
    console.print(table)

    missing = missing_evidence(events)
    if missing:
        details = "\n".join(f"• {item}" for item in missing)
        console.print(
            Panel(
                f"[bold yellow]INCOMPLETE[/bold yellow]\nMissing evidence:\n{details}",
                title="Evidence validation",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(
                "[bold green]PASS[/bold green]\nAll required observations support the dependency path.",
                title="Evidence validation",
                border_style="green",
            )
        )

    if model_answer.strip():
        console.print(
            Panel(Markdown(model_answer.strip()), title="Model draft", border_style="blue")
        )

    label, verdict = evidence_verdict(events)
    style = "green" if label == "SUPPORTED" else "yellow"
    console.print(
        Panel(
            f"[bold {style}]{label}[/bold {style}]\n{verdict}",
            title="Verdict",
            border_style=style,
        )
    )
