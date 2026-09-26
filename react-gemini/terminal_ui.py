"""Shared Rich output for observable agent actions and evidence validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table


console = Console()

REQUIRED_EVIDENCE = [
    ("checkout-service metrics", "get_service_metrics", "checkout-service"),
    ("checkout-service dependencies", "get_dependency_health", "checkout-service"),
    ("payment-service dependencies", "get_dependency_health", "payment-service"),
    ("payment-service logs mentioning fraud-service", "get_logs", "payment-service"),
    ("fraud-service metrics", "get_service_metrics", "fraud-service"),
]


@dataclass
class ToolEvent:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


def show_start(query: str, mode: str) -> None:
    console.print(
        Panel(
            f"[bold]{query}[/bold]\n[dim]{mode}[/dim]",
            title="[bold cyan]ReAct Incident Investigation[/bold cyan]",
            border_style="cyan",
        )
    )


def show_iteration(iteration: int) -> None:
    console.rule(f"[bold blue]Iteration {iteration}[/bold blue]")


def show_agent_phase() -> None:
    console.rule("[bold cyan]Phase 1 · Agent investigation[/bold cyan]")


def show_continuation(missing: list[str], round_number: int) -> None:
    details = "\n".join(f"• {item}" for item in missing)
    console.print(
        Panel(
            f"The agent tried to finish before meeting the evidence contract.\n"
            f"Requesting investigation round {round_number}.\n\nMissing:\n{details}",
            title="Agent continuation",
            border_style="yellow",
        )
    )


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
        return summary, "yellow" if any(d["status"] == "degraded" for d in dependencies) else "green"
    if "entries" in result:
        entries = result["entries"]
        summary = f"{result.get('matches', len(entries))} matches"
        if entries:
            summary += "\n" + "\n".join(item["message"] for item in entries)
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


def _find_event(events: list[ToolEvent], name: str, service: str) -> ToolEvent | None:
    return next(
        (event for event in events if event.name == name and event.arguments.get("service") == service),
        None,
    )


def missing_evidence(events: list[ToolEvent]) -> list[str]:
    """Return human-readable requirements not satisfied by agent observations."""
    missing: list[str] = []
    for label, name, service in REQUIRED_EVIDENCE:
        event = _find_event(events, name, service)
        if event is None:
            missing.append(label)
        elif name == "get_logs" and not event.result.get("entries"):
            missing.append(label + " with matching entries")
    return missing


def evidence_verdict(events: list[ToolEvent]) -> tuple[str, str, list[tuple[str, str]]]:
    """Derive a cautious verdict only from tool observations in this run."""
    checkout = _find_event(events, "get_service_metrics", "checkout-service")
    checkout_deps = _find_event(events, "get_dependency_health", "checkout-service")
    payment_deps = _find_event(events, "get_dependency_health", "payment-service")
    payment_logs = _find_event(events, "get_logs", "payment-service")
    fraud = _find_event(events, "get_service_metrics", "fraud-service")

    evidence: list[tuple[str, str]] = []
    if checkout:
        data = checkout.result
        evidence.append(("Checkout", f"p95 {data['p95_latency_ms']} ms vs {data['baseline_p95_latency_ms']} ms baseline; local DB {data['database_latency_ms']} ms"))
    if checkout_deps:
        payment = next((d for d in checkout_deps.result.get("dependencies", []) if d["service"] == "payment-service"), None)
        if payment:
            evidence.append(("Payment path", f"payment-service is {payment['status']} at {payment['p95_latency_ms']} ms"))
    if payment_deps:
        fraud_dependency = next(
            (d for d in payment_deps.result.get("dependencies", []) if d["service"] == "fraud-service"),
            None,
        )
        if fraud_dependency:
            evidence.append(("Fraud path", f"fraud-service is {fraud_dependency['status']} at {fraud_dependency['p95_latency_ms']} ms"))
    if payment_logs:
        entries = payment_logs.result.get("entries", [])
        if entries:
            evidence.append(("Payment logs", entries[0]["message"]))
    if fraud:
        data = fraud.result
        evidence.append(("Fraud", f"p95 {data['p95_latency_ms']} ms vs {data['baseline_p95_latency_ms']} ms baseline; local DB {data['database_latency_ms']} ms"))

    complete = not missing_evidence(events)
    if complete:
        return (
            "SUPPORTED CONCLUSION",
            "fraud-service is the likely bottleneck. Its elevated latency propagates through payment-service to checkout-service. The available evidence does not establish why fraud-service itself is slow.",
            evidence,
        )
    return (
        "INCONCLUSIVE",
        "The agent stopped before collecting the complete evidence path needed for a reliable root-cause verdict.",
        evidence,
    )


def _unsupported_claims(model_answer: str) -> list[str]:
    answer = model_answer.lower()
    phrases = {
        "cpu saturation": "CPU saturation",
        "cpu bottleneck": "CPU bottleneck",
        "database bottleneck": "database bottleneck",
        "resource saturation": "resource saturation",
    }
    return [label for phrase, label in phrases.items() if phrase in answer]


def show_final(model_answer: str, events: list[ToolEvent]) -> None:
    console.rule("[bold magenta]Phase 2 · Evidence validation[/bold magenta]")
    label, verdict, evidence = evidence_verdict(events)
    table = Table(title="Evidence collected by the agent", header_style="bold magenta")
    table.add_column("Signal", style="cyan", no_wrap=True)
    table.add_column("Observed evidence")
    for signal, observation in evidence:
        table.add_row(signal, observation)
    console.print(table)

    missing = missing_evidence(events)
    unsupported = _unsupported_claims(model_answer)
    if missing:
        details = "\n".join(f"• {item}" for item in missing)
        console.print(
            Panel(
                f"[bold yellow]INCOMPLETE[/bold yellow]\nMissing evidence:\n{details}",
                title="Evidence validation",
                border_style="yellow",
            )
        )
    elif unsupported:
        details = ", ".join(unsupported)
        console.print(
            Panel(
                f"[bold red]UNSUPPORTED CLAIM[/bold red]\nRejected claim: {details}",
                title="Evidence validation",
                border_style="red",
            )
        )
    else:
        console.print(
            Panel(
                "[bold green]PASS[/bold green]\nMinimum evidence collected; no unsupported deeper-cause claim detected.",
                title="Evidence validation",
                border_style="green",
            )
        )

    # Preserve the model's answer when validated. Replace unsupported wording
    # with the narrow conclusion justified by the observations.
    final_answer = verdict if unsupported else (model_answer or verdict)
    title = "Final answer" if not missing else "Agent final answer (evidence incomplete)"
    console.print(Panel(Markdown(final_answer), title=title, border_style="blue"))
