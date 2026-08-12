"""Rich terminal output for EvalSet.summary() and EvalResults.summary()."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from litmus.models import EvalResults, EvalSet

_console = Console()


def print_eval_set_summary(eval_set: EvalSet) -> None:
    _console.print(f"\n[bold]Eval Set[/bold] (tier={eval_set.tier}, version={eval_set.version})")
    _console.print(f"  {len(eval_set.records)} records, {len(eval_set.chunks)} chunks")
    _console.print(f"  Document coverage: {eval_set.document_coverage:.0%}")

    table = Table(title="Question Type Distribution")
    table.add_column("Type")
    table.add_column("Count", justify="right")
    for qtype, count in sorted(eval_set.question_type_distribution.items(), key=lambda kv: -kv[1]):
        table.add_row(qtype, str(count))
    _console.print(table)

    table2 = Table(title="Noise Profile Distribution")
    table2.add_column("Profile")
    table2.add_column("Count", justify="right")
    for profile, count in sorted(eval_set.noise_distribution.items(), key=lambda kv: -kv[1]):
        table2.add_row(profile, str(count))
    _console.print(table2)


def print_results_summary(results: EvalResults) -> None:
    n = len(results.records)
    _console.print(f"\n[bold]Eval Results[/bold] (tier={results.tier}, {n} records)")
    if n == 0:
        _console.print("  No records.")
        return

    stub_warnings = results.metadata.get("stub_rag_warnings") or []
    for warning in stub_warnings:
        _console.print(f"[bold red]⚠ WARNING:[/bold red] {warning}")
    if stub_warnings:
        _console.print("[red]  The scores below likely do not reflect a real RAG system. See warning(s) above.[/red]\n")

    overall = sum(r.overall_score for r in results.records) / n
    faithfulness = sum(r.faithfulness.score for r in results.records) / n
    correctness = sum(r.correctness.score for r in results.records) / n
    abstention = sum(r.abstention.score for r in results.records) / n
    set_recall = sum(1.0 if r.set_recall else 0.0 for r in results.records) / n

    table = Table(title="Summary")
    table.add_column("Metric")
    table.add_column("Score", justify="right")
    table.add_row("Overall", f"{overall:.2f}")
    table.add_row("Faithfulness", f"{faithfulness:.2f}")
    table.add_row("Correctness", f"{correctness:.2f}")
    table.add_row("Abstention", f"{abstention:.2f}")
    completeness_vals = [r.completeness.score for r in results.records if r.completeness]
    if completeness_vals:
        table.add_row("Completeness", f"{sum(completeness_vals) / len(completeness_vals):.2f}")
    conciseness_vals = [r.conciseness.score for r in results.records if r.conciseness]
    if conciseness_vals:
        table.add_row("Conciseness", f"{sum(conciseness_vals) / len(conciseness_vals):.2f}")
    table.add_row("Set Recall", f"{set_recall:.2f}")
    _console.print(table)

    table2 = Table(title="By Question Type")
    table2.add_column("Type")
    table2.add_column("Count", justify="right")
    table2.add_column("Mean Overall", justify="right")
    for qtype, agg in sorted(results.by_question_type().items(), key=lambda kv: kv[1].mean_overall):
        table2.add_row(qtype.value, str(agg.count), f"{agg.mean_overall:.2f}")
    _console.print(table2)

    failed = results.failed_records()
    if failed:
        _console.print(f"\n[yellow]{len(failed)} records below 0.7 overall score.[/yellow]")
