"""Diff two EvalResults runs (Section on Comparison Between Runs)."""

from __future__ import annotations

from rich.console import Console

from litmus.models import ComparisonReport, EvalResults

_console = Console()


def _dim_mean(results: EvalResults, dim: str) -> float:
    values = []
    for r in results.records:
        score = getattr(r, dim)
        if score is not None:
            values.append(score.score)
    return sum(values) / len(values) if values else 0.0


def compare_results(new: EvalResults, old: EvalResults) -> ComparisonReport:
    new_by_id = {r.record_id: r for r in new.records}
    old_by_id = {r.record_id: r for r in old.records}
    common_ids = set(new_by_id) & set(old_by_id)

    def mean_overall(results: EvalResults) -> float:
        return sum(r.overall_score for r in results.records) / len(results.records) if results.records else 0.0

    overall_delta = mean_overall(new) - mean_overall(old)

    dimension_deltas = {}
    for dim in ("faithfulness", "correctness", "abstention", "completeness", "conciseness"):
        new_mean = _dim_mean(new, dim)
        old_mean = _dim_mean(old, dim)
        if new_mean or old_mean:
            dimension_deltas[dim] = new_mean - old_mean

    def mean_set_recall(results: EvalResults) -> float:
        n = len(results.records) or 1
        return sum(1.0 if r.set_recall else 0.0 for r in results.records) / n

    retrieval_deltas = {"set_recall": mean_set_recall(new) - mean_set_recall(old)}

    by_qtype_deltas = {}
    new_by_type = new.by_question_type()
    old_by_type = old.by_question_type()
    for qtype in set(new_by_type) | set(old_by_type):
        new_score = new_by_type[qtype].mean_overall if qtype in new_by_type else None
        old_score = old_by_type[qtype].mean_overall if qtype in old_by_type else None
        if new_score is not None and old_score is not None:
            by_qtype_deltas[qtype.value] = new_score - old_score

    by_noise_deltas = {}
    new_by_noise = new.by_noise_profile()
    old_by_noise = old.by_noise_profile()
    for profile in set(new_by_noise) | set(old_by_noise):
        new_score = new_by_noise[profile].mean_overall if profile in new_by_noise else None
        old_score = old_by_noise[profile].mean_overall if profile in old_by_noise else None
        if new_score is not None and old_score is not None:
            by_noise_deltas[profile] = new_score - old_score

    flipped_pass_to_fail = []
    flipped_fail_to_pass = []
    for rid in common_ids:
        old_pass = old_by_id[rid].overall_score >= 0.7
        new_pass = new_by_id[rid].overall_score >= 0.7
        if old_pass and not new_pass:
            flipped_pass_to_fail.append(rid)
        elif not old_pass and new_pass:
            flipped_fail_to_pass.append(rid)

    return ComparisonReport(
        overall_delta=overall_delta,
        dimension_deltas=dimension_deltas,
        retrieval_deltas=retrieval_deltas,
        by_question_type_deltas=by_qtype_deltas,
        by_noise_profile_deltas=by_noise_deltas,
        flipped_pass_to_fail=flipped_pass_to_fail,
        flipped_fail_to_pass=flipped_fail_to_pass,
    )


def _arrow(delta: float) -> str:
    if delta > 0.001:
        return "[green]↑[/green]"
    if delta < -0.001:
        return "[red]↓[/red]"
    return "→"


def print_comparison_summary(report: ComparisonReport) -> None:
    _console.print(f"\n[bold]Comparison[/bold]")
    _console.print(f"  Overall: {report.overall_delta:+.2f} {_arrow(report.overall_delta)}")
    for dim, delta in report.dimension_deltas.items():
        _console.print(f"  {dim.capitalize()}: {delta:+.2f} {_arrow(delta)}")
    for metric, delta in report.retrieval_deltas.items():
        _console.print(f"  Retrieval {metric}: {delta:+.2f} {_arrow(delta)}")

    if report.by_question_type_deltas:
        improvements = sorted(report.by_question_type_deltas.items(), key=lambda kv: -kv[1])[:3]
        regressions = sorted(report.by_question_type_deltas.items(), key=lambda kv: kv[1])[:3]
        _console.print("\n  Biggest improvements:")
        for qtype, delta in improvements:
            if delta > 0:
                _console.print(f"    {qtype}: {delta:+.2f}")
        _console.print("  Regressions:")
        for qtype, delta in regressions:
            if delta < 0:
                _console.print(f"    {qtype}: {delta:+.2f}")

    if report.flipped_fail_to_pass or report.flipped_pass_to_fail:
        _console.print(f"\n  Records that flipped:")
        _console.print(f"    {len(report.flipped_fail_to_pass)} records: fail -> pass")
        _console.print(f"    {len(report.flipped_pass_to_fail)} records: pass -> fail")
