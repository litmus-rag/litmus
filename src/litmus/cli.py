"""CLI entry point: `litmus generate|evaluate|validate|staleness|compare|report|review-export|review-import|estimate`."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Optional

import typer

import litmus

app = typer.Typer(help="litmus: synthetic eval-set generation and scoring for RAG systems")


@app.command()
def estimate(
    docs_dir: str = typer.Argument(..., help="Path to corpus directory"),
    llm: str = typer.Option("azure/gpt-5.4", help="litellm model string"),
    tier: str = typer.Option("medium", help="minimal | medium | exhaustive"),
    size: str = typer.Option("auto", help="'auto' or an exact integer"),
):
    """Estimate cost before running generate()."""
    est = litmus.estimate_cost(docs_dir=docs_dir, llm=llm, tier=tier, size=size)
    typer.echo(str(est))


@app.command()
def generate(
    docs_dir: str = typer.Argument(..., help="Path to corpus directory"),
    llm: str = typer.Option("azure/gpt-5.4", help="litellm model string"),
    tier: str = typer.Option("medium", help="minimal | medium | exhaustive"),
    size: str = typer.Option("auto", help="'auto' or an exact integer"),
    save: str = typer.Option(..., "--save", help="Path to save the generated eval set JSON"),
    max_workers: int = typer.Option(1, help="Parallel LLM calls"),
    seed: Optional[int] = typer.Option(None),
):
    """Generate an eval set from a document corpus."""
    litmus.generate(docs_dir=docs_dir, llm=llm, tier=tier, size=size, save_path=save, max_workers=max_workers, seed=seed)
    typer.echo(f"Saved eval set to {Path(save).resolve()}")


@app.command()
def validate(eval_set_path: str = typer.Argument(..., help="Path to eval set JSON")):
    """Run eval-set self-validation quality checks."""
    eval_set = litmus.load(eval_set_path)
    report = eval_set.validate()
    typer.echo(f"Overall pass: {report.overall_pass}")
    typer.echo(f"Flagged records: {len(report.flagged_records)}")
    for dim, score in report.dimension_scores.items():
        typer.echo(f"  {dim}: {score:.2f}")
    for rec in report.recommendations:
        typer.echo(f"  - {rec}")


@app.command()
def staleness(
    eval_set_path: str = typer.Argument(...),
    docs: str = typer.Option(..., "--docs", help="Path to the (possibly updated) corpus directory"),
):
    """Check whether source documents changed since eval-set generation."""
    eval_set = litmus.load(eval_set_path)
    report = eval_set.check_staleness(docs)
    typer.echo(f"Stale: {report.stale_count}/{report.total_records} ({report.stale_ratio:.0%})")
    for rec in report.recommendations:
        typer.echo(f"  - {rec}")


def _load_rag_callable(spec: str):
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


@app.command()
def evaluate(
    eval_set_path: str = typer.Argument(...),
    rag: str = typer.Option(..., "--rag", help="module:function path to your RAG callable"),
    llm: Optional[str] = typer.Option(None),
    save: str = typer.Option(..., "--save"),
    max_workers: int = typer.Option(1),
    timeout: int = typer.Option(60),
):
    """Evaluate a RAG system against a saved eval set."""
    eval_set = litmus.load(eval_set_path)
    rag_fn = _load_rag_callable(rag)
    results = litmus.evaluate(eval_set, rag=rag_fn, llm=llm, save_path=save, max_workers=max_workers, timeout=timeout)
    results.summary()
    typer.echo(f"Saved results to {Path(save).resolve()}")


@app.command()
def compare(results_v1: str = typer.Argument(...), results_v2: str = typer.Argument(...)):
    """Diff two saved evaluation runs."""
    from litmus.models import EvalResults

    r1 = EvalResults.load(results_v1)
    r2 = EvalResults.load(results_v2)
    diff = r2.compare(r1)
    diff.summary()


@app.command()
def report(
    results_path: str = typer.Argument(...),
    html: Optional[str] = typer.Option(None, "--html"),
    csv: Optional[str] = typer.Option(None, "--csv"),
):
    """Render an HTML/CSV report from saved results."""
    from litmus.models import EvalResults

    results = EvalResults.load(results_path)
    if html:
        results.to_html(html)
        typer.echo(f"Wrote {Path(html).resolve()}")
    if csv:
        df = results.to_dataframe()
        df.to_csv(csv, index=False)
        typer.echo(f"Wrote {Path(csv).resolve()}")
    if not html and not csv:
        results.summary()


@app.command("review-export")
def review_export(eval_set_path: str = typer.Argument(...), csv: str = typer.Option(..., "--csv")):
    """Export an eval set for human review."""
    eval_set = litmus.load(eval_set_path)
    eval_set.to_review_csv(csv)
    typer.echo(f"Wrote {Path(csv).resolve()}")


@app.command("review-import")
def review_import(eval_set_path: str = typer.Argument(...), csv: str = typer.Option(..., "--csv")):
    """Import human review corrections back into an eval set."""
    eval_set = litmus.load(eval_set_path)
    eval_set.apply_review(csv)
    eval_set.save(eval_set_path)
    typer.echo(f"Applied review from {Path(csv).resolve()}, saved to {Path(eval_set_path).resolve()}")


if __name__ == "__main__":
    app()
