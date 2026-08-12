"""Self-contained HTML report for EvalResults (medium+ tier)."""

from __future__ import annotations

from pathlib import Path

from litmus.models import EvalResults

_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>litmus eval report</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem; color: #1a1a1a; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 0.5rem; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.9rem; }}
  th {{ background: #f5f5f5; }}
  .bar-bg {{ background: #eee; border-radius: 3px; width: 200px; display: inline-block; height: 12px; vertical-align: middle; }}
  .bar-fg {{ background: #4a7; border-radius: 3px; height: 12px; }}
  .score {{ font-weight: 600; }}
  .low {{ color: #c33; }}
  .mid {{ color: #b80; }}
  .high {{ color: #2a2; }}
  .failed-row {{ background: #fff5f5; }}
</style>
</head>
<body>
<h1>litmus eval report - tier: {tier}</h1>
<p>{n_records} records evaluated.</p>

<h2>Summary</h2>
<table>
<tr><th>Metric</th><th>Score</th></tr>
{summary_rows}
</table>

<h2>By Question Type</h2>
<table>
<tr><th>Type</th><th>Count</th><th>Mean Overall</th></tr>
{by_type_rows}
</table>

<h2>By Noise Profile</h2>
<table>
<tr><th>Profile</th><th>Count</th><th>Mean Overall</th></tr>
{by_noise_rows}
</table>

<h2>Failed Records (overall &lt; 0.7)</h2>
<table>
<tr><th>ID</th><th>Type</th><th>Overall</th><th>Question</th></tr>
{failed_rows}
</table>
</body>
</html>"""


def _score_class(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "mid"
    return "low"


def _bar(score: float) -> str:
    width = max(0, min(100, round(score * 100)))
    return f'<span class="bar-bg"><span class="bar-fg" style="width:{width}%"></span></span> {score:.2f}'


def render_html(results: EvalResults) -> str:
    n = len(results.records)
    if n == 0:
        summary_rows = "<tr><td colspan=2>No records</td></tr>"
        by_type_rows = by_noise_rows = failed_rows = "<tr><td colspan=3>-</td></tr>"
    else:
        overall = sum(r.overall_score for r in results.records) / n
        faithfulness = sum(r.faithfulness.score for r in results.records) / n
        correctness = sum(r.correctness.score for r in results.records) / n
        abstention = sum(r.abstention.score for r in results.records) / n
        set_recall = sum(1.0 if r.set_recall else 0.0 for r in results.records) / n

        summary_rows = "\n".join(
            f'<tr><td>{name}</td><td class="score {_score_class(v)}">{_bar(v)}</td></tr>'
            for name, v in [
                ("Overall", overall),
                ("Faithfulness", faithfulness),
                ("Correctness", correctness),
                ("Abstention", abstention),
                ("Set Recall", set_recall),
            ]
        )

        by_type_rows = "\n".join(
            f'<tr><td>{qtype.value}</td><td>{agg.count}</td>'
            f'<td class="score {_score_class(agg.mean_overall)}">{_bar(agg.mean_overall)}</td></tr>'
            for qtype, agg in sorted(results.by_question_type().items(), key=lambda kv: kv[1].mean_overall)
        )

        by_noise_rows = "\n".join(
            f'<tr><td>{profile}</td><td>{agg.count}</td>'
            f'<td class="score {_score_class(agg.mean_overall)}">{_bar(agg.mean_overall)}</td></tr>'
            for profile, agg in sorted(results.by_noise_profile().items(), key=lambda kv: kv[1].mean_overall)
        )

        failed = results.failed_records()
        failed_rows = "\n".join(
            f'<tr class="failed-row"><td>{r.record_id}</td><td>{r.question_type.value}</td>'
            f'<td class="score low">{r.overall_score:.2f}</td><td>{r.question[:120]}</td></tr>'
            for r in failed
        ) or "<tr><td colspan=4>None</td></tr>"

    return _TEMPLATE.format(
        tier=results.tier,
        n_records=n,
        summary_rows=summary_rows,
        by_type_rows=by_type_rows,
        by_noise_rows=by_noise_rows,
        failed_rows=failed_rows,
    )


def write_html_report(results: EvalResults, path: str) -> None:
    Path(path).write_text(render_html(results))
