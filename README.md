# litmus

A Python library for evaluating RAG (Retrieval-Augmented Generation) systems with synthetically generated, realistically-noised eval sets and binary-decomposition scoring.

Based on the methodology in [`synth`](./synth) (a strategy document for building synthetic RAG eval sets — realistic noise injection, multi-document coverage, and diagnosable binary-decomposition scoring). Implements the full [`litmus` spec](./litmus).

## Why this exists

Most RAG evals are either skipped, or built from clean, grammatically-perfect questions that use the exact vocabulary of the source documents. That flatters the system under test and hides the failure modes that actually hurt users in production — typos, vague phrasing, wrong assumptions, cross-document synthesis, contradictory sources. `litmus` generates an eval set that deliberately includes these, then scores answers with binary yes/no questions (not a single opaque 1–5 score) so failures are traceable to a specific cause.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs `litmus` in editable mode plus its dependencies (`litellm`, `sentence-transformers`, `pymupdf`, `python-docx`, `beautifulsoup4`, `pydantic`, `rich`, `typer`, etc). For pandas/HTML report export and running the test suite:

```bash
pip install -e ".[dev]"
```

### LLM credentials

`litmus` calls LLMs through [litellm](https://github.com/BerriAI/litellm), so any litellm-supported model string works. This repo's default is the `azure/gpt-5.4` deployment on an Azure OpenAI resource, configured via the repo-root `conf` file (env-style: `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`). `litmus.llm.client.configure_azure_from_conf()` loads this automatically the first time an `azure/...` model string is used — you don't need to export anything yourself.

To use a different provider, pass any litellm model string (e.g. `"gpt-4o"`, `"claude-3-5-sonnet-20241022"`) and set that provider's usual environment variables.

### Embeddings

Chunk embeddings (used for contradiction-candidate pre-filtering) run locally via `sentence-transformers` (`all-MiniLM-L6-v2`), not through the Azure resource — no embeddings deployment is confirmed live there (see `CLAUDE.md` in this repo). The model downloads from Hugging Face on first use.

> **Note on this machine specifically:** Python's default SSL context fails to verify `huggingface.co` here even though the system trust store is fine (a certifi-bundle quirk). `litmus/ingest/embedder.py` works around this with `truststore.inject_into_ssl()` before any HF download. If you hit `CERTIFICATE_VERIFY_FAILED` errors elsewhere, the same fix applies.

## Quickstart

```python
import litmus

# Step 1: generate an eval set from your docs
eval_set = litmus.generate(
    docs_dir="./my_docs",
    tier="medium",       # "minimal" | "medium" | "exhaustive"
    save_path="eval_set.json",
)

# Step 2: evaluate your RAG system against it
# NOTE: this is a placeholder. Replace the body with your actual retrieval
# + generation logic - litmus can't evaluate a RAG system that doesn't exist.
# If you run this literally as-is, litmus detects the placeholder output and
# prints a warning instead of silently reporting meaningless scores.
def my_rag(question: str) -> dict:
    # ... your retrieval + generation logic ...
    return {"answer": "...", "contexts": ["chunk text 1", "chunk text 2"]}

results = litmus.evaluate(eval_set, rag=my_rag, save_path="results.json")
results.summary()
```

`results.summary()` prints a rich terminal table: overall score, per-dimension scores (faithfulness/correctness/abstention/...), set recall, and a breakdown by question type.

If `rag` returns placeholder-looking text (like the stub above) or the exact same answer for every question, `evaluate()` prints a warning (and records it in `results.metadata["stub_rag_warnings"]`) rather than silently handing back scores that don't mean anything.

### Where things get written

`save_path` is optional on both `generate()` and `evaluate()` — if you omit it, the `EvalSet`/`EvalResults` object is only returned in memory and nothing touches disk. Whenever a path *is* given (or a `cache_dir` is used, which is on by default), `litmus` logs the resolved **absolute** path it wrote to — not whatever relative path you passed in — so there's never ambiguity about where a file landed:

```
[litmus.generate] Cache/checkpoint directory: /Users/you/project/.litmus_cache
[litmus.generate] Saved eval set to /Users/you/project/eval_set.json
[litmus.evaluate] Saved results to /Users/you/project/results.json
```

If you pass `save_path=None` and forget to call `.save(...)`/`.to_json(...)` yourself, you'll see an explicit `"No save_path given - ... was NOT written to disk"` line instead of silence.


### Try it against the sample corpus

This repo ships a real 13-document corpus of FDA drug label PDFs at `sample1/`. Estimate cost first, then generate a small eval set:

```bash
litmus estimate sample1 --tier minimal
litmus generate sample1 --tier minimal --size 20 --save eval_set.json --max-workers 4
```

## The three tiers

`tier` controls not just question count but the entire depth of generation, noise, scoring, and diagnostics — each is a self-contained strategy, not a subset of the next.

| | `minimal` | `medium` | `exhaustive` |
|---|---|---|---|
| Question count | 40–80 | 80–200 | 200–600 |
| Question types | 4 (single-chunk, cross-doc, unanswerable, adversarial) | 7 (+ contradiction, comparative, compound) | 9 (+ procedural, wrong-assumption, ambiguous) |
| Noise layers | 2 | 6 | 9 |
| Binary scoring | 9 questions / 3 dimensions | 9 questions / 3 dimensions | 24 questions / 5 dimensions |
| Retrieval metrics | set recall@K | + chunk recall, MRR | + precision@K, gold chunk ranks |
| Quality checks | leakage filter | + eval-set self-validation | + coverage/diversity, contradiction detection, staleness |
| Alt. chunk discovery | no | yes | yes + second-pass validation |

Use `minimal` for a first eval or a small corpus (runs in minutes). Use `medium` for ongoing regression testing. Use `exhaustive` when you need defensible numbers or are optimizing a mature system.

## Core concepts

### EvalRecord

Every generated question is an `EvalRecord`: the (possibly noised) question, the original clean question, its type, applied noise profile, a gold answer, one or more alternative valid **gold chunk sets** (so a correct-but-different retrieval path isn't penalized), and an **intent decomposition** — atomic facts the answer must convey, each marked required or preferred.

### RAG callable interface

Your RAG function can return any of three shapes:

```python
def my_rag(question: str) -> dict:
    return {"answer": "...", "contexts": ["...", "..."]}

def my_rag(question: str) -> tuple[str, list[str]]:
    return ("...", ["...", "..."])

def my_rag(question: str) -> litmus.RAGResponse:
    return litmus.RAGResponse(answer="...", contexts=["..."])
```

### Binary decomposition scoring

Instead of asking an LLM judge for one holistic 1–5 score, `litmus` asks a series of yes/no questions per dimension (faithfulness, correctness, abstention, and — at the exhaustive tier — completeness and conciseness). Each dimension's score is the fraction of "yes" verdicts; the overall score is a weighted average. Every failure traces to a specific binary question and a one-sentence reason, so "why did this fail?" is always answerable.

```python
result = results.records[0]
print(result.faithfulness.score)                  # 0.67
print(result.faithfulness.verdicts["F2"].reason)   # "The answer states a 500MB limit not present in context."
```

### Custom scoring

You can decouple *generation* depth from *scoring* depth — e.g. generate with `tier="medium"` but score with a domain-specific rubric:

```python
from litmus import BinaryQuestion

medical_scoring = [
    BinaryQuestion(id="MF1", dimension="medical_faithfulness",
                   text="Are all drug names and dosages consistent with the retrieved sources?"),
    BinaryQuestion(id="MS1", dimension="safety",
                   text="Does the answer include safety warnings present in the source material?"),
]

results = litmus.evaluate(
    eval_set, rag=my_rag,
    scoring=medical_scoring,
    weights={"medical_faithfulness": 0.6, "safety": 0.4},
)
```

## Diagnostics and reports

```python
results.failure_patterns()      # which question types/binary questions fail most
results.retrieval_summary()     # aggregate retrieval metrics
results.scoring_health()        # exhaustive tier: yes-rate spread, phi-coefficient redundancy, uncovered failure modes
results.by_question_type()      # AggregateScore per QuestionType
results.by_noise_profile()      # AggregateScore per noise combination
results.to_html("report.html")
results.to_dataframe()          # requires pandas

eval_set.validate()              # eval-set self-validation (medium+): are the gold answers/chunks actually good?
eval_set.check_staleness(docs_dir="./my_docs")   # flag records whose source docs changed since generation
```

### Comparing two runs

```python
results_v2 = litmus.evaluate(eval_set, rag=new_rag_version)
diff = results_v2.compare(results_v1)
diff.summary()   # overall delta, per-dimension delta, which records flipped pass<->fail
```

### Human review workflow

```python
eval_set.to_review_csv("for_review.csv")
# ... a human fills in reviewer_verdict (approve/reject/edit) and reviewer_notes ...
eval_set.apply_review("for_review.csv")
eval_set.save("eval_set.json")
```

## CLI

```bash
litmus estimate ./docs --tier medium
litmus generate ./docs --tier medium --save eval_set.json --max-workers 4
litmus validate eval_set.json
litmus staleness eval_set.json --docs ./docs
litmus evaluate eval_set.json --rag my_module:rag_function --save results.json
litmus compare results_v1.json results_v2.json
litmus report results.json --html report.html --csv results.csv
litmus review-export eval_set.json --csv for_review.csv
litmus review-import eval_set.json --csv reviewed.csv
```

## Caching and resumability

LLM responses are cached by `sha256(prompt + model + temperature)` in `cache_dir/llm_cache.json` (default `.litmus_cache/`) — re-running an interrupted `generate()`/`evaluate()` call resumes from the last checkpoint rather than re-paying for identical calls. Clear it with `litmus.clear_cache()`.

## Concurrency

`max_workers=1` (default) runs everything sequentially — easiest to debug, deterministic ordering. Set `max_workers=5` or higher to parallelize LLM calls via a thread pool. See `CLAUDE.md` for why this is a thread pool rather than `asyncio` throughout.

## Development

```bash
pip install -e ".[dev]"
pytest                          # unit + fuzz tests (fast, no LLM calls)
pytest -m live                  # live tests against the real gpt-5.4 deployment (see CLAUDE.md — no mocking)
pytest --hypothesis-seed=0      # reproduce a specific fuzz failure
```

Project layout, testing philosophy, and design decisions are documented in [`CLAUDE.md`](./CLAUDE.md).
