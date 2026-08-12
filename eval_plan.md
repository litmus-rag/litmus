# Litmus Evaluation Study — Execution Plan for Claude Code

## A Minimal Systems/Benchmark Paper Validating the Litmus Framework

---

## 0. Framing Recap (read this before running anything)

**Paper type:** Systems/benchmark paper. Not a novel-algorithm paper. The claim is:
*"We built and operationalized an end-to-end synthetic RAG eval framework combining
techniques from HopWeaver, BINEVAL, ARES, CARE, and SeedRG, and we show empirically
that it works — its judge agrees with independent human annotation, and it surfaces
more/harder failure modes than the existing standard tool (Ragas) on the same corpora."*

**Target venue:** EMNLP/NAACL System Demonstrations track, or a relevant workshop
(e.g. SynIRgy at ECIR). Not a main research track — this scope does not clear that bar,
and should not try to.

**What this study produces (3 results, no more):**
1. Judge calibration — litmus's MVP judge (F1/F2/F3 faithfulness questions) vs.
   RAGTruth's independent human hallucination annotations. Zero labeling required.
2. Generation comparison — litmus vs. Ragas TestsetGenerator, same corpora,
   compared on litmus's own diagnostics (yes-rate spread, question-type coverage,
   leakage rate).
3. Leakage validation — litmus's leakage filter run against a known-leaky public
   corpus, showing it correctly flags a meaningful fraction.

**What this study explicitly does NOT include** (cut for minimal scope — list these
in the paper's Limitations section, don't pretend they don't matter):
- No ablation study (noise layers, bridge-entity vs. naive cross-doc, etc.)
- No self-labeled human calibration set (RAGTruth substitutes for this on faithfulness;
  correctness/abstention dimensions are NOT calibrated against human judgment in this
  study — say so explicitly)
- No multiple RAG systems under test — one simple RAG pipeline only
- Only the MVP (9-question) judge tier is evaluated, not the 24-question exhaustive tier

---

## 0.5 Environment & Setup

- **Install both stacks side by side:** `litmus` (your package, editable install)
  plus `ragas` (`pip install ragas`) in the same environment so both generators can
  run against identical document loaders where possible.
- **Model consistency matters more than model quality here.** Use the *same*
  underlying LLM for: (a) litmus's generation, (b) Ragas's generation, (c) the
  litmus judge scoring both eval sets. If the models differ across these, any
  difference you observe between litmus and Ragas is confounded with "different
  LLM did the generating," which a reviewer will flag immediately. Pick one model,
  document it once, reuse it everywhere in this study.
- **Run `litmus.api.estimate_cost()` before Step 6/7.** You already have this
  built — use it on the EnterpriseRAG-Bench subset and the second corpus before
  committing to a generation run, so you know the API budget going in rather than
  discovering it mid-run.
- **Seed everything, including Ragas.** Litmus's `seed` param is already wired
  through the pipeline. Ragas's `TestsetGenerator` also accepts a seed/random_state
  argument (check the installed version's signature) — set it and log it alongside
  litmus's seed in whatever run-metadata file you're keeping, so both generation
  runs are reproducible.
- **Output layout.** Keep everything under one `results/` directory so Step 13
  (assembling tables) doesn't require hunting across scratch files:
  ```
  results/
    ragtruth_calibration/        # Study 1: mapped rows, judge verdicts, table1.json
    corpus_a_enterpriserag/      # Study 2: litmus eval set, ragas eval set, RAG outputs, scores
    corpus_b_<name>/             # Study 2: same, for the second corpus
    tables/                      # final table1-4 outputs, ready to paste into the paper
  ```

---

## 1. Corpora

### 1.1 Primary corpus: EnterpriseRAG-Bench
- Source: `github.com/onyx-dot-app/EnterpriseRAG-Bench` (Sun et al., 2026, arXiv:2605.05253)
- Why: its question taxonomy (single-doc lookup, multi-doc reasoning, constrained
  retrieval, conflict resolution, absent information) maps closely onto litmus's own
  `QuestionType` enum, giving you a legitimate "independently designed taxonomy
  validates our coverage model" claim in the paper.
- Use: pull a manageable subset of the underlying document corpus (not the full
  500K-document synthetic set — pick a few thousand documents across 2-3 of their
  nine source types, e.g. Confluence + GitHub + Slack, to keep generation cost sane).

### 1.2 Second corpus: MuSiQue — full comparison, doing double duty
Per your ask: two corpora, both get the full litmus-vs-Ragas comparison (§3). Using
MuSiQue as corpus B rather than picking something unrelated lets it serve two
purposes at once instead of needing a third corpus:

- **Full comparison role (§3):** same treatment as EnterpriseRAG-Bench — litmus
  generation, Ragas generation, both scored by the same RAG system and judge,
  same Tables 2/3 produced for this corpus too.
- **Leakage-validation role (§4):** MuSiQue is *also* the corpus SeedRG reports
  documented leakage rates on (your own doc cites 2WikiMultihopQA 62%, HotpotQA
  52%, QASC 75% — MuSiQue itself is comparatively cleaner but still a known
  multi-hop leakage testbed), which makes it a natural fit for the leakage-filter
  check without spinning up a third corpus just for that purpose.

Pull the documents/passages MuSiQue questions are grounded in and use those as the
document set for litmus's and Ragas's generation runs on this corpus.

### 1.3 RAG system under test
Keep this deliberately simple — you are not benchmarking RAG quality, you are
demonstrating the eval framework works. A naive retrieve-then-generate pipeline
(e.g. an embedding index over chunks + top-k retrieval + a single LLM call) is
sufficient and defensible for a systems/demo paper. Document its exact architecture
in a short paper subsection so it's reproducible — that's all reviewers need.

---

## 2. Study 1 — Judge Calibration via RAGTruth (no labeling required)

### 2.1 Data
- Repo: `github.com/ParticleMedia/RAGTruth` (MIT license, no gating)
- Files needed: `source_info.jsonl` and `response.jsonl`, filtered to `task_type == "QA"`
- Field mapping into litmus's judge input:

| RAGTruth field | Litmus judge input |
|---|---|
| `source_info.question` | `question` |
| `source_info.passages` | `retrieved_context` (split on the `passage N:` markers) |
| `response.response` | `generated_answer` |
| `gold_answer` | **not available** — pass empty string; F1/F2/F3 don't require it |
| `is_unanswerable` | `False` for all QA rows (RAGTruth QA is not an abstention benchmark) |
| derived: `len(response.labels) > 0` | ground truth `has_hallucination` label |

### 2.2 Sampling
- Full QA subset: 989 source instances → 5,934 responses (1,724 hallucinated, ~29%).
- Take a stratified random sample of ~250-300 responses, preserving roughly the
  29% hallucination rate, to control judge-API cost. Set and record a fixed seed.

### 2.3 What to run
- Run **only** the faithfulness questions from litmus's MVP judge (F1, F2, F3) —
  these are the ones semantically aligned with "does this response contain
  unsupported/contradictory claims relative to the passages," which is exactly what
  RAGTruth's span annotations measure. Do NOT attempt to calibrate C1-C3 or A1-A3
  against RAGTruth — there's no ground truth for those in this dataset, and forcing
  it would be a methodological error worth a reviewer flagging.
- Derive a per-response judge verdict: `judge_says_hallucinated = NOT(F1 AND F2 AND F3)`
  (i.e., any "no" on a faithfulness question counts as the judge detecting a problem).
- Compare `judge_says_hallucinated` against RAGTruth's `has_hallucination` ground truth.

### 2.4 Metrics to report
- **Cohen's kappa**, not just raw agreement — this is the metric your own §6.5
  prescribes (0.8 kappa as the working calibration threshold), and using it here
  keeps the paper consistent with the methodology it's built on. Raw agreement
  rate alone is inflated by class imbalance (RAGTruth's QA hallucination rate is
  ~29%, so a judge that always guesses "no hallucination" scores ~71% raw
  agreement while being useless — kappa corrects for this).
- Precision / recall / F1 of the judge treating "hallucinated" as positive class
  (report alongside kappa, not instead of it).
- Per-question (F1/F2/F3) individual kappa, not just the combined verdict — this
  is the level of granularity your own §6.5 spec calls for and it's the more
  interesting number for the paper (which sub-question is doing the work?
  which one is closest to chance?).
- Break down by `model` (GPT-4 vs. Llama-2-7B, etc.) if you want an extra table —
  optional, cut if time-constrained.

### 2.5 Honest caveat to state in the paper
RAGTruth's responses were generated by 2023-era models (GPT-3.5/4-0613, Llama-2,
Mistral-7B). This is fine for judge calibration (you're testing whether the judge
detects hallucination, not benchmarking model quality) but should be stated as a
limitation — the judge's agreement rate on newer, better-calibrated models' outputs
is untested here.

---

## 3. Study 2 — Litmus vs. Ragas, run on BOTH corpora

Run everything in this section twice — once on EnterpriseRAG-Bench (§1.1), once on
MuSiQue (§1.2) — with identical procedure and identical model choice (§0.5) each
time, so the two runs are directly comparable to each other as well as each
individually comparable to Ragas.

### 3.1 Generation runs (repeat per corpus)
- Litmus: `litmus.generate(docs_dir=..., tier="minimal", llm=<fixed model>, seed=<fixed, logged>)`
  — use the `minimal` tier (4 question types, single vocab-mismatch noise layer) to
  match Ragas's comparable scope; don't compare litmus's `exhaustive` tier against
  Ragas's default output, that's an unfair comparison and a reviewer will catch it.
- Ragas: run `TestsetGenerator` on the identical document set, same LLM, same
  target question count, seed set and logged per §0.5.

### 3.2 What to compare
Everything below is already computable from code you have — this is compute time,
not new engineering:
1. **Question-type coverage.** Does Ragas's output naturally cover unanswerable,
   contradiction, and adversarial-distractor categories? (Expected: no, or only
   incidentally.) Litmus's tier config guarantees proportions — report the actual
   generated distribution for both.
2. **Yes-rate spread / scoring health**, computed via
   `litmus.evaluate.diagnostics.compute_scoring_health` — run your RAG system
   against BOTH eval sets (litmus's and Ragas's, using litmus's own MVP judge for
   scoring both, so the judge is held constant and only the *eval set* varies) and
   compare `ScoringHealthReport.verdict` and yes-rate spread between the two.
   This is your strongest, most novel comparison point — nobody else reports this
   diagnostic, so there's no existing baseline number to beat, just your own
   framework's story of "here's what a naive eval set looks like on our own
   diagnostics vs. our tiered generation."
3. **Cross-doc question genuineness** (qualitative). Pull 10-15 cross-doc examples
   from each tool's output and manually check: is the question actually answerable
   from a single retrieved chunk despite being labeled multi-hop? (This is the
   "pseudo multi-hop" check from HopWeaver/your own bridge.py rationale.) Report
   as a small qualitative table with examples, not a large-N study — this is
   illustrative, not a statistical claim.

### 3.3 What NOT to do here
Don't try to get human judges to rate "which question set is better" — that's a
much bigger study than this scope allows. Stick to the structural/diagnostic
comparisons above; they're defensible and don't require additional labeling.

---

## 4. Study 3 — Leakage Filter Validation (reuses corpus B's generation run)

- No new document loading needed — reuse the litmus `EvalSet` already generated
  on MuSiQue in Study 2 (§3.1). Its `minimal`-tier generation already includes
  clean single-chunk and cross-doc questions over these passages.
- Run `litmus.generate.leakage.filter_leaked_records` over that same generated
  set and report the discard rate as a standalone result (Table 4), separate from
  the Study 2 comparison tables.
- Frame the result carefully: this measures "does our filter catch questions our
  own generator produced that are answerable from parametric memory," which is a
  test of the filter's sensitivity — not a reproduction of SeedRG's leakage-rate
  numbers on MuSiQue's original human-written questions. Say this explicitly to
  avoid a reviewer conflating the two.

---

## 5. Tables and Figures for the Paper (plan these now, not after)

1. **Table 1 — Judge calibration.** RAGTruth kappa/precision/recall/F1, overall
   and per-binary-question (F1/F2/F3).
2. **Table 2 — Question-type/noise coverage comparison.** Litmus vs. Ragas,
   generated counts per category. One row-block per corpus (EnterpriseRAG-Bench,
   MuSiQue) so the comparison's consistency across corpora is visible.
3. **Table 3 — Scoring health comparison.** Yes-rate spread, mean phi, verdict
   ("healthy"/"needs_work"/"broken") for litmus-generated set vs. Ragas-generated
   set, same judge, same RAG system. Again, one block per corpus.
4. **Table 4 — Leakage filter validation.** Discard rate on MuSiQue-derived,
   litmus-generated questions (reusing Study 2's MuSiQue generation run).
5. **Figure/Table (optional, cut if short on time) — qualitative cross-doc examples.**
   3-5 side-by-side examples per corpus: a genuine bridge-entity question (litmus)
   vs. a pseudo-multi-hop question (Ragas), with a one-line note on why each
   is/isn't genuinely cross-doc.

---

## 6. Execution Checklist (ordered, hand this to Claude Code)

Work through these in order. Each step names the litmus module it exercises so
Claude Code can locate the relevant code without re-deriving the plan.

- [ ] **Step 0.** Set up environment per §0.5: install `litmus` + `ragas`, pick and
  fix the one LLM used across all generation/judging in this study, create the
  `results/` directory layout, confirm API key access.
- [ ] **Step 1.** Clone `ParticleMedia/RAGTruth`. Write a loader that joins
  `source_info.jsonl` + `response.jsonl` on `source_id`, filters to `task_type == "QA"`,
  and produces the field mapping in §2.1. Output: a flat JSONL of
  `{question, retrieved_context, generated_answer, has_hallucination, model}`.
- [ ] **Step 2.** Stratified-sample ~250-300 rows per §2.2 (fixed seed, log the seed).
- [ ] **Step 3.** Write a thin wrapper around `litmus.evaluate.answer_scorer.run_judge`
  that accepts the mapped RAGTruth rows directly (bypassing the normal `EvalRecord`/
  `RAGResponse` flow, since RAGTruth isn't a litmus-native eval set) and runs only the
  F1/F2/F3 questions from `MVP_QUESTIONS`. Save raw verdicts per row.
- [ ] **Step 4.** Compute kappa/precision/recall/F1 (overall and per-question)
  against `has_hallucination`. Save as Table 1 data.
- [ ] **Step 5.** Pull the EnterpriseRAG-Bench document subset (corpus A: 2-3
  source types, a few thousand docs) and the MuSiQue passage set (corpus B).
  Load both via `litmus.ingest.loader`. Run `litmus.api.estimate_cost()` on each
  before proceeding — confirm budget is acceptable.
- [ ] **Step 6.** For EACH corpus (A, B): run
  `litmus.api.generate(tier="minimal", llm=<fixed model>, seed=<fixed, logged>, ...)`.
  Save the resulting `EvalSet` per corpus.
- [ ] **Step 7.** For EACH corpus: run Ragas's `TestsetGenerator` on the identical
  document set, same target size, same fixed LLM, seed set and logged. Save its
  output separately per corpus (a thin adapter exposing `question`/`contexts`/
  `ground_truth` is enough — don't force it into litmus's `EvalRecord` schema).
- [ ] **Step 8.** Stand up the simple RAG pipeline (§1.3), once, reused across
  both corpora. Run it against all four eval sets (litmus-A, ragas-A, litmus-B,
  ragas-B).
- [ ] **Step 9.** Score all four sets of RAG outputs with `litmus.api.evaluate`
  (MVP judge, same fixed model as Step 0, held constant across all four). Save
  four `EvalResults`.
- [ ] **Step 10.** Run `EvalResults.scoring_health()` on all four. Save as Table 3
  data (one block per corpus). Compute question-type/noise-profile distributions
  from each `EvalSet` directly for Table 2 (one block per corpus).
- [ ] **Step 11.** Reuse corpus B's litmus `EvalSet` from Step 6 — run
  `litmus.generate.leakage.filter_leaked_records` over it. Save discard rate as
  Table 4 data. No new generation needed for this step.
- [ ] **Step 12.** Manually pull 10-15 cross-doc examples from each corpus's
  Step 6/7 outputs (litmus-A vs ragas-A, litmus-B vs ragas-B) for the qualitative
  table (this step is manual review, not code).
- [ ] **Step 13.** Assemble all tables. Draft the paper using the skeleton in §7 below.

---

## 7. Paper Skeleton

Target: EMNLP/NAACL System Demonstrations format (check current page limit on the
venue site before finalizing — these change year to year).

1. **Abstract.** One-paragraph version of the framing in §0.
2. **Introduction.** Motivate why synthetic RAG eval sets need realism (vocab
   mismatch, noise) and diagnosability (binary decomposition) — this is mostly
   already written in your original methodology doc's §1 and §4 intro.
3. **Related Work.** Your original doc's §11 is most of this section already —
   position litmus as an integration/operationalization of HopWeaver, BINEVAL,
   ARES, CARE, SeedRG, contrasted with Ragas/DeepEval/LlamaIndex as the
   existing-tool baseline.
4. **System Description.** Architecture overview: `generate/` (question types,
   noise layers, bridge-entity cross-doc, leakage filter, tiering) and `evaluate/`
   (binary-decomposition judge, retrieval scoring, scoring-health diagnostics).
   Use your `config.py` tier table directly as a system-description table.
5. **Experimental Setup.** Corpora (§1), RAG system under test (§1.3), Ragas
   comparison setup (§3.1).
6. **Results.** Written up in full in §7.6 below, from the actual scoped-down
   run (see `CLAUDE.md`'s "Eval study results" section for the underlying
   numbers, code pointers, and three code bugs found/fixed along the way).
   - 6.1 Judge calibration (Table 1)
   - 6.2 Generation comparison vs. Ragas (Tables 2-3, qualitative examples)
   - 6.3 Leakage filter validation (Table 4)
   - 6.4 Ablation: harder corpus/tier (Table 5)
   - 6.5 Judge validation against RAGAS's own benchmark (WikiEval)
   - 6.6 Scale sensitivity (2x corpus size)
7. **Limitations.** State every cut from §0 explicitly here — no ablation, no
   self-labeled correctness/abstention calibration, single RAG system, MVP tier
   only, RAGTruth's models are dated. This section is not optional; a demo/systems
   paper without an honest limitations section reads as unaware of its own scope.
8. **Conclusion.**

---

## 7.6 Results (written)

This is the actual Results section content for §7 item 6, from a scoped-down
run (sizes cut for iteration speed per the note at the top of §0; see
`eval_study/config.py` for the exact constants — corpus A: 50/100 docs,
corpus B: 20/40 MuSiQue instances, RAGTruth: 110 rows, all against a single
fixed `azure/gpt-5.4` deployment). All numbers below are traceable to
`results/tables/table{1,2,3,4,5}.json`, `results/wikieval_comparison/`, and
`CLAUDE.md`'s "Eval study results" section, which also documents three real
code bugs found and fixed while producing these numbers (bridge-entity
under-sampling, a JSON-array parsing gap under Azure's structured-output
mode, and a checkpoint-key collision across corpora) — call these out in the
System Description or as a footnote; they are legitimate engineering
findings from operationalizing the spec, not just cleanup.

### 6.1 Judge calibration (Table 1)

Litmus's MVP judge's faithfulness sub-questions (F1–F3) were run against 110
stratified rows from RAGTruth's QA subset (~29% hallucination rate
preserved by the sample), comparing `judge_says_hallucinated = NOT(F1 AND F2
AND F3)` against RAGTruth's human span-annotated ground truth.

| Metric | Value |
|---|---|
| Cohen's κ (combined verdict) | **0.48** |
| Precision (hallucination = positive class) | 0.53 |
| Recall | 0.94 |
| F1 | 0.67 |
| Raw agreement | 0.74 |
| κ — F1 | 0.48 |
| κ — F2 | 0.60 |
| κ — F3 | 0.52 |

κ=0.48 is moderate agreement, below this document's own §6.5 working
threshold of 0.8. The judge is high-recall/low-precision: it over-flags,
predicting hallucination on 52% of responses against a 29% ground-truth
rate. F2 (fabricated-fact detection) is the strongest individual
sub-question; F1 (general claim support) the weakest. State plainly: **this
is not yet evidence the judge is reliable enough to be a sole faithfulness
gate**, and RAGTruth's responses come from 2023-era models (GPT-3.5/4-0613,
Llama-2, Mistral-7B) per the caveat in §2.5 — the judge's behavior on
newer, better-calibrated models is untested by this result.

### 6.2 Generation comparison vs. Ragas (Tables 2–3)

Litmus (`minimal` tier) and Ragas's `TestsetGenerator` were run on identical
document sets for both corpora, same fixed model, seeds logged, then both
eval sets were scored by litmus's own MVP judge against one `SimpleRAG` test
pipeline held constant within each corpus.

**Corpus A — EnterpriseRAG-Bench (GitHub PR text), 50 docs:**

| | litmus | Ragas |
|---|---|---|
| n | 25 | 27 |
| Question-type coverage | single_chunk, cross_doc, unanswerable, adversarial (tier-guaranteed proportions) | 100% single/multi-hop (no unanswerable/adversarial/noise categories) |
| `scoring_health()` verdict | broken | needs_work |
| Yes-rate spread — faithfulness / correctness / abstention | 0.04 / 0.13 / 0.21 | 0.19 / 0.22 / 0.04 |

**Corpus B — MuSiQue passages, 20 instances:**

| | litmus | Ragas |
|---|---|---|
| n | 19 | 15 |
| `scoring_health()` verdict | broken | broken |
| Yes-rate spread — faithfulness / correctness / abstention | 0.0 / 0.0 / 0.53 | 0.0 / 0.06 / 0.07 |

**Findings:**
- Ragas's own diagnostic scoring beats litmus's on corpus A (better spread
  on 2 of 3 dimensions); both saturate ("broken") on corpus B, with the
  simple RAG pipeline performing too well on either eval set to stress the
  judge on faithfulness/correctness there.
- Litmus's structural differentiator is **guaranteed taxonomy coverage**
  (every generated set contains unanswerable, adversarial, and noise-injected
  questions by tier-config construction), not proven higher-quality
  individual questions — Ragas's output was 100% single/multi-hop with none
  of those categories, by construction of its own method.
- Qualitative review of Ragas's cross-doc/multi-hop output found a
  non-trivial fraction of degenerate items — keyword-soup non-questions,
  garbled multi-clause prose (examples in
  `results/qualitative_review_corpus_*.md`) — a genuine quality gap on
  Ragas's side, orthogonal to the coverage-vs-quality framing above.
- **Scale sensitivity (§6.6 below) shows this comparison is corpus-dependent,
  not a fixed ordering** — report both sizes rather than picking one.

### 6.3 Leakage filter validation (Table 4)

Litmus's leakage filter was run over its own corpus-B (MuSiQue) generation
output: of 30 answerable candidate questions, 24 were discarded as
answerable from parametric memory alone (an 80% discard rate), leaving 6
answerable + 12 exempt-unanswerable = 18 final records. State explicitly
that this measures the filter's *sensitivity on litmus's own generated
questions*, not a reproduction of SeedRG's leakage rates on MuSiQue's
original human-written questions (§4) — the two are not directly
comparable numbers.

### 6.4 Ablation: harder corpus/tier (Table 5, not in original plan — added
after Study 2's diagnostics saturated)

Run on `sample1/`'s 13 real FDA drug-label PDFs at `medium` tier (adds
contradiction/comparative/compound question types and heavier noise
layers), litmus-only, no Ragas comparison. `sample1` has genuine built-in
cross-entity structure (several drugs share active ingredients — e.g.
Ozempic/Wegovy/Rybelsus are all semaglutide), which is what makes this a
fair stress test rather than just "a different corpus."

| Metric | Value |
|---|---|
| n | 31 records (6 genuine cross_doc, from 25 bridge-entity candidates) |
| Document coverage | 85% (up from 69% pre-fix) |
| Correctness spread | **0.32** (above the 0.20 healthy threshold) |
| Abstention spread | **0.26** (above threshold) |
| Faithfulness spread | 0.06 (still saturated) |
| Overall verdict | broken (faithfulness alone triggers it) |

The best discrimination result in the whole study — two of three dimensions
genuinely healthy. Faithfulness stays saturated because the `SimpleRAG` test
pipeline only ever echoes retrieved context and rarely fabricates
regardless of question difficulty; harder questions here produce
wrong-but-faithful answers, not more hallucination. This is a property of
the test RAG under evaluation, not evidence that litmus's judge cannot
discriminate faithfulness given a RAG that actually hallucinates.

### 6.5 Judge validation against RAGAS's own benchmark (WikiEval)

To get a result directly comparable to RAGAS's own published validation
(Es et al., 2024, Table 4), litmus's judge and Ragas's shipped metrics were
both run on WikiEval (50 Wikipedia-derived examples with pre-built
contrastive good/bad pairs), same judge model for both, pairwise accuracy
as the metric (does the judge score the "good" side higher than the "bad"
side).

| Dimension | litmus | Ragas (this run) | Ragas (2024 paper) |
|---|---|---|---|
| Faithfulness | 0.98 | 0.98 | 0.95 |
| Answer relevance | 0.87 (C1 proxy) | 0.53 | 0.78 |
| Context relevance | n/a | 0.50 (pure ties) | 0.70 |

**This is the strongest single result in the study for litmus.** On
faithfulness, litmus performs equivalently to Ragas's own metric on Ragas's
own validation set, both above the paper's originally-reported number. On
answer relevance, litmus's closest proxy (C1: "does the answer address the
question asked") clearly outperforms Ragas's current embedding-cosine-based
`AnswerRelevancy` — traced to a real, confirmed weakness in that metric
(it cannot distinguish "topically restates the question" from "actually
answers it," so evasive-but-on-topic answers score falsely high). Ragas's
`ContextRelevance` scored a dead 0.50 tie on every single example — traced
to its rating rubric awarding max score for "contains any relevant
information," which cannot penalize irrelevant padding despite the metric's
name. Both root causes are documented in `CLAUDE.md` with code pointers;
worth a short paragraph in Related Work or Results as a finding about the
comparison baseline, not just about litmus.

### 6.6 Scale sensitivity (corpus size ablation, 2x)

Both Study 2 corpora were re-run at roughly double size (corpus A: 50→100
docs; corpus B: 20→40 MuSiQue instances) to check whether §6.2's
litmus-vs-Ragas ordering is stable.

| | Corpus A (1x → 2x) | Corpus B (1x → 2x) |
|---|---|---|
| litmus correctness spread | 0.13 → 0.23 | 0.0 → 0.21 |
| Ragas correctness spread | 0.22 → 0.17 | 0.06 → 0.05 |

At 2x, corpus A's gap narrows (litmus nearly catches up on correctness);
corpus B *flips* — litmus goes from tied-to-behind to clearly ahead of Ragas
on correctness discrimination (0.21 vs. 0.05), while both remain
faithfulness-saturated. **Report this as evidence the litmus-vs-Ragas
comparison is corpus-dependent, not a fixed ordering** — a single-size,
single-corpus comparison would have overstated confidence in whichever
direction it happened to point. This is a genuinely useful finding for a
systems paper: it's evidence the authors checked robustness rather than
reporting the first number they got, and it belongs in Results proper
(not buried in Limitations) since it's a real result, not just a caveat.

---



- **If EnterpriseRAG-Bench's full corpus is too large/expensive to run generation
  over:** subsample document types rather than randomly sampling documents —
  keeping 2-3 whole source types intact preserves the taxonomy-coverage story
  better than a scattered random sample.
- **If Ragas's TestsetGenerator errors or behaves unexpectedly on a document type**
  (this happens with e.g. very short or very structured documents): note it as a
  qualitative observation — "Ragas's generator required X" is itself a legitimate,
  citable finding for the comparison section, not just a bug to route around silently.
- **If judge agreement with RAGTruth comes back low** (near chance): don't bury
  this — report it honestly and discuss possible causes (prompt wording, RAGTruth's
  span-level labels being finer-grained than litmus's binary per-response
  question). A demo paper survives an honest negative-ish result; it does not
  survive a reviewer discovering a hidden one.
