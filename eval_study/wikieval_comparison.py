"""WikiEval comparison: litmus's judge vs. Ragas's own metrics, replicated
against the RAGAS paper's own benchmark (Es et al., EACL 2024, Table 4).

WikiEval (50 examples) provides pre-built contrastive pairs:
- faithfulness: answer (grounded) vs. ungrounded_answer, both against context_v1
- answer relevance: answer (good) vs. poor_answer, given the question
- context relevance: context_v1 (focused) vs. context_v2 (context_v1 + irrelevant backlink text)

For each dimension we compute pairwise accuracy: does the judge score the
"good" side higher than the "bad" side? This is directly comparable to the
paper's Table 4 (faithfulness 0.95, answer relevance 0.78, context relevance
0.70 for RAGAS itself).

litmus has no direct answer-relevance or context-relevance metric, so:
- Faithfulness: litmus's F1-F3 (fraction of yes) vs. Ragas's Faithfulness metric.
- Answer relevance: litmus's C1 ("addresses the question that was asked") as
  the closest proxy vs. Ragas's real AnswerRelevancy metric.
- Context relevance: litmus has no equivalent at all (no per-record binary
  question targets retrieval focus) -- Ragas's ContextRelevance metric only,
  no litmus comparison for this dimension.
"""

from __future__ import annotations

import asyncio
import json
import os

import pandas as pd
import truststore

truststore.inject_into_ssl()

from openai import AsyncAzureOpenAI

from eval_study import config

WIKIEVAL_PATH = config.SCRATCH_DIR / "wikieval.parquet"
OUT_DIR = config.RESULTS_DIR / "wikieval_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_ragas_llm():
    from ragas.llms.base import llm_factory

    client = AsyncAzureOpenAI(
        api_key=os.environ["AZURE_API_KEY"],
        azure_endpoint=os.environ["AZURE_API_BASE"],
        api_version=os.environ["AZURE_API_VERSION"],
    )
    llm = llm_factory("gpt-5.4", provider="openai", client=client)
    llm.model_args.pop("max_tokens", None)
    llm.model_args.pop("top_p", None)
    llm.model_args["max_completion_tokens"] = 3000
    llm.model_args["temperature"] = 1.0
    return llm


def build_local_embeddings():
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.run_config import RunConfig
    from sentence_transformers import SentenceTransformer

    class LocalEmbeddings(BaseRagasEmbedding):
        def __init__(self, model_name="all-MiniLM-L6-v2"):
            self.model = SentenceTransformer(model_name)

        def embed_text(self, text, **kwargs):
            return self.model.encode([text], normalize_embeddings=True).tolist()[0]

        async def aembed_text(self, text, **kwargs):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.embed_text, text)

    return LocalEmbeddings()


def load_wikieval() -> pd.DataFrame:
    return pd.read_parquet(WIKIEVAL_PATH)


async def run_ragas_faithfulness(llm, rows: list[dict]) -> list[dict]:
    from ragas.metrics.collections import Faithfulness

    metric = Faithfulness(llm=llm)
    results = []
    for i, row in enumerate(rows):
        context = list(row["context_v1"])
        good = await metric.ascore(user_input=row["question"], response=row["answer"], retrieved_contexts=context)
        bad = await metric.ascore(
            user_input=row["question"], response=row["ungrounded_answer"], retrieved_contexts=context
        )
        results.append({"source": row["source"], "good_score": good.value, "bad_score": bad.value})
        print(f"  ragas faithfulness {i + 1}/{len(rows)}: good={good.value:.2f} bad={bad.value:.2f}")
    return results


async def run_ragas_answer_relevancy(llm, embeddings, rows: list[dict]) -> list[dict]:
    from ragas.metrics.collections import AnswerRelevancy

    metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
    results = []
    for i, row in enumerate(rows):
        good = await metric.ascore(user_input=row["question"], response=row["answer"])
        bad = await metric.ascore(user_input=row["question"], response=row["poor_answer"])
        results.append({"source": row["source"], "good_score": good.value, "bad_score": bad.value})
        print(f"  ragas answer_relevancy {i + 1}/{len(rows)}: good={good.value:.2f} bad={bad.value:.2f}")
    return results


async def run_ragas_context_relevance(llm, rows: list[dict]) -> list[dict]:
    from ragas.metrics.collections import ContextRelevance

    metric = ContextRelevance(llm=llm)
    results = []
    for i, row in enumerate(rows):
        good = await metric.ascore(user_input=row["question"], retrieved_contexts=list(row["context_v1"]))
        bad = await metric.ascore(user_input=row["question"], retrieved_contexts=list(row["context_v2"]))
        results.append({"source": row["source"], "good_score": good.value, "bad_score": bad.value})
        print(f"  ragas context_relevance {i + 1}/{len(rows)}: good={good.value:.2f} bad={bad.value:.2f}")
    return results


def run_litmus_faithfulness(rows: list[dict]) -> list[dict]:
    from litmus.evaluate.answer_scorer import MVP_QUESTIONS, run_judge
    from litmus.llm.client import LLMClient

    client = LLMClient(model=config.FIXED_LLM)
    faithfulness_qs = [q for q in MVP_QUESTIONS if q.dimension == "faithfulness"]

    results = []
    for i, row in enumerate(rows):
        context = list(row["context_v1"])
        good_verdicts = run_judge(row["question"], context, row["answer"], "", False, faithfulness_qs, client)
        bad_verdicts = run_judge(
            row["question"], context, row["ungrounded_answer"], "", False, faithfulness_qs, client
        )
        good_score = sum(v.verdict for v in good_verdicts.values()) / len(good_verdicts)
        bad_score = sum(v.verdict for v in bad_verdicts.values()) / len(bad_verdicts)
        results.append({"source": row["source"], "good_score": good_score, "bad_score": bad_score})
        print(f"  litmus faithfulness {i + 1}/{len(rows)}: good={good_score:.2f} bad={bad_score:.2f}")
    return results


def run_litmus_answer_relevance_proxy(rows: list[dict]) -> list[dict]:
    """litmus has no answer-relevance metric; C1 ("does the answer address
    the question that was asked") is the closest existing binary question."""
    from litmus.evaluate.answer_scorer import MVP_QUESTIONS, run_judge
    from litmus.llm.client import LLMClient

    client = LLMClient(model=config.FIXED_LLM)
    c1 = [q for q in MVP_QUESTIONS if q.id == "C1"]

    results = []
    for i, row in enumerate(rows):
        context = list(row["context_v1"])
        good_verdicts = run_judge(row["question"], context, row["answer"], "", False, c1, client)
        bad_verdicts = run_judge(row["question"], context, row["poor_answer"], "", False, c1, client)
        good_score = 1.0 if good_verdicts["C1"].verdict else 0.0
        bad_score = 1.0 if bad_verdicts["C1"].verdict else 0.0
        results.append({"source": row["source"], "good_score": good_score, "bad_score": bad_score})
        print(f"  litmus C1-proxy {i + 1}/{len(rows)}: good={good_score:.2f} bad={bad_score:.2f}")
    return results


def pairwise_accuracy(results: list[dict]) -> dict:
    wins = sum(1 for r in results if r["good_score"] > r["bad_score"])
    ties = sum(1 for r in results if r["good_score"] == r["bad_score"])
    losses = sum(1 for r in results if r["good_score"] < r["bad_score"])
    n = len(results)
    # Ties broken randomly per the paper's methodology -> expected 0.5 credit
    accuracy = (wins + 0.5 * ties) / n if n else float("nan")
    return {"n": n, "wins": wins, "ties": ties, "losses": losses, "accuracy": accuracy}


def main() -> None:
    df = load_wikieval()
    rows = df.to_dict(orient="records")
    print(f"Loaded {len(rows)} WikiEval examples")

    llm = build_ragas_llm()
    embeddings = build_local_embeddings()

    print("\n=== Faithfulness ===")
    litmus_faith = run_litmus_faithfulness(rows)
    ragas_faith = asyncio.run(run_ragas_faithfulness(llm, rows))

    print("\n=== Answer relevance ===")
    litmus_ar = run_litmus_answer_relevance_proxy(rows)
    ragas_ar = asyncio.run(run_ragas_answer_relevancy(llm, embeddings, rows))

    print("\n=== Context relevance (Ragas only, no litmus equivalent) ===")
    ragas_cr = asyncio.run(run_ragas_context_relevance(llm, rows))

    summary = {
        "faithfulness": {
            "litmus": pairwise_accuracy(litmus_faith),
            "ragas": pairwise_accuracy(ragas_faith),
            "paper_reported_ragas": 0.95,
        },
        "answer_relevance": {
            "litmus_C1_proxy": pairwise_accuracy(litmus_ar),
            "ragas": pairwise_accuracy(ragas_ar),
            "paper_reported_ragas": 0.78,
        },
        "context_relevance": {
            "ragas": pairwise_accuracy(ragas_cr),
            "paper_reported_ragas": 0.70,
            "note": "no litmus equivalent metric exists",
        },
    }

    with open(OUT_DIR / "raw_results.json", "w") as f:
        json.dump(
            {
                "litmus_faithfulness": litmus_faith,
                "ragas_faithfulness": ragas_faith,
                "litmus_answer_relevance_proxy": litmus_ar,
                "ragas_answer_relevancy": ragas_ar,
                "ragas_context_relevance": ragas_cr,
            },
            f,
            indent=2,
        )
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
