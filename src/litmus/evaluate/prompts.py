"""Judge prompt templates for binary decomposition scoring and intent coverage.

The 9-question MVP judge (minimal/medium tiers) and 24-question exhaustive
judge share one template shape (Appendix A.6 of the synth doc): dynamic
question injection so ``BinaryQuestion`` lists of any length/dimension mix
work, including fully custom domain-specific question sets.
"""

from __future__ import annotations

from litmus.models import BinaryQuestion

BINARY_JUDGE_PROMPT = """You are evaluating a RAG system's answer. Read carefully and answer each of the \
following questions with exactly "yes" or "no", followed by a one-sentence explanation.

<question>
{question}
</question>

<retrieved_context>
{retrieved_context}
</retrieved_context>

<generated_answer>
{generated_answer}
</generated_answer>

<gold_answer>
{gold_answer}
</gold_answer>

<is_question_unanswerable>
{is_unanswerable}
</is_question_unanswerable>

Answer each question independently. Do not let your answer to one question influence another.

{questions_block}

Output as JSON, one key per question ID:
{{
{json_keys}
}}"""


def render_questions_block(questions: list[BinaryQuestion]) -> str:
    by_dimension: dict[str, list[BinaryQuestion]] = {}
    for q in questions:
        by_dimension.setdefault(q.dimension, []).append(q)
    lines = []
    for dimension, qs in by_dimension.items():
        lines.append(dimension.upper() + ":")
        for q in qs:
            lines.append(f"{q.id}. {q.text}")
        lines.append("")
    return "\n".join(lines).strip()


def render_json_keys(questions: list[BinaryQuestion]) -> str:
    return ",\n".join(f'  "{q.id}": {{"verdict": "yes|no", "reason": "..."}}' for q in questions)


def build_judge_prompt(
    question: str,
    retrieved_context: list[str],
    generated_answer: str,
    gold_answer: str,
    is_unanswerable: bool,
    questions: list[BinaryQuestion],
) -> str:
    context_block = "\n\n---\n\n".join(retrieved_context) if retrieved_context else "(no context retrieved)"
    return BINARY_JUDGE_PROMPT.format(
        question=question,
        retrieved_context=context_block,
        generated_answer=generated_answer or "(no answer given)",
        gold_answer=gold_answer,
        is_unanswerable=is_unanswerable,
        questions_block=render_questions_block(questions),
        json_keys=render_json_keys(questions),
    )


INTENT_COVERAGE_PROMPT = """Does the following generated answer clearly convey this specific point? \
Answer "yes" or "no" plus a one-sentence reason.

<point_to_check>
{intent_point_text}
</point_to_check>

<generated_answer>
{generated_answer}
</generated_answer>

Output as JSON:
{{"covered": true or false, "reason": "..."}}"""
