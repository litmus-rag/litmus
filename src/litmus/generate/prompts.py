"""Prompt templates for eval-question generation.

Templates are adapted from Appendix A of the synth methodology doc, extended
to cover all 9 question types (not just the 4-type MVP) and all 7 noise
layers. Every generation prompt asks for JSON output; callers pass the
result through ``litmus.llm.client.extract_json``.
"""

from __future__ import annotations

TERM_EXTRACTION_PROMPT = """List the 5-10 most distinctive technical or formal terms in this passage \
that a casual user would be unlikely to use in everyday speech.

<passage>
{chunk_text}
</passage>

Output as a JSON array of strings, nothing else."""


SINGLE_CHUNK_PROMPT = """You are generating an evaluation question for a retrieval system.

Read the following passage:

<passage>
{chunk_text}
</passage>

Write ONE question that:
- A real user would plausibly ask
- Can be fully answered from this passage alone
- Is specific enough to have a clear correct answer
- Does NOT quote the passage verbatim

Then write the gold answer, using only information from the passage.

Then decompose the gold answer into 2-4 atomic intent points - the individual facts or claims that must be \
conveyed for the answer to count as correct. Mark each as "required" or "preferred".

Output as JSON:
{{
  "question": "...",
  "gold_answer": "...",
  "intent_points": [
    {{"id": "P1", "text": "...", "required": true}},
    {{"id": "P2", "text": "...", "required": true}},
    {{"id": "P3", "text": "...", "required": false}}
  ]
}}"""


CROSS_DOC_PROMPT = """You are generating a cross-document evaluation question.

Below are two passages that both relate to "{bridge_entity}":

<passage_1 source="{doc_id_1}">
{chunk_text_1}
</passage_1>

<passage_2 source="{doc_id_2}">
{chunk_text_2}
</passage_2>

Write ONE question that:
- Requires information from BOTH passages to answer completely
- Cannot be answered correctly using only one passage
- A real user would plausibly ask
- Is about {bridge_entity} and how it relates across the two passages

Then write the gold answer, integrating information from both.

Then decompose the gold answer into atomic intent points, marking which passage(s) each comes from.

Output as JSON:
{{
  "question": "...",
  "gold_answer": "...",
  "intent_points": [
    {{"id": "P1", "text": "...", "required": true, "from": ["passage_1"]}},
    {{"id": "P2", "text": "...", "required": true, "from": ["passage_2"]}}
  ]
}}"""


UNANSWERABLE_PROMPT = """You are generating an "unanswerable" evaluation question - one that a user would \
plausibly ask but that the documentation below does NOT cover.

Here is a sample of what the documentation DOES cover:
{topic_sample}

Write ONE question that:
- Sounds like a reasonable user question in this domain
- Is adjacent to but NOT covered by the documentation
- Would tempt a naive system to hallucinate an answer
- Is not absurd or clearly out of scope

Output as JSON:
{{
  "question": "...",
  "gold_answer": "The documentation does not cover [specific topic]. [Optional: suggest what IS covered nearby.]",
  "intent_points": [
    {{"id": "P1", "text": "Explicit acknowledgment that this is not documented", "required": true}},
    {{"id": "P2", "text": "Optional pointer to related documented topics", "required": false}}
  ]
}}"""


ADVERSARIAL_PROMPT = """You are generating an adversarial/distractor evaluation question.

Below is the CORRECT passage that answers the question, plus one or more DISTRACTOR passages that are \
topically similar but would give a wrong answer if blended in:

<correct_passage source="{doc_id}">
{chunk_text}
</correct_passage>

<distractor_passages>
{distractor_text}
</distractor_passages>

Write ONE question that:
- Is answerable correctly ONLY from the correct passage
- Would tempt a system with weak retrieval/generation discrimination to blend in details from \
the distractor passages
- A real user would plausibly ask

Then write the gold answer using ONLY the correct passage. Explicitly note what the answer must NOT include \
(details unique to the distractors).

Output as JSON:
{{
  "question": "...",
  "gold_answer": "...",
  "intent_points": [
    {{"id": "P1", "text": "...", "required": true}}
  ],
  "must_not_include": "brief description of distractor-only details that would signal blending"
}}"""


CONTRADICTION_PROMPT = """You are generating a contradiction evaluation question.

Below are two passages from different documents that give conflicting answers to the same underlying question:

<passage_1 source="{doc_id_1}">
{chunk_text_1}
</passage_1>

<passage_2 source="{doc_id_2}">
{chunk_text_2}
</passage_2>

Write ONE question that both passages address, but with contradictory answers.

Then write the gold answer. It should:
- Surface the conflict explicitly ("Sources disagree...")
- Present both positions with their sources
- If applicable, note a resolution policy (e.g. "the more recent document supersedes")

Output as JSON:
{{
  "question": "...",
  "gold_answer": "...",
  "intent_points": [
    {{"id": "P1", "text": "Acknowledges that sources conflict", "required": true}},
    {{"id": "P2", "text": "States position from source 1 with attribution", "required": true}},
    {{"id": "P3", "text": "States position from source 2 with attribution", "required": true}},
    {{"id": "P4", "text": "Notes resolution policy or defers to user", "required": false}}
  ]
}}"""


COMPARATIVE_PROMPT = """You are generating a comparative evaluation question.

Below are two passages describing related but distinct things worth comparing:

<passage_1 source="{doc_id_1}">
{chunk_text_1}
</passage_1>

<passage_2 source="{doc_id_2}">
{chunk_text_2}
</passage_2>

Write ONE question that asks the user to compare or contrast these two things (e.g. "How does X compare to Y?").

Then write the gold answer with an explicit side-by-side comparison, preserving the correct direction \
of any difference (which one is bigger/faster/cheaper/etc).

Output as JSON:
{{
  "question": "...",
  "gold_answer": "...",
  "intent_points": [
    {{"id": "P1", "text": "...", "required": true, "from": ["passage_1"]}},
    {{"id": "P2", "text": "...", "required": true, "from": ["passage_2"]}},
    {{"id": "P3", "text": "Explicit comparison framing", "required": false}}
  ]
}}"""


COMPOUND_PROMPT = """You are merging {n} related clean questions into one compound, run-on query - the way a \
real user might ask several things in a single message.

Sub-questions and their gold answers:
{sub_qa_block}

Write ONE compound question that combines all {n} sub-questions naturally (not as a numbered list).

Then write a gold answer that addresses every sub-question.

Then produce intent points covering all sub-questions, noting which sub-question each point answers.

Output as JSON:
{{
  "question": "...",
  "gold_answer": "...",
  "intent_points": [
    {{"id": "P1", "text": "...", "required": true, "sub_question": "Q1"}},
    {{"id": "P2", "text": "...", "required": true, "sub_question": "Q2"}}
  ]
}}"""


PROCEDURAL_PROMPT = """You are generating a procedural (how-to) evaluation question.

Read the following passage, which describes a step-by-step process:

<passage>
{chunk_text}
</passage>

Write ONE "how do I..." question that this passage answers with an ordered procedure.

Then write the gold answer, preserving the exact order of steps from the passage.

Then decompose into intent points, one per step (mark all as required - dropping or reordering a step is a \
failure), plus any preferred context points.

Output as JSON:
{{
  "question": "...",
  "gold_answer": "...",
  "intent_points": [
    {{"id": "P1", "text": "Step 1: ...", "required": true}},
    {{"id": "P2", "text": "Step 2: ...", "required": true}}
  ]
}}"""


WRONG_ASSUMPTION_SEED_PROMPT = """Read the following passage:

<passage>
{chunk_text}
</passage>

Identify ONE plausible-but-wrong assumption a user might have about the topic in this passage (something \
they might believe is true but the passage contradicts or doesn't support).

Write a question that bakes in this wrong assumption (e.g. "How do I do X" when X isn't actually supported).

Then write the gold answer, which must explicitly correct the false premise while still being helpful \
(pointing to what IS true/supported).

Output as JSON:
{{
  "question": "...",
  "wrong_assumption": "brief description of the false premise",
  "gold_answer": "...",
  "intent_points": [
    {{"id": "P1", "text": "Explicitly corrects the false premise", "required": true}},
    {{"id": "P2", "text": "Provides the actual correct path/fact", "required": true}}
  ]
}}"""


AMBIGUOUS_PROMPT = """You are generating an ambiguous evaluation question - one with more than one \
reasonable interpretation.

Read the following passage(s), which cover multiple distinct things that could all plausibly be the \
referent of a vague question:

<passages>
{chunk_text}
</passages>

Write ONE question that is genuinely ambiguous (e.g. "What's the limit?" when the passage discusses several \
different limits).

Then list the acceptable interpretations, and for each, what the gold answer would be under that reading.

Then write a "best-effort" gold answer that either asks for clarification or addresses the most common \
interpretation while acknowledging the ambiguity exists.

Output as JSON:
{{
  "question": "...",
  "interpretations": [
    {{"reading": "...", "answer_under_this_reading": "..."}}
  ],
  "gold_answer": "...",
  "intent_points": [
    {{"id": "P1", "text": "Acknowledges the question could mean multiple things, or picks the most likely reading and says so", "required": true}}
  ]
}}"""


# ---------------------------------------------------------------------------
# Noise transformation prompts
# ---------------------------------------------------------------------------


VOCAB_MISMATCH_PROMPT = """Rewrite the following question as if the person asking has NEVER read the source \
documentation and is using everyday, casual language.

Rules:
1. Do NOT use any of these words or their close variants: {avoid_terms}
2. Replace technical/formal terms with how a non-expert would describe the same thing
3. Keep the underlying question the same - the same answer should still apply
4. Sound like a real user typing in a hurry, not a polished FAQ entry

Original question: {question}

Output only the rewritten question, no explanation."""


INDIRECT_PROMPT = """Rewrite the following question as if the person is describing their situation in a \
casual message to a coworker, rather than asking a direct question.

Rules:
1. Do NOT ask a direct question - describe what happened and what they're trying to figure out
2. Include some personal context
3. Sound informal - contractions and minor grammatical looseness are fine
4. The underlying information need must be the same

Original question: {question}

Output only the rewritten message, no explanation."""


FRAGMENT_PROMPT = """Rewrite the following question as terse shorthand - how someone would type it into a \
search box in a hurry.

Rules:
1. Drop articles, auxiliaries, and full sentence structure
2. Keep the essential nouns and any critical qualifiers
3. It should still be understandable, just abbreviated
4. Length: 3-6 words typically

Original question: {question}

Output only the rewritten fragment, no explanation."""


WRONG_ASSUMPTION_NOISE_PROMPT = """Rewrite the following question to bake in this false assumption, so it \
reads as a natural user question that assumes something untrue:

False assumption to bake in: {wrong_assumption}

Original question: {question}

Output only the rewritten question, no explanation."""


COMPOUND_NOISE_PROMPT = """Merge the following question with a plausible, related second question a real \
user might ask in the same breath, forming one run-on compound question.

Original question: {question}

Output only the rewritten compound question, no explanation."""


REGISTER_VARIATION_PROMPT = """Rewrite the following question at a {register} formality level.

register options:
- "very_casual": Slack-message style, contractions, minimal punctuation
- "formal": enterprise support-ticket style, complete sentences, polite framing

Original question: {question}

Output only the rewritten question, no explanation."""


LEAKAGE_CHECK_PROMPT = """Answer the following question using only your own general knowledge. Do not assume \
access to any external documents.

Question: {question}

If you can answer this confidently and correctly without needing to look anything up in a specific \
document, answer it. Otherwise, say you don't know.

Output as JSON:
{{
  "can_answer_from_memory": true or false,
  "answer": "your answer, or empty string if you cannot answer"
}}"""
