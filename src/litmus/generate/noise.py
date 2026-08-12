"""Realism transformation layers (Section 4.2 of the synth methodology doc).

Layer 1 Vocabulary Mismatch, Layer 3 Indirect Phrasing, Layer 4 Fragment,
Layer 5 Wrong Assumption, Layer 6 Compound, and Layer 7 Register Variation
all go through the LLM (prompts in generate/prompts.py).

Layer 2 Typos is deliberately NOT an LLM call — the spec is explicit that
LLMs over-correct or produce unrealistic typos. Typos are injected
programmatically here using a small set of realistic error patterns
(adjacent-key swap, dropped letter, doubled letter, phonetic misspelling).
"""

from __future__ import annotations

import random
import re

from litmus.generate.prompts import (
    COMPOUND_NOISE_PROMPT,
    FRAGMENT_PROMPT,
    INDIRECT_PROMPT,
    REGISTER_VARIATION_PROMPT,
    TERM_EXTRACTION_PROMPT,
    VOCAB_MISMATCH_PROMPT,
    WRONG_ASSUMPTION_NOISE_PROMPT,
)
from litmus.llm.client import LLMClient
from litmus.models import NoiseType

# Adjacent-key map for keyboard-proximity typos (QWERTY).
_ADJACENT_KEYS = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "erfcxs", "e": "wsdr",
    "f": "rtgdcv", "g": "tyhfvb", "h": "yujgbn", "i": "ujko", "j": "uikhnm",
    "k": "iojlm", "l": "opk", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}

_PHONETIC_SUBS = {
    "guarantee": "garuntee",
    "separate": "seperate",
    "definitely": "definately",
    "receive": "recieve",
    "necessary": "neccessary",
    "occurred": "occured",
    "reimbursement": "reimbursment",
    "authentication": "authentification",
    "cancellation": "cancelation",
    "eligibility": "elligibility",
}


def _adjacent_key_swap(word: str, rng: random.Random) -> str:
    if len(word) < 3:
        return word
    idx = rng.randrange(1, len(word) - 1)
    ch = word[idx].lower()
    if ch not in _ADJACENT_KEYS:
        return word
    replacement = rng.choice(_ADJACENT_KEYS[ch])
    return word[:idx] + replacement + word[idx + 1 :]


def _dropped_letter(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    idx = rng.randrange(1, len(word) - 1)
    return word[:idx] + word[idx + 1 :]


def _doubled_letter(word: str, rng: random.Random) -> str:
    if len(word) < 3:
        return word
    idx = rng.randrange(1, len(word) - 1)
    return word[: idx + 1] + word[idx] + word[idx + 1 :]


def _phonetic_misspelling(word: str) -> str | None:
    lower = word.lower()
    if lower in _PHONETIC_SUBS:
        sub = _PHONETIC_SUBS[lower]
        if word[0].isupper():
            sub = sub.capitalize()
        return sub
    return None


_TYPO_STRATEGIES = [_adjacent_key_swap, _dropped_letter, _doubled_letter]


def inject_typos(question: str, num_typos: int = 1, seed: int | None = None) -> str:
    """Introduce 1-2 realistic typos into a question, programmatically.

    Words shorter than 3 characters and non-alphabetic tokens are skipped as
    typo targets so the question stays legible.
    """
    rng = random.Random(seed)
    words = question.split(" ")
    candidate_indices = [i for i, w in enumerate(words) if re.match(r"^[A-Za-z]{3,}[?.,!]?$", w)]
    if not candidate_indices:
        return question
    rng.shuffle(candidate_indices)
    applied = 0
    for idx in candidate_indices:
        if applied >= num_typos:
            break
        word = words[idx]
        trailing = ""
        core = word
        if core and core[-1] in "?.,!":
            trailing = core[-1]
            core = core[:-1]
        if not core:
            continue
        phonetic = _phonetic_misspelling(core)
        if phonetic:
            words[idx] = phonetic + trailing
            applied += 1
            continue
        strategy = rng.choice(_TYPO_STRATEGIES)
        mutated = strategy(core, rng)
        if mutated != core:
            words[idx] = mutated + trailing
            applied += 1
    return " ".join(words)


def apply_vocab_mismatch(question: str, source_chunk_text: str, client: LLMClient) -> str:
    try:
        terms_raw = client.complete_json_array(
            TERM_EXTRACTION_PROMPT.format(chunk_text=source_chunk_text[:2000]),
            temperature=0.0,
            max_tokens=200,
        )
        terms = ", ".join(str(t) for t in terms_raw)
    except Exception:  # noqa: BLE001
        terms = ""
    prompt = VOCAB_MISMATCH_PROMPT.format(avoid_terms=terms or "(none identified)", question=question)
    return client.complete(prompt, temperature=0.7, max_tokens=200).strip()


def apply_indirect(question: str, client: LLMClient) -> str:
    return client.complete(INDIRECT_PROMPT.format(question=question), temperature=0.7, max_tokens=250).strip()


def apply_fragment(question: str, client: LLMClient) -> str:
    return client.complete(FRAGMENT_PROMPT.format(question=question), temperature=0.5, max_tokens=50).strip()


def apply_wrong_assumption_noise(question: str, wrong_assumption: str, client: LLMClient) -> str:
    prompt = WRONG_ASSUMPTION_NOISE_PROMPT.format(wrong_assumption=wrong_assumption, question=question)
    return client.complete(prompt, temperature=0.6, max_tokens=200).strip()


def apply_compound_noise(question: str, client: LLMClient) -> str:
    return client.complete(COMPOUND_NOISE_PROMPT.format(question=question), temperature=0.7, max_tokens=250).strip()


def apply_register_variation(question: str, register: str, client: LLMClient) -> str:
    prompt = REGISTER_VARIATION_PROMPT.format(register=register, question=question)
    return client.complete(prompt, temperature=0.6, max_tokens=200).strip()


def apply_noise_layers(
    question_clean: str,
    layers: list[str],
    client: LLMClient,
    *,
    source_chunk_text: str = "",
    wrong_assumption: str = "",
    seed: int | None = None,
) -> tuple[str, list[NoiseType]]:
    """Apply a sequence of noise layers, returning (final_question, applied_types)."""
    question = question_clean
    applied: list[NoiseType] = []
    for layer in layers:
        if layer == "vocab_mismatch":
            question = apply_vocab_mismatch(question, source_chunk_text, client)
            applied.append(NoiseType.VOCAB_MISMATCH)
        elif layer == "indirect":
            question = apply_indirect(question, client)
            applied.append(NoiseType.INDIRECT)
        elif layer == "typo":
            question = inject_typos(question, num_typos=1, seed=seed)
            applied.append(NoiseType.TYPO)
        elif layer == "fragment":
            question = apply_fragment(question, client)
            applied.append(NoiseType.FRAGMENT)
        elif layer == "wrong_assumption":
            question = apply_wrong_assumption_noise(question, wrong_assumption, client)
            applied.append(NoiseType.WRONG_ASSUMPTION)
        elif layer == "compound":
            question = apply_compound_noise(question, client)
            applied.append(NoiseType.COMPOUND)
        elif layer == "register_variation":
            register = random.Random(seed).choice(["very_casual", "formal"])
            question = apply_register_variation(question, register, client)
            applied.append(NoiseType.REGISTER_VARIATION)
        else:
            raise ValueError(f"Unknown noise layer: {layer!r}")
    if not applied:
        applied = [NoiseType.CLEAN]
    return question, applied
