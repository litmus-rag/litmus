"""Token counting and cost estimation helpers, built on tiktoken + litellm."""

from __future__ import annotations

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import tiktoken  # noqa: E402

_ENCODING = None


def _encoding():
    global _ENCODING
    if _ENCODING is None:
        try:
            _ENCODING = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001
            _ENCODING = tiktoken.get_encoding("gpt2")
    return _ENCODING


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoding().encode(text))


# Per-million-token USD pricing fallback for models not present in litellm's
# cost map (e.g. a fresh Azure deployment name like "gpt-5.4"). Pulled from
# the closest known-family pricing as a rough estimate; used only when
# litellm.cost_per_token raises for the given model string.
_FALLBACK_PRICE_PER_MILLION: dict[str, tuple[float, float]] = {
    "default": (5.00, 15.00),  # (input, output) USD per 1M tokens
}


def estimate_llm_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Best-effort USD cost estimate for a model call.

    Tries litellm's cost map first (bare model name, stripping any
    "provider/" prefix and Azure deployment naming quirks), then falls back
    to a flat per-million-token rate.
    """
    import litellm

    bare_model = model.split("/", 1)[-1]
    for candidate in (model, bare_model):
        try:
            input_cost, output_cost = litellm.cost_per_token(
                model=candidate, prompt_tokens=input_tokens, completion_tokens=output_tokens
            )
            return float(input_cost + output_cost)
        except Exception:  # noqa: BLE001
            continue
    price_in, price_out = _FALLBACK_PRICE_PER_MILLION["default"]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
