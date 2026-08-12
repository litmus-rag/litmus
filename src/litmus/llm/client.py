"""litellm wrapper: config loading, retry, JSON parsing, sync + async calls.

Defaults to the ``azure/gpt-5.4`` deployment on the configured Azure
OpenAI resource (see repo-root ``conf``), but any litellm model string works
if the caller supplies their own credentials via environment variables.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm  # noqa: E402  (env var must be set before import)

litellm.suppress_debug_info = True

DEFAULT_CONF_PATH = Path(__file__).resolve().parents[3] / "conf"


class LLMError(RuntimeError):
    """Raised when an LLM call fails after exhausting retries."""


def load_conf(path: str | Path | None = None) -> dict[str, str]:
    """Parse the repo's env-style `conf` file into a dict."""
    conf_path = Path(path) if path else DEFAULT_CONF_PATH
    conf: dict[str, str] = {}
    if not conf_path.exists():
        return conf
    for line in conf_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        conf[key.strip()] = value.strip()
    return conf


def configure_azure_from_conf(path: str | Path | None = None) -> bool:
    """Populate AZURE_API_KEY / AZURE_API_BASE / AZURE_API_VERSION from `conf`.

    Returns True if credentials were found and applied (either already in
    the environment, or freshly loaded from the conf file).
    """
    if os.environ.get("AZURE_API_KEY") and os.environ.get("AZURE_API_BASE"):
        return True
    conf = load_conf(path)
    if not conf.get("AZURE_API_KEY"):
        return False
    os.environ.setdefault("AZURE_API_KEY", conf["AZURE_API_KEY"])
    os.environ.setdefault("AZURE_API_BASE", conf["AZURE_API_BASE"])
    os.environ.setdefault("AZURE_API_VERSION", conf.get("AZURE_API_VERSION", "2025-04-01-preview"))
    return True


def credentials_available(path: str | Path | None = None) -> bool:
    """Best-effort check of whether a live LLM call could succeed."""
    return configure_azure_from_conf(path)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON object/array from LLM output.

    Handles: raw JSON, JSON inside markdown fences, and JSON with leading/
    trailing prose by locating the outermost balanced {...} or [...].
    """
    text = text.strip()
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"Could not extract JSON from LLM response: {text[:300]!r}")


def extract_json_array(text: str) -> list[Any]:
    """Extract a JSON array from LLM output, tolerating Azure/OpenAI JSON
    mode's inability to return a bare top-level array.

    When a prompt asks for "a JSON array" under ``json_mode=True``, Azure's
    structured-output constraint still requires a JSON *object* at the top
    level, so the model wraps the array under a single key (observed keys
    include "json", "array", "terms", etc. -- the exact key is not
    controllable via prompting). Callers that expect a bare list must go
    through this helper rather than checking ``isinstance(result, list)``
    directly, or they silently get an empty result on every JSON-mode call.
    """
    result = extract_json(text)
    if isinstance(result, list):
        return result
    if isinstance(result, dict) and len(result) == 1:
        (value,) = result.values()
        if isinstance(value, list):
            return value
    raise ValueError(f"Expected a JSON array (or single-key object wrapping one), got: {result!r}")


class LLMClient:
    """Thin synchronous/asynchronous wrapper around litellm.completion."""

    def __init__(
        self,
        model: str = "azure/gpt-5.4",
        conf_path: str | Path | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        if model.startswith("azure/"):
            configure_azure_from_conf(conf_path)

    def _build_kwargs(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1000,
        reasoning_effort: str | None = None,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1000,
        reasoning_effort: str | None = None,
        json_mode: bool = False,
    ) -> str:
        kwargs = self._build_kwargs(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            json_mode=json_mode,
        )
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = litellm.completion(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - retry on any provider error
                last_err = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_base_delay * (2**attempt))
        raise LLMError(f"LLM call failed after {self.max_retries} attempts: {last_err}") from last_err

    async def acomplete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1000,
        reasoning_effort: str | None = None,
        json_mode: bool = False,
    ) -> str:
        kwargs = self._build_kwargs(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            json_mode=json_mode,
        )
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = await litellm.acompletion(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_base_delay * (2**attempt))
        raise LLMError(f"LLM call failed after {self.max_retries} attempts: {last_err}") from last_err

    def complete_json(self, prompt: str, **kwargs) -> Any:
        text = self.complete(prompt, json_mode=True, **kwargs)
        return extract_json(text)

    def complete_json_array(self, prompt: str, **kwargs) -> list[Any]:
        text = self.complete(prompt, json_mode=True, **kwargs)
        return extract_json_array(text)

    async def acomplete_json(self, prompt: str, **kwargs) -> Any:
        text = await self.acomplete(prompt, json_mode=True, **kwargs)
        return extract_json(text)

    async def amap(
        self,
        prompts: list[str],
        *,
        max_concurrency: int = 5,
        json_mode: bool = False,
        **kwargs,
    ) -> list[Any]:
        """Run many prompts concurrently, bounded by a semaphore."""
        sem = asyncio.Semaphore(max_concurrency)

        async def _one(p: str) -> Any:
            async with sem:
                if json_mode:
                    return await self.acomplete_json(p, **kwargs)
                return await self.acomplete(p, **kwargs)

        return await asyncio.gather(*(_one(p) for p in prompts))
