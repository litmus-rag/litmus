"""LLM response caching and checkpoint/resume support.

Cache key: sha256(prompt + model + temperature). Stored as a flat JSON map
in ``cache_dir/llm_cache.json``. Checkpoints (per-batch progress markers for
``generate()``, per-record progress for ``evaluate()``) live alongside in
``cache_dir/checkpoint.json``.

Both files are plain JSON so they're inspectable/diffable, and small enough
in practice (eval sets max out in the hundreds of records) that a full
read-modify-write on every update is not a bottleneck.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _cache_key(prompt: str, model: str, temperature: float | None) -> str:
    raw = f"{model}|{temperature}|{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CacheManager:
    def __init__(self, cache_dir: str | None):
        self.enabled = cache_dir is not None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache: dict[str, str] = {}
        self._checkpoint: dict[str, Any] = {}
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()
            self._load_checkpoint()

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / "llm_cache.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.cache_dir / "checkpoint.json"

    def _load_cache(self) -> None:
        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text())
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def _load_checkpoint(self) -> None:
        if self.checkpoint_path.exists():
            try:
                self._checkpoint = json.loads(self.checkpoint_path.read_text())
            except (json.JSONDecodeError, OSError):
                self._checkpoint = {}

    def get(self, prompt: str, model: str, temperature: float | None = None) -> str | None:
        if not self.enabled:
            return None
        return self._cache.get(_cache_key(prompt, model, temperature))

    def put(self, prompt: str, model: str, response: str, temperature: float | None = None) -> None:
        if not self.enabled:
            return
        self._cache[_cache_key(prompt, model, temperature)] = response
        self.cache_path.write_text(json.dumps(self._cache))

    def clear(self) -> None:
        self._cache = {}
        if self.enabled and self.cache_path.exists():
            self.cache_path.unlink()

    # -- checkpoint / resume --------------------------------------------

    def get_checkpoint(self, key: str, default: Any = None) -> Any:
        return self._checkpoint.get(key, default)

    def set_checkpoint(self, key: str, value: Any) -> None:
        self._checkpoint[key] = value
        if self.enabled:
            self.checkpoint_path.write_text(json.dumps(self._checkpoint))

    def clear_checkpoint(self, key: str | None = None) -> None:
        if key is None:
            self._checkpoint = {}
        else:
            self._checkpoint.pop(key, None)
        if self.enabled:
            if self.checkpoint_path.exists() and not self._checkpoint:
                self.checkpoint_path.unlink()
            elif self.enabled:
                self.checkpoint_path.write_text(json.dumps(self._checkpoint))


def clear_cache(cache_dir: str = ".litmus_cache") -> None:
    """Module-level convenience matching ``litmus.clear_cache()`` in the public API."""
    mgr = CacheManager(cache_dir)
    mgr.clear()
    mgr.clear_checkpoint()
