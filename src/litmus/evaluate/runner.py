"""Runs the user's RAG callable against eval questions with timeout + error handling."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Callable

from litmus.models import RAGResponse, normalize_rag_response


def run_rag_sync(rag: Callable, question: str, timeout: int = 60) -> tuple[RAGResponse, str | None]:
    """Run the RAG callable for one question with a wall-clock timeout.

    Uses a thread pool + future.result(timeout=...) so this works whether
    `rag` is a plain blocking function (the common case) or something that
    happens to be async-friendly internally.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rag, question)
        try:
            raw = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return RAGResponse(answer="", contexts=[]), f"RAG call timed out after {timeout}s"
        except Exception as exc:  # noqa: BLE001
            return RAGResponse(answer="", contexts=[]), f"RAG call raised: {exc}"
    try:
        return normalize_rag_response(raw), None
    except TypeError as exc:
        return RAGResponse(answer="", contexts=[]), str(exc)


async def run_rag_batch(
    rag: Callable,
    questions: list[str],
    timeout: int = 60,
    max_workers: int = 1,
) -> list[tuple[RAGResponse, str | None]]:
    """Run the RAG callable over many questions, bounded by max_workers concurrency."""
    if max_workers <= 1:
        return [run_rag_sync(rag, q, timeout) for q in questions]

    loop = asyncio.get_event_loop()
    sem = asyncio.Semaphore(max_workers)

    async def _one(question: str) -> tuple[RAGResponse, str | None]:
        async with sem:
            return await loop.run_in_executor(None, run_rag_sync, rag, question, timeout)

    return await asyncio.gather(*(_one(q) for q in questions))
