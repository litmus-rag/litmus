"""Full generation pipeline: chunk, budget, generate, noise, leakage, checkpoint.

Concurrency note: ``max_workers > 1`` parallelizes the per-record LLM calls
using a thread pool rather than asyncio. The question-type generator
functions in ``question_types.py`` are synchronous (they call
``LLMClient.complete_json``), and litellm's underlying HTTP call releases
the GIL while waiting on the network, so a thread pool gets real
concurrency without duplicating every generator into an async twin. This is
a deliberate simplification versus a fully async pipeline; documented in
CLAUDE.md.
"""

from __future__ import annotations

import hashlib
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from litmus.cache.manager import CacheManager
from litmus.config import NOISE_TRANSFORMATION_STRATEGY, get_tier_config
from litmus.generate.bridge import find_bridge_candidates
from litmus.generate.leakage import filter_leaked_records
from litmus.generate.noise import apply_noise_layers
from litmus.generate.question_types import (
    generate_adversarial,
    generate_ambiguous,
    generate_comparative,
    generate_compound,
    generate_contradiction,
    generate_cross_doc,
    generate_procedural,
    generate_single_chunk,
    generate_unanswerable,
    generate_wrong_assumption,
    pick_distractors,
)
from litmus.generate.sizing import calculate_size
from litmus.ingest.chunker import chunk_documents
from litmus.ingest.contradiction_detector import find_contradictions
from litmus.ingest.loader import load_directory
from litmus.llm.client import LLMClient
from litmus.models import (
    Chunk,
    Difficulty,
    EvalRecord,
    EvalSet,
    EvalSetMetadata,
    IntentPoint,
    NoiseType,
    QuestionType,
)

_HARD_TYPES = {
    QuestionType.CROSS_DOC,
    QuestionType.CONTRADICTION,
    QuestionType.COMPARATIVE,
    QuestionType.ADVERSARIAL,
    QuestionType.AMBIGUOUS,
}


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[litmus.generate] {message}")


def _checkpoint_scope(docs_dir: str, tier: str) -> str:
    """Fingerprint identifying this corpus+tier, so checkpoint keys never
    collide across two different generate() calls sharing a cache_dir."""
    return hashlib.sha256(f"{Path(docs_dir).resolve()}|{tier}".encode()).hexdigest()[:16]


def _run_batch(fn: Callable[[Any], Any], items: list[Any], max_workers: int) -> list[Any]:
    """Run fn(item) over items, catching per-item failures (returns None for failures)."""
    if max_workers <= 1 or len(items) <= 1:
        results = []
        for item in items:
            try:
                results.append(fn(item))
            except Exception:  # noqa: BLE001
                results.append(None)
        return results

    results: list[Any] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:  # noqa: BLE001
                results[idx] = None
    return results


def _assign_difficulty(question_type: QuestionType, noise_types: list[NoiseType]) -> Difficulty:
    is_clean = noise_types == [NoiseType.CLEAN]
    if question_type in _HARD_TYPES:
        return Difficulty.MEDIUM if is_clean else Difficulty.HARD
    if question_type == QuestionType.SINGLE_CHUNK:
        return Difficulty.EASY if is_clean else Difficulty.MEDIUM
    return Difficulty.MEDIUM


def _build_record(
    record_id: str,
    raw: dict[str, Any],
    question_type: QuestionType,
    chunks: dict[str, Chunk],
    client: LLMClient,
    noise_layer_name: str,
    rng: random.Random,
) -> EvalRecord:
    question_clean = raw.get("question", "")
    gold_chunk_ids = raw.get("gold_chunk_ids", [])
    gold_chunks_text = [[chunks[cid].text for cid in group if cid in chunks] for group in gold_chunk_ids]

    layers = NOISE_TRANSFORMATION_STRATEGY.get(noise_layer_name, [])
    source_chunk_text = gold_chunks_text[0][0] if gold_chunks_text and gold_chunks_text[0] else ""
    scoring_notes = raw.get("scoring_notes", "")
    wrong_assumption_text = scoring_notes if question_type == QuestionType.WRONG_ASSUMPTION else ""

    if layers and question_clean:
        question_final, noise_applied = apply_noise_layers(
            question_clean,
            layers,
            client,
            source_chunk_text=source_chunk_text,
            wrong_assumption=wrong_assumption_text,
            seed=rng.randint(0, 2**31 - 1),
        )
    else:
        question_final, noise_applied = question_clean, [NoiseType.CLEAN]

    intent_points = [
        IntentPoint(
            id=p.get("id", f"P{i + 1}"),
            text=p.get("text", ""),
            required=bool(p.get("required", True)),
        )
        for i, p in enumerate(raw.get("intent_points", []))
        if isinstance(p, dict)
    ]

    source_doc_ids = list(dict.fromkeys(raw.get("source_doc_ids", [])))

    return EvalRecord(
        id=record_id,
        question=question_final,
        question_clean=question_clean,
        question_type=question_type,
        noise_profile=noise_applied,
        difficulty=_assign_difficulty(question_type, noise_applied),
        gold_answer=raw.get("gold_answer", ""),
        gold_chunk_ids=gold_chunk_ids,
        gold_chunks_text=gold_chunks_text,
        intent_points=intent_points,
        unanswerable=bool(raw.get("unanswerable", False)),
        requires_synthesis=bool(raw.get("requires_synthesis", False)),
        domain_tags=source_doc_ids,
        source_doc_ids=source_doc_ids,
        scoring_notes=scoring_notes if isinstance(scoring_notes, str) else str(scoring_notes),
    )


def _chunk_cycle(chunks: list[Chunk], rng: random.Random):
    pool = list(chunks)
    rng.shuffle(pool)
    i = 0
    while True:
        if i >= len(pool):
            rng.shuffle(pool)
            i = 0
        yield pool[i]
        i += 1


def _generate_type_batch(
    question_type: QuestionType,
    count: int,
    chunks: dict[str, Chunk],
    client: LLMClient,
    rng: random.Random,
    max_workers: int,
    *,
    bridge_candidates=None,
    contradiction_pairs=None,
    comparative_pairs=None,
) -> list[dict[str, Any]]:
    chunk_list = list(chunks.values())
    if not chunk_list or count <= 0:
        return []
    cycle = _chunk_cycle(chunk_list, rng)

    if question_type == QuestionType.SINGLE_CHUNK:
        picks = [next(cycle) for _ in range(count)]
        results = _run_batch(lambda c: generate_single_chunk(c, client), picks, max_workers)

    elif question_type == QuestionType.PROCEDURAL:
        picks = [next(cycle) for _ in range(count)]
        results = _run_batch(lambda c: generate_procedural(c, client), picks, max_workers)

    elif question_type == QuestionType.WRONG_ASSUMPTION:
        picks = [next(cycle) for _ in range(count)]
        results = _run_batch(lambda c: generate_wrong_assumption(c, client), picks, max_workers)

    elif question_type == QuestionType.UNANSWERABLE:
        sample_size = min(8, len(chunk_list))
        picks = [rng.sample(chunk_list, sample_size) for _ in range(count)]
        results = _run_batch(lambda cs: generate_unanswerable(cs, client), picks, max_workers)

    elif question_type == QuestionType.ADVERSARIAL:
        picks = [next(cycle) for _ in range(count)]

        def _adv(chunk: Chunk) -> dict | None:
            distractors = pick_distractors(chunk, chunks, k=2, seed=rng.randint(0, 2**31 - 1))
            if not distractors:
                return None
            return generate_adversarial(chunk, distractors, client)

        results = _run_batch(_adv, picks, max_workers)

    elif question_type == QuestionType.AMBIGUOUS:
        picks = [rng.sample(chunk_list, min(2, len(chunk_list))) for _ in range(count)]
        results = _run_batch(lambda cs: generate_ambiguous(cs, client), picks, max_workers)

    elif question_type == QuestionType.CROSS_DOC:
        candidates = bridge_candidates or []
        if not candidates:
            return []
        picks = [candidates[i % len(candidates)] for i in range(count)]
        results = _run_batch(lambda cand: generate_cross_doc(cand, chunks, client), picks, max_workers)
        results = [r for r in results if r is not None]

    elif question_type == QuestionType.CONTRADICTION:
        pairs = contradiction_pairs or []
        if not pairs:
            return []
        picks = [pairs[i % len(pairs)] for i in range(count)]
        results = _run_batch(
            lambda pair: generate_contradiction(chunks[pair.chunk_id_a], chunks[pair.chunk_id_b], client),
            picks,
            max_workers,
        )

    elif question_type == QuestionType.COMPARATIVE:
        pairs = comparative_pairs or []
        if not pairs:
            return []
        picks = [pairs[i % len(pairs)] for i in range(count)]
        results = _run_batch(
            lambda pair: generate_comparative(chunks[pair[0]], chunks[pair[1]], client), picks, max_workers
        )

    elif question_type == QuestionType.COMPOUND:
        sub_picks = [next(cycle) for _ in range(count * 2)]
        sub_results = _run_batch(lambda c: generate_single_chunk(c, client), sub_picks, max_workers)
        sub_results = [r for r in sub_results if r]
        results = []
        for i in range(0, len(sub_results) - 1, 2):
            try:
                results.append(generate_compound(sub_results[i : i + 2], client))
            except Exception:  # noqa: BLE001
                results.append(None)
            if len(results) >= count:
                break

    else:
        raise ValueError(f"No generator wired for question type {question_type!r}")

    return [r for r in results if r]


def _find_comparative_pairs(chunks: dict[str, Chunk], max_pairs: int = 20) -> list[tuple[str, str]]:
    """Pick cross-document chunk pairs with moderate topical similarity for comparison."""
    from litmus.ingest.embedder import embed_texts, top_similar_pairs

    ids = list(chunks.keys())
    if len(ids) < 2:
        return []
    embeddings = embed_texts([chunks[i].text for i in ids])
    doc_map = {i: chunks[i].doc_id for i in ids}
    pairs = top_similar_pairs(embeddings, ids, threshold=0.3, exclude_same_doc=doc_map)
    return [(a, b) for a, b, _ in pairs[:max_pairs]]


def generate_eval_set(
    docs_dir: str,
    llm: str = "azure/gpt-5.4",
    tier: str = "medium",
    size: int | str = "auto",
    save_path: str | None = None,
    chunking: str = "auto",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    seed: int | None = None,
    max_workers: int = 1,
    cache_dir: str | None = ".litmus_cache",
    verbose: bool = True,
) -> EvalSet:
    tier_config = get_tier_config(tier)
    rng = random.Random(seed)
    client = LLMClient(model=llm)
    cache = CacheManager(cache_dir)
    # Scope checkpoint keys to this corpus+tier so two different generate()
    # calls sharing a cache_dir never resume each other's records -- a flat
    # "generate:{type_name}" key previously collided across any two corpora
    # generated into the same cache_dir, silently splicing one corpus's
    # records into another's eval set.
    checkpoint_scope = _checkpoint_scope(docs_dir, tier)

    if cache_dir:
        _log(verbose, f"Cache/checkpoint directory: {Path(cache_dir).resolve()}")

    _log(verbose, f"Loading documents from {docs_dir}")
    docs = load_directory(docs_dir)
    chunks = chunk_documents(docs, chunking=chunking, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    _log(verbose, f"{len(docs)} documents -> {len(chunks)} chunks")

    if size == "auto":
        total_size = calculate_size(len(docs), len(chunks), tier)
    else:
        total_size = int(size)
    _log(verbose, f"Target eval set size: {total_size} questions (tier={tier})")

    type_proportions = tier_config["question_types"]
    counts = {qt: round(total_size * prop) for qt, prop in type_proportions.items()}
    diff = total_size - sum(counts.values())
    if counts:
        largest_type = max(counts, key=counts.get)
        counts[largest_type] += diff

    # Pre-compute expensive shared context once, reused across type batches.
    bridge_candidates = None
    contradiction_pairs = None
    comparative_pairs = None
    if counts.get("cross_doc", 0) > 0:
        _log(verbose, "Scanning for bridge entities (cross-doc question support)...")
        # Scale with document count (at least a few chunks per doc) rather than a flat
        # cap -- a fixed 40-chunk scan starves corpora with many documents of any real
        # chance to observe the same entity recur across documents.
        scan_limit = min(len(chunks), max(40, len(docs) * 3))
        bridge_candidates = find_bridge_candidates(chunks, client, max_chunks_to_scan=scan_limit)
        _log(verbose, f"Found {len(bridge_candidates)} bridge-entity candidates")
    if counts.get("contradiction", 0) > 0:
        _log(verbose, "Scanning for contradictions...")
        contradiction_pairs = find_contradictions(chunks, client)
        if not contradiction_pairs:
            _log(verbose, "No natural contradictions found in this corpus.")
    if counts.get("comparative", 0) > 0:
        comparative_pairs = _find_comparative_pairs(chunks)

    all_records: list[EvalRecord] = []
    noise_layers_dist = tier_config["noise_layers"]
    noise_keys = list(noise_layers_dist.keys())
    noise_weights = list(noise_layers_dist.values())

    for type_name, count in counts.items():
        if count <= 0:
            continue
        checkpoint_key = f"generate:{checkpoint_scope}:{type_name}"
        cached = cache.get_checkpoint(checkpoint_key)
        if cached:
            _log(verbose, f"Resuming {type_name} from checkpoint ({len(cached)} records)")
            for entry in cached:
                all_records.append(EvalRecord.from_dict(entry))
            continue

        question_type = QuestionType(type_name)
        _log(verbose, f"Generating {count} {type_name} questions...")
        raw_batch = _generate_type_batch(
            question_type,
            count,
            chunks,
            client,
            rng,
            max_workers,
            bridge_candidates=bridge_candidates,
            contradiction_pairs=contradiction_pairs,
            comparative_pairs=comparative_pairs,
        )

        type_records = []
        for i, raw in enumerate(raw_batch):
            noise_choice = rng.choices(noise_keys, weights=noise_weights, k=1)[0] if noise_keys else "clean"
            record = _build_record(
                record_id=f"eval-{type_name}-{i:04d}",
                raw=raw,
                question_type=question_type,
                chunks=chunks,
                client=client,
                noise_layer_name=noise_choice,
                rng=rng,
            )
            type_records.append(record)

        all_records.extend(type_records)
        cache.set_checkpoint(checkpoint_key, [r.to_dict() for r in type_records])
        _log(verbose, f"  -> {len(type_records)}/{count} generated successfully")

    if "leakage" in tier_config["quality_checks"]:
        _log(verbose, "Running leakage filter...")
        all_records, discarded = filter_leaked_records(all_records, client)
        if discarded:
            _log(verbose, f"Discarded {len(discarded)} leaked questions (answerable from parametric memory)")

    # Re-index ids to be contiguous and stable after any filtering.
    for i, record in enumerate(all_records):
        record.id = f"eval-{i:04d}"

    from litmus.version import compute_doc_hashes

    metadata = EvalSetMetadata(
        docs_dir=docs_dir,
        llm=llm,
        num_source_docs=len(docs),
        num_chunks=len(chunks),
        generated_at=str(time.time()),
        chunking=chunking if isinstance(chunking, str) else "callable",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        seed=seed,
        doc_hashes=compute_doc_hashes(docs_dir),
    )

    eval_set = EvalSet(records=all_records, chunks=chunks, metadata=metadata, tier=tier)

    if save_path:
        eval_set.save(save_path)
        _log(verbose, f"Saved eval set to {Path(save_path).resolve()}")
    else:
        _log(verbose, "No save_path given - eval set was NOT written to disk (only returned in memory).")

    cache.clear_checkpoint()
    _log(verbose, f"Done: {len(all_records)} eval records generated.")
    return eval_set
