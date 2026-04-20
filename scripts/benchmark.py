"""Benchmark multiple LLM models against the same human-graded sample.

Grades the provided student-notes file (which must also contain the
human-grader columns expected by ``evaluate.evaluate``) with each requested
model, then computes agreement metrics per model. Models are run
sequentially (safer for rate limits); sections within each model run are
graded in parallel via the existing grader thread pool.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Callable, Optional

import config
from evaluate import evaluate
from grader import process_excel_file_with_key, read_rubric_and_key
from providers import create_caller

logger = logging.getLogger("osce_grader.benchmark")


# (provider_name, model_name) pairs passed in; keep the helper tolerant so
# callers can pass either `{"model": ..., "provider": ...}` dicts or tuples.
BenchmarkTarget = dict


class BenchmarkError(Exception):
    """Raised when a benchmark run cannot proceed."""


def _metrics_from_results(
    results: dict, model: str, n_students: int
) -> dict:
    """Extract the headline numbers the admin UI needs."""
    overall = results.get("_overall", {}) if isinstance(results, dict) else {}
    return {
        "model": model,
        "within1_pct": overall.get("within1_pct"),
        "mae": overall.get("mae"),
        "bias": overall.get("bias"),
        "total_comparisons": overall.get("total"),
        "est_cost_per_1k": overall.get("est_cost_per_1k"),
        "est_cost_run": overall.get("est_cost"),
        "n_students": n_students,
    }


def run_benchmark(
    targets: list[BenchmarkTarget],
    rubric_path: str,
    answer_key_path: str,
    notes_path: str,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    workers: Optional[int] = None,
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> list[dict]:
    """Run the grader + evaluator for each target model, in sequence.

    Args:
        targets: list of dicts with keys "model" and "provider".
        rubric_path / answer_key_path / notes_path: benchmark sample files.
            The notes file must already contain ``{section}_grader_1`` and
            ``{section}_grader_2`` human-score columns.
        temperature / top_p / workers: override config defaults when set.
        output_dir: where to write per-model ``*_results.xlsx`` files.
            Defaults to a temp dir.
        progress_callback: invoked as ``cb(index, total, model_name)`` right
            before each model is graded.

    Returns:
        A list of metric dicts, one per model, in the order requested.
    """
    if not targets:
        raise BenchmarkError("No benchmark targets provided.")
    for p in (rubric_path, answer_key_path, notes_path):
        if not os.path.isfile(p):
            raise BenchmarkError(f"Benchmark sample file missing: {p}")

    temperature = temperature if temperature is not None else config.TEMPERATURE
    top_p = top_p if top_p is not None else config.TOP_P
    workers = workers if workers is not None else config.MAX_WORKERS

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="osce_benchmark_")
    os.makedirs(output_dir, exist_ok=True)

    # Load rubric + answer key once; every model grades the same content.
    rubric_content, answer_key_content = read_rubric_and_key(
        rubric_path, answer_key_path
    )

    results: list[dict] = []
    total = len(targets)
    for i, tgt in enumerate(targets):
        model = tgt["model"]
        provider = tgt["provider"]

        if progress_callback:
            progress_callback(i, total, model)

        output_file = os.path.join(output_dir, f"{_safe(model)}_results.xlsx")
        if os.path.exists(output_file):
            # Clear any prior run's resume-state so we always benchmark fresh.
            try:
                os.remove(output_file)
            except OSError:
                pass

        start = time.time()
        per_model_error: Optional[str] = None
        try:
            caller = create_caller(provider, model=model)
            process_excel_file_with_key(
                caller,
                notes_path,
                rubric_content,
                answer_key_content,
                output_file,
                temperature,
                top_p,
                max_workers=workers,
            )
            metrics_raw = evaluate(output_file, model=model, verbose=False)
        except Exception as exc:
            logger.exception("Benchmark failed for %s", model)
            per_model_error = f"{type(exc).__name__}: {exc}"
            metrics_raw = {}

        elapsed = time.time() - start

        # Count the students actually graded (same as evaluate() uses).
        try:
            import pandas as pd
            n_students = len(pd.read_excel(notes_path))
        except Exception:
            n_students = 0

        metrics = _metrics_from_results(metrics_raw, model, n_students)
        metrics["provider"] = provider
        metrics["elapsed_sec"] = round(elapsed, 1)
        metrics["output_file"] = output_file
        metrics["error"] = per_model_error
        results.append(metrics)

    if progress_callback:
        progress_callback(total, total, "")
    return results


def _safe(name: str) -> str:
    """Filesystem-safe version of a model name."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
