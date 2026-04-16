"""OSCE Grader - AI-powered grading for medical student post-encounter notes.

Reads student notes, a rubric, and an answer key from Excel files, sends each
section to an LLM for evaluation, and writes scored results back to Excel.

Supports multiple LLM providers: OpenAI, Anthropic (Claude), and Google (Gemini).
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd

import config
from providers import LLMCaller, SUPPORTED_PROVIDERS, create_caller
from run_context import RunContext  # noqa: F401  — re-exported for typing


class GraderError(Exception):
    """Raised when a grading operation fails (bad input, missing files, etc.)."""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("osce_grader")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_excel_safe(path: str, label: str) -> pd.DataFrame:
    """Read an Excel file with user-friendly error handling.

    *label* is a human-readable name for the file (e.g. "rubric",
    "answer key", "student notes") used in error messages.
    """
    try:
        return pd.read_excel(path)
    except Exception as exc:
        logger.error(
            "Failed to read %s file '%s': %s",
            label,
            path,
            exc,
        )
        raise GraderError(f"Failed to read {label} file '{path}': {exc}")


def read_rubric_and_key(
    rubric_file: str, answer_key_file: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Load the rubric and answer key from Excel files.

    Each file is expected to have a header row with section names and at least
    one data row containing the rubric / answer-key text for each section.

    Raises GraderError if either file is empty or cannot be read.
    """
    rubric_df = _read_excel_safe(rubric_file, "rubric")
    if rubric_df.empty:
        raise GraderError(f"Rubric file '{rubric_file}' contains no data rows.")
    rubric: dict[str, str] = rubric_df.iloc[0].to_dict()

    answer_key_df = _read_excel_safe(answer_key_file, "answer key")
    if answer_key_df.empty:
        raise GraderError(f"Answer-key file '{answer_key_file}' contains no data rows.")
    answer_key: dict[str, str] = answer_key_df.iloc[0].to_dict()

    return rubric, answer_key


# ---------------------------------------------------------------------------
# Interaction logging (to file)
# ---------------------------------------------------------------------------
_log_lock = threading.Lock()


def log_interaction(
    log_file: str,
    messages: list[dict[str, str]],
    response: str,
) -> None:
    """Append an LLM interaction to the log file.

    Thread-safe via ``_log_lock``.  Failures are logged as warnings but do
    not interrupt grading.
    """
    try:
        with _log_lock, open(log_file, "a", encoding="utf-8") as log:
            log.write("----- Interaction -----\n")
            for message in messages:
                log.write(f"{message['role']}: {message['content']}\n")
            log.write(f"Response: {response}\n")
            log.write("-----------------------\n\n")
    except OSError as exc:
        logger.warning("Could not write to log file '%s': %s", log_file, exc)


# ---------------------------------------------------------------------------
# LLM API wrapper
# ---------------------------------------------------------------------------

def call_llm(
    caller: LLMCaller,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
) -> str:
    """Call the LLM with retry logic.

    Retries up to ``config.MAX_RETRIES`` times with exponential back-off on
    transient failures (rate-limits, network errors, server errors).
    """
    from audit import log_event as _audit_log
    delay = config.RETRY_DELAY
    last_exception: Optional[Exception] = None
    _provider = getattr(caller, "provider", "unknown")
    _model = getattr(caller, "model", "unknown")

    for attempt in range(1, config.MAX_RETRIES + 1):
        t0 = time.time()
        try:
            response = caller(messages, temperature, top_p)
            latency_ms = int((time.time() - t0) * 1000)
            _audit_log(
                "llm.call",
                stream="system",
                severity="info",
                outcome="success",
                detail={
                    "provider": _provider,
                    "model": _model,
                    "latency_ms": latency_ms,
                    "attempt": attempt,
                },
            )
            return response
        except Exception as exc:
            last_exception = exc
            if attempt < config.MAX_RETRIES:
                logger.warning(
                    "API call failed (attempt %d/%d): %s",
                    attempt,
                    config.MAX_RETRIES,
                    exc,
                )
                logger.info("Retrying in %ds...", delay)
                _audit_log(
                    "llm.retry",
                    stream="system",
                    severity="warn",
                    outcome="failure",
                    error_code=type(exc).__name__,
                    detail={
                        "provider": _provider,
                        "model": _model,
                        "attempt": attempt,
                        "retry_delay_s": delay,
                    },
                )
                time.sleep(delay)
                delay *= 2  # exponential back-off
            else:
                logger.error(
                    "API call failed after %d attempts: %s",
                    config.MAX_RETRIES,
                    exc,
                )
                _audit_log(
                    "llm.failure",
                    stream="system",
                    severity="error",
                    outcome="failure",
                    error_code=type(exc).__name__,
                    detail={
                        "provider": _provider,
                        "model": _model,
                        "attempt": attempt,
                    },
                )
                raise last_exception  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

def extract_score(text: str) -> Optional[int]:
    """Extract the numeric score from the grading response.

    Tries patterns in order of specificity:
      1. A bare integer on its own line (what the prompt asks for).
      2. A "Score: N" or "Score = N" pattern (common LLM format).
      3. Last bare integer anywhere in the text (broadest fallback).

    Returns ``None`` if no integer can be found.
    """
    # 1. Bare integer on its own line (MULTILINE so ^ / $ match each line)
    matches = re.findall(r"^\s*(\d+)\s*$", text, re.MULTILINE)
    if matches:
        return int(matches[-1])

    # 2. "Score: N", "Score = N", "Score - N" patterns
    match = re.search(
        r"(?:score|total)\s*[:=\-]\s*(\d+)", text, re.IGNORECASE
    )
    if match:
        return int(match.group(1))

    # 3. Last standalone integer anywhere in the text
    matches = re.findall(r"\b(\d+)\b", text)
    if matches:
        return int(matches[-1])

    return None


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def _row_already_graded(row: pd.Series, sections: list[str]) -> bool:
    """Check whether a row has already been fully graded.

    Returns True if every section that has content also has a score.
    Used for resume capability — skip rows that were graded in a
    previous (interrupted) run.
    """
    for section in sections:
        has_content = pd.notna(row.get(section))
        has_score = pd.notna(row.get(f"{section}_gpt_score"))
        if has_content and not has_score:
            return False
    return True


def grade_section_with_key(
    caller: LLMCaller,
    rubric_content: dict[str, str],
    answer_key_content: dict[str, str],
    section_content: str,
    section_name: str,
    log_file: str,
    temperature: float,
    top_p: float,
) -> tuple[str, Optional[int]]:
    """Grade a single section by sending it to the LLM with the rubric and key.

    Returns a tuple of ``(explanation_text, numeric_score)``.
    """
    rubric_text = rubric_content.get(
        section_name.lower(), "No rubric available for this section."
    )
    answer_key_text = answer_key_content.get(section_name.lower(), "")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": config.GRADING_PROMPT},
        {
            "role": "user",
            "content": (
                f"Refer to the rubric: {rubric_text}.\n"
                f"Here is the answer key for {section_name}: {answer_key_text}.\n"
                f"Please evaluate the following {section_name} and provide a "
                f"score: {section_content}"
            ),
        },
    ]

    score_text = call_llm(caller, messages, temperature, top_p)
    log_interaction(log_file, messages, score_text)

    numeric_score = extract_score(score_text)
    if numeric_score is None:
        logger.warning(
            "Could not extract a numeric score for section '%s'. "
            "Saving explanation only.",
            section_name,
        )

    return score_text, numeric_score


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def compute_summary_stats(df: pd.DataFrame, sections: list[str]) -> list[dict]:
    """Return per-section stats as a list of dicts."""
    stats = []
    for section in sections:
        score_col = f"{section}_gpt_score"
        if score_col in df.columns:
            scores = pd.to_numeric(df[score_col], errors="coerce").dropna()
            if len(scores) > 0:
                stats.append({
                    "section": section.upper(),
                    "mean": round(scores.mean(), 1),
                    "median": round(scores.median(), 1),
                    "min": int(scores.min()),
                    "max": int(scores.max()),
                    "std": round(scores.std(), 1),
                    "count": len(scores),
                })
    return stats


def run_dry_run(
    notes_path: str,
    rubric_path: str,
    answer_key_path: str,
) -> dict:
    """Count API calls and estimate cost without actually grading.

    Returns a dict with summary info.
    """
    rubric_content, answer_key_content = read_rubric_and_key(
        rubric_path, answer_key_path
    )
    df = _read_excel_safe(notes_path, "student notes")
    sections = config.SECTIONS

    missing = validate_input_columns(df, sections)
    if missing:
        raise ValueError(
            f"Missing columns in student notes: {', '.join(missing)}. "
            f"Expected: {', '.join(sections)}"
        )

    total_rows = len(df)
    if total_rows == 0:
        raise ValueError("Student notes file contains no data rows.")

    api_calls = 0
    for _, row in df.iterrows():
        for section in sections:
            if pd.notna(row.get(section)):
                api_calls += 1

    cost_per_call = config.MODEL_COSTS.get(config.MODEL, (0.15, 0.60))
    # Rough estimate: ~1K input tokens + ~0.5K output tokens per call
    est_cost = api_calls * (cost_per_call[0] * 1.0 / 1000 + cost_per_call[1] * 0.5 / 1000)

    return {
        "total_rows": total_rows,
        "api_calls": api_calls,
        "est_cost": f"${est_cost:.2f}",
        "model": config.MODEL,
        "sections": sections,
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_input_columns(
    df: pd.DataFrame, sections: list[str]
) -> list[str]:
    """Return the names of any expected section columns missing from *df*."""
    return [s for s in sections if s not in df.columns]


def validate_files_exist(*paths: str) -> None:
    """Raise ``SystemExit`` if any of the given file paths do not exist."""
    for path in paths:
        if not os.path.isfile(path):
            logger.error("Input file not found: '%s'", path)
            raise SystemExit(1)


def validate_output_directory(output_path: str) -> None:
    """Raise ``SystemExit`` if the output directory does not exist."""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        logger.error(
            "Output directory does not exist: '%s'. "
            "Please create it before running the grader.",
            output_dir,
        )
        raise SystemExit(1)


def _save_results(df: pd.DataFrame, output_file: str) -> None:
    """Write the DataFrame to an Excel file with error handling."""
    try:
        df.to_excel(output_file, index=False)
    except Exception as exc:
        logger.error(
            "Failed to write results to '%s': %s. "
            "Graded data may be lost.",
            output_file,
            exc,
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Generic assessment processing loop
# ---------------------------------------------------------------------------

def process_assessment(
    assessment_type,
    caller: LLMCaller,
    file_paths: dict,
    output_file: str,
    temperature: float,
    top_p: float,
    max_workers: int = 4,
    progress_callback=None,
    *,
    ctx: "RunContext | None" = None,
) -> pd.DataFrame:
    """Grade student responses using any AssessmentType implementation.

    This is the generic grading loop that works with the AssessmentType
    abstraction.  It does not modify any existing UK-specific code paths.

    Args:
        assessment_type: An AssessmentType instance.
        caller: An LLMCaller for making LLM API calls.
        file_paths: Dict of file paths (keys depend on assessment type).
        output_file: Path to write the results Excel file.
        temperature: LLM temperature parameter.
        top_p: LLM top-p parameter.
        max_workers: Number of sections to grade in parallel per student.
        progress_callback: Optional callable(current, total) for progress.
        ctx: Optional RunContext whose settings override temperature/top_p/workers.

    Returns:
        The results DataFrame.
    """
    if ctx is not None:
        temperature = ctx.temperature
        top_p = ctx.top_p
        max_workers = ctx.workers

    from assessment_types.base import GradingResult

    log_file = os.path.splitext(output_file)[0] + ".log"
    sections = assessment_type.get_sections()

    # --- Load inputs ---
    df, rubric_data = assessment_type.load_inputs(**file_paths)

    # --- Load rubric criteria ---
    # Check for DB-stored rubric first (synthetic data path — no LLM re-parsing)
    rubric_id = file_paths.get("rubric_id") or rubric_data.get("rubric_id")
    if rubric_id:
        from database import get_rubric_sections_as_parsed
        rubric_data["parsed_rubric"] = get_rubric_sections_as_parsed(rubric_id)
        for sec_key, sec_data in rubric_data["parsed_rubric"].items():
            has_checklist = bool(sec_data.get("checklist_items"))
            has_levels = bool(sec_data.get("score_levels"))
            criteria_preview = sec_data.get("criteria", "")[:80]
            logger.info(
                "  Section %s: max=%s checklist=%s levels=%s criteria=%s...",
                sec_key, sec_data.get("max_score"), has_checklist, has_levels,
                criteria_preview,
            )
        logger.info(
            "Rubric loaded from database (id=%s). Sections: %s",
            rubric_id, list(rubric_data["parsed_rubric"].keys()),
        )
    elif hasattr(assessment_type, '_rubric_task_type') and rubric_data.get("rubric_path"):
        # Fallback: parse rubric .docx via LLM for uploaded files
        from assessment_types.kpsom_osce import parse_rubric_with_llm

        task_type = assessment_type._rubric_task_type
        if task_type in ("checklist", "milestone"):
            parse_type = task_type
        else:
            parse_type = task_type

        logger.info("Parsing rubric with LLM (one-time)...")
        if hasattr(assessment_type, '_rubric_parse_prompt'):
            import json as _json
            raw_text = __import__('convert_rubric').convert_docx_to_text(
                rubric_data["rubric_path"]
            )
            messages = [
                {"role": "system", "content": assessment_type._rubric_parse_prompt},
                {"role": "user", "content": raw_text},
            ]
            response = call_llm(caller, messages, temperature=0.0, top_p=1.0)
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
            parsed_rubric = _json.loads(cleaned)
        else:
            parsed_rubric = parse_rubric_with_llm(
                rubric_data["rubric_path"],
                caller,
                parse_type,
            )
        rubric_data["parsed_rubric"] = parsed_rubric
        logger.info("Rubric parsed. Sections found: %s", list(parsed_rubric.keys()))
    else:
        logger.warning(
            "No rubric loaded! rubric_id=%s, rubric_path=%s. "
            "Grading will use generic criteria.",
            rubric_id, rubric_data.get("rubric_path"),
        )

    # --- Faculty scores lookup ---
    # Support two patterns:
    # 1. faculty_scores: DataFrame with Student column (handoff type)
    # 2. faculty_scores_dict: dict keyed by student_id (documentation/ethics)
    faculty_lookup = {}
    faculty_is_dict = False

    faculty_dict = rubric_data.get("faculty_scores_dict")
    if faculty_dict is not None:
        faculty_lookup = faculty_dict
        faculty_is_dict = True
    else:
        faculty_df = rubric_data.get("faculty_scores")
        if faculty_df is not None:
            student_col = None
            for col in faculty_df.columns:
                if col.lower().strip() in ("student", "student id"):
                    student_col = col
                    break
            if student_col:
                for _, frow in faculty_df.iterrows():
                    sid = frow[student_col]
                    faculty_lookup[sid] = frow

    # --- Resume support ---
    if os.path.isfile(output_file):
        try:
            existing_df = pd.read_excel(output_file)
            # Check if it looks like a results file for this assessment type
            score_cols = [f"{s}_ai_score" for s in sections]
            if all(c in existing_df.columns for c in score_cols):
                logger.info("Found existing results file for resume check.")
        except Exception:
            pass

    total_rows = len(df)
    if total_rows == 0:
        logger.warning("No student data rows found. Nothing to grade.")
        return pd.DataFrame()

    effective_workers = min(max_workers, len(sections))
    results: list[GradingResult] = []

    # --- Find student ID column ---
    student_col = None
    for col in df.columns:
        if col.lower().strip() in ("student", "student id"):
            student_col = col
            break

    for idx, row in df.iterrows():
        current = len(results)
        if progress_callback:
            progress_callback(current, total_rows)

        # Determine student ID
        if student_col and pd.notna(row.get(student_col)):
            student_id = str(row[student_col])
        else:
            student_id = str(idx)

        logger.info("Processing student %s (%d/%d)...", student_id, current + 1, total_rows)

        section_scores: dict[str, float | None] = {}
        section_explanations: dict[str, str] = {}

        # Collect gradable sections
        gradable: list[tuple[str, str]] = []
        for section in sections:
            content = row.get(section)
            if pd.notna(content) and str(content).strip():
                gradable.append((section, str(content)))
            else:
                section_scores[section] = None
                section_explanations[section] = ""

        def _grade_one(sec: str, content: str) -> tuple[str, str, float | None]:
            system_prompt = assessment_type.build_grading_prompt(sec, rubric_data)
            user_msg = assessment_type.build_user_message(sec, content, rubric_data)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
            response = call_llm(caller, messages, temperature, top_p)
            log_interaction(log_file, messages, response)
            explanation, score = assessment_type.parse_llm_response(response, sec)
            return sec, explanation, score

        if effective_workers <= 1 or len(gradable) <= 1:
            for section, content in gradable:
                sec, explanation, score = _grade_one(section, content)
                section_scores[sec] = score
                section_explanations[sec] = explanation
        else:
            futures = {}
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                for section, content in gradable:
                    future = pool.submit(_grade_one, section, content)
                    futures[section] = future
            for section, future in futures.items():
                sec, explanation, score = future.result()
                section_scores[sec] = score
                section_explanations[sec] = explanation

        # Compute total
        scored_values = [v for v in section_scores.values() if v is not None]
        total_score = sum(scored_values) if scored_values else None

        # Milestone derivation (for handoff type)
        milestone = None
        if hasattr(assessment_type, '_derive_milestone_for_result'):
            result_tmp = GradingResult(
                student_id=student_id,
                section_scores=section_scores,
                total_score=total_score,
            )
            milestone = assessment_type._derive_milestone_for_result(result_tmp)

        # Faculty scores
        fac_scores = None
        try:
            sid_int = int(float(student_id))
        except (ValueError, TypeError):
            sid_int = None

        if sid_int is not None and sid_int in faculty_lookup:
            frow = faculty_lookup[sid_int]

            if faculty_is_dict:
                # Dict pattern (documentation/ethics): frow is already a dict
                fac_scores = dict(frow)
            else:
                # DataFrame row pattern (handoff): frow is a pandas Series
                fac_scores = {}
                for sec in sections:
                    fac_val = frow.get(sec)
                    if fac_val is not None and pd.notna(fac_val):
                        try:
                            fac_scores[sec] = float(fac_val)
                        except (ValueError, TypeError):
                            fac_scores[sec] = 0.0
                    else:
                        fac_scores[sec] = 0.0

                # Total and milestone from faculty
                total_col = None
                for col in faculty_lookup[sid_int].index:
                    if col.lower().strip() == "total":
                        total_col = col
                        break
                if total_col and pd.notna(frow.get(total_col)):
                    try:
                        fac_scores["total"] = float(frow[total_col])
                    except (ValueError, TypeError):
                        fac_scores["total"] = None
                else:
                    fac_scores["total"] = None

                milestone_col = None
                for col in faculty_lookup[sid_int].index:
                    if col.lower().strip() == "milestone":
                        milestone_col = col
                        break
                if milestone_col and pd.notna(frow.get(milestone_col)):
                    fac_scores["milestone"] = str(frow[milestone_col])

                comments_col = None
                for col in faculty_lookup[sid_int].index:
                    if col.lower().strip() == "comments":
                        comments_col = col
                        break
                if comments_col and pd.notna(frow.get(comments_col)):
                    fac_scores["comments"] = str(frow[comments_col])

        result = GradingResult(
            student_id=student_id,
            section_scores=section_scores,
            section_explanations=section_explanations,
            total_score=total_score,
            milestone=milestone,
            faculty_scores=fac_scores,
        )
        results.append(result)

        # Intermediate save
        out_df = assessment_type.build_output_df(results)
        _save_results(out_df, output_file)

    if progress_callback:
        progress_callback(total_rows, total_rows)

    out_df = assessment_type.build_output_df(results)
    _save_results(out_df, output_file)
    logger.info(
        "Grading completed. Results saved to %s. Log saved to %s.",
        output_file, log_file,
    )
    return out_df


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_excel_file_with_key(
    caller: LLMCaller,
    excel_file: str,
    rubric_content: dict[str, str],
    answer_key_content: dict[str, str],
    output_file: str,
    temperature: float,
    top_p: float,
    max_workers: int = 1,
) -> None:
    """Grade every student row in the Excel file and write results.

    When *max_workers* > 1 the sections within each student are graded
    concurrently using a thread pool, which can dramatically reduce wall-
    clock time (e.g. 4 sections in parallel ≈ 4× faster).
    """
    df = _read_excel_safe(excel_file, "student notes")
    log_file = os.path.splitext(output_file)[0] + ".log"
    sections = config.SECTIONS

    # --- Input validation ---
    missing = validate_input_columns(df, sections)
    if missing:
        logger.error(
            "The following expected columns are missing from '%s': %s\n"
            "Expected columns: %s",
            excel_file,
            ", ".join(missing),
            ", ".join(sections),
        )
        raise SystemExit(1)

    total_rows = len(df)
    if total_rows == 0:
        logger.warning(
            "Student notes file '%s' contains no data rows. Nothing to grade.",
            excel_file,
        )
        return

    effective_workers = min(max_workers, len(sections))
    if effective_workers > 1:
        logger.info(
            "Parallel mode: grading up to %d sections concurrently.",
            effective_workers,
        )

    # --- Resume support: check for existing output file ---
    if os.path.isfile(output_file):
        try:
            existing_df = pd.read_excel(output_file)
            if set(df.columns).issubset(set(existing_df.columns)):
                df = existing_df
                logger.info("Loaded existing results from '%s' for resume.", output_file)
        except Exception:
            pass  # fresh start if existing file can't be read

    for index, row in df.iterrows():
        # Skip rows already fully graded (resume capability)
        if _row_already_graded(row, sections):
            logger.info("Row %d/%d already graded, skipping.", index + 1, total_rows)
            continue

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            "%s - Processing row %d/%d...", current_time, index + 1, total_rows
        )

        # Collect sections that have content to grade
        gradable: list[tuple[str, str]] = []
        for section in sections:
            section_content = row[section]
            if pd.notna(section_content):
                gradable.append((section, str(section_content)))

        if effective_workers <= 1 or len(gradable) <= 1:
            # --- Sequential path (original behaviour) ---
            for section, content in gradable:
                explanation, numeric_score = grade_section_with_key(
                    caller, rubric_content, answer_key_content,
                    content, section, log_file, temperature, top_p,
                )
                df.at[index, f"{section}_gpt_explanation"] = explanation
                df.at[index, f"{section}_gpt_score"] = numeric_score
        else:
            # --- Parallel path: grade all sections concurrently ---
            futures: dict[str, any] = {}
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                for section, content in gradable:
                    future = pool.submit(
                        grade_section_with_key,
                        caller, rubric_content, answer_key_content,
                        content, section, log_file, temperature, top_p,
                    )
                    futures[section] = future

            # Collect results (all futures are done after exiting the 'with')
            for section, future in futures.items():
                explanation, numeric_score = future.result()
                df.at[index, f"{section}_gpt_explanation"] = explanation
                df.at[index, f"{section}_gpt_score"] = numeric_score

        # --- Intermediate save after each student row ---
        _save_results(df, output_file)

    logger.info(
        "Grading completed. Results saved to %s. Log saved to %s.",
        output_file,
        log_file,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    from audit import log_event
    log_event("app.start", stream="system", severity="info", detail={"surface": "cli"})
    parser = argparse.ArgumentParser(
        description="Grade OSCE post-encounter notes from an Excel file "
        "using an LLM with a rubric and answer key.",
    )
    parser.add_argument(
        "--rubric",
        type=str,
        default=config.DEFAULT_RUBRIC_PATH,
        help="Path to the rubric Excel file (default: %(default)s)",
    )
    parser.add_argument(
        "--answer_key",
        type=str,
        default=config.DEFAULT_ANSWER_KEY_PATH,
        help="Path to the answer key Excel file (default: %(default)s)",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=config.DEFAULT_NOTES_PATH,
        help="Path to the student notes Excel file (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=config.DEFAULT_OUTPUT_PATH,
        help="Path to save the graded output Excel file (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.TEMPERATURE,
        help="Temperature setting for the model, 0.0-2.0 (default: %(default)s)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=config.TOP_P,
        help="Top-p (nucleus sampling) setting, 0.0-1.0 (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.MAX_WORKERS,
        help=(
            "Number of sections to grade in parallel per student. "
            "Set to 1 for sequential processing (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=config.PROVIDER,
        choices=SUPPORTED_PROVIDERS,
        help="LLM provider to use (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model name to use. If not specified, uses config.py MODEL or "
            "the provider's default model."
        ),
    )
    args = parser.parse_args()

    # --- Configure console logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # --- Resolve model for the selected provider ---
    if args.model:
        model = args.model
    elif args.provider != config.PROVIDER:
        # User switched providers via CLI but didn't specify --model.
        # Use the provider's default model instead of the config.py default.
        model = config.DEFAULT_MODELS.get(args.provider, config.MODEL)
    else:
        model = config.MODEL

    # --- Validate parameter ranges ---
    max_temp = 1.0 if args.provider == "anthropic" else 2.0
    if not 0.0 <= args.temperature <= max_temp:
        logger.error(
            "Invalid --temperature value: %.2f. Must be between 0.0 and %.1f"
            " for %s.",
            args.temperature, max_temp, args.provider,
        )
        raise SystemExit(1)

    if not 0.0 <= args.top_p <= 1.0:
        logger.error(
            "Invalid --top_p value: %.2f. Must be between 0.0 and 1.0.",
            args.top_p,
        )
        raise SystemExit(1)

    # --- Validate input files exist before doing any work ---
    validate_files_exist(args.rubric, args.answer_key, args.notes)
    validate_output_directory(args.output)

    # --- Guard against overwriting the input file ---
    notes_abs = os.path.abspath(args.notes)
    output_abs = os.path.abspath(args.output)
    if notes_abs == output_abs:
        logger.error(
            "The --notes file and --output file resolve to the same path "
            "('%s'). This would overwrite the input data. "
            "Please specify a different output path.",
            notes_abs,
        )
        raise SystemExit(1)

    # --- Initialise the LLM caller ---
    caller = create_caller(args.provider)
    logger.info("Provider: %s | Model: %s", args.provider, model)

    rubric_content, answer_key_content = read_rubric_and_key(
        args.rubric, args.answer_key
    )

    from identity import cli_stub_user
    stub = cli_stub_user()
    import uuid
    ctx = RunContext(
        run_id=str(uuid.uuid4()),
        actor_email=stub.email,
        actor_role=stub.role,
        auth_session_id=stub.session_id,
        provider=args.provider,
        model=model,
        temperature=args.temperature,
        top_p=args.top_p,
        workers=args.workers,
        max_tokens=config.MAX_TOKENS,
        assessment_type=args.assessment_type if hasattr(args, "assessment_type") else "uk_osce",
        sections=list(config.SECTIONS),
    )

    process_excel_file_with_key(
        caller,
        args.notes,
        rubric_content,
        answer_key_content,
        args.output,
        args.temperature,
        args.top_p,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
