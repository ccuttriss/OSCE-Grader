"""Background grading worker with progress tracking.

Replicates the loop from grader.process_excel_file_with_key() but injects
progress callbacks so the web UI can show a live progress bar.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import tempfile

import pandas as pd
from openai import OpenAI

# Add the scripts directory to sys.path so we can import grader and config
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import config
from grader import (
    _read_excel_safe,
    _row_already_graded,
    _save_results,
    grade_section_with_key,
    read_rubric_and_key,
    validate_input_columns,
)

logger = logging.getLogger("osce_grader.web")


def compute_summary_stats(df: pd.DataFrame, sections: list[str]) -> list[dict]:
    """Return per-section stats as a list of dicts for template rendering."""
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

    est_cost = api_calls * 0.001  # ~$0.001 per call for gpt-4o-mini

    return {
        "total_rows": total_rows,
        "api_calls": api_calls,
        "est_cost": f"${est_cost:.2f}",
        "model": config.MODEL,
        "sections": sections,
    }


def run_grading(
    job_id: str,
    notes_path: str,
    rubric_path: str,
    answer_key_path: str,
    temperature: float,
    output_dir: str,
    progress_callback=None,
) -> dict:
    """Run the full grading pipeline. Called from a background thread.

    Returns a dict with paths to output files and summary stats.
    """
    # Read inputs
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

    # Set up output paths
    output_file = os.path.join(output_dir, f"results_{job_id}.xlsx")
    log_file = os.path.join(output_dir, f"results_{job_id}.log")

    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)

    graded = 0
    for index, row in df.iterrows():
        if progress_callback:
            progress_callback(graded, total_rows, f"Grading student {graded + 1} of {total_rows}...")

        for section in sections:
            section_content = row[section]
            if pd.notna(section_content):
                explanation, numeric_score = grade_section_with_key(
                    client,
                    rubric_content,
                    answer_key_content,
                    section_content,
                    section,
                    log_file,
                    temperature,
                    config.TOP_P,
                )
                df.at[index, f"{section}_gpt_explanation"] = explanation
                df.at[index, f"{section}_gpt_score"] = numeric_score

        graded += 1
        # Intermediate save after each student
        _save_results(df, output_file)

    if progress_callback:
        progress_callback(total_rows, total_rows, "Grading complete!")

    stats = compute_summary_stats(df, sections)

    return {
        "output_file": output_file,
        "log_file": log_file,
        "total_rows": total_rows,
        "graded": graded,
        "stats": stats,
    }
