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
        raise SystemExit(1)


def read_rubric_and_key(
    rubric_file: str, answer_key_file: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Load the rubric and answer key from Excel files.

    Each file is expected to have a header row with section names and at least
    one data row containing the rubric / answer-key text for each section.

    Raises SystemExit if either file is empty or cannot be read.
    """
    rubric_df = _read_excel_safe(rubric_file, "rubric")
    if rubric_df.empty:
        logger.error("Rubric file '%s' contains no data rows.", rubric_file)
        raise SystemExit(1)
    rubric: dict[str, str] = rubric_df.iloc[0].to_dict()

    answer_key_df = _read_excel_safe(answer_key_file, "answer key")
    if answer_key_df.empty:
        logger.error(
            "Answer-key file '%s' contains no data rows.", answer_key_file
        )
        raise SystemExit(1)
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
    delay = config.RETRY_DELAY
    last_exception: Optional[Exception] = None

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            return caller(messages, temperature, top_p)
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
                time.sleep(delay)
                delay *= 2  # exponential back-off
            else:
                logger.error(
                    "API call failed after %d attempts: %s",
                    config.MAX_RETRIES,
                    exc,
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

    for index, row in df.iterrows():
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
        config.MODEL = args.model
    elif args.provider != config.PROVIDER:
        # User switched providers via CLI but didn't specify --model.
        # Use the provider's default model instead of the config.py default.
        config.MODEL = config.DEFAULT_MODELS.get(args.provider, config.MODEL)

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
    logger.info("Provider: %s | Model: %s", args.provider, config.MODEL)

    rubric_content, answer_key_content = read_rubric_and_key(
        args.rubric, args.answer_key
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
