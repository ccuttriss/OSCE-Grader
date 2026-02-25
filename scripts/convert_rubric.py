"""Convert a rubric file from PDF or DOCX into Excel or CSV.

This is a **starting-point** utility.  It extracts raw text from the source
document and writes each line into a single-column spreadsheet.  The output
will almost certainly require manual restructuring (adding section-name column
headers, mapping content to the correct sections, etc.) before it can be used
as input to ``grader.py``.
"""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd
from docx import Document
from pdfminer.high_level import extract_text

logger = logging.getLogger("osce_grader.convert_rubric")


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def convert_pdf_to_text(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        return extract_text(pdf_path)
    except Exception as exc:
        logger.error("Failed to extract text from PDF '%s': %s", pdf_path, exc)
        raise SystemExit(1)


def convert_docx_to_text(docx_path: str) -> str:
    """Extract text from a DOCX file."""
    try:
        doc = Document(docx_path)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as exc:
        logger.error(
            "Failed to extract text from DOCX '%s': %s", docx_path, exc
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def save_text_to_excel(text: str, output_path: str) -> None:
    """Save extracted text lines into an Excel file (one row per line)."""
    df = pd.DataFrame({"Rubric": text.split("\n")})
    try:
        df.to_excel(output_path, index=False)
    except Exception as exc:
        logger.error("Failed to write Excel file '%s': %s", output_path, exc)
        raise SystemExit(1)


def save_text_to_csv(text: str, output_path: str) -> None:
    """Save extracted text lines into a CSV file (one row per line)."""
    df = pd.DataFrame({"Rubric": text.split("\n")})
    try:
        df.to_csv(output_path, index=False)
    except Exception as exc:
        logger.error("Failed to write CSV file '%s': %s", output_path, exc)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def convert_rubric(input_file: str, output_file: str) -> None:
    """Convert a rubric from PDF or DOCX to an Excel or CSV file.

    The output contains a single ``Rubric`` column with one row per line of
    extracted text.  You will need to manually restructure the output into the
    column format expected by ``grader.py`` (e.g. ``hpi``, ``pex``, ``sum``,
    ``ddx``, ``support``, ``plan``).
    """
    # --- Validate input file ---
    if not os.path.isfile(input_file):
        logger.error("Input file not found: '%s'", input_file)
        raise SystemExit(1)

    ext = os.path.splitext(input_file)[-1].lower()

    if ext == ".pdf":
        text = convert_pdf_to_text(input_file)
    elif ext == ".docx":
        text = convert_docx_to_text(input_file)
    else:
        logger.error(
            "Unsupported input format '%s'. Please provide a .pdf or .docx "
            "file.",
            ext,
        )
        raise SystemExit(1)

    # --- Validate output directory ---
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.isdir(output_dir):
        logger.error("Output directory does not exist: '%s'", output_dir)
        raise SystemExit(1)

    output_ext = os.path.splitext(output_file)[-1].lower()

    if output_ext == ".xlsx":
        save_text_to_excel(text, output_file)
    elif output_ext == ".csv":
        save_text_to_csv(text, output_file)
    else:
        logger.error(
            "Unsupported output format '%s'. Please specify .xlsx or .csv.",
            output_ext,
        )
        raise SystemExit(1)

    logger.info("Conversion successful! Output saved to %s", output_file)
    logger.info(
        "NOTE: The output contains raw extracted text in a single 'Rubric' "
        "column. You will need to manually restructure it into the section "
        "columns expected by grader.py (e.g. hpi, pex, sum, ddx, support, "
        "plan) before using it for grading."
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Convert a rubric from PDF or DOCX to Excel or CSV format. "
            "The output is a starting point and will require manual "
            "restructuring before it can be used with grader.py."
        ),
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input PDF or DOCX file.",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output Excel (.xlsx) or CSV (.csv) file.",
    )
    args = parser.parse_args()

    convert_rubric(args.input_file, args.output_file)
