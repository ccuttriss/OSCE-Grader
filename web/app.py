"""OSCE Grader — Flask web application."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import threading

import pandas as pd
from flask import Flask, render_template, request, send_file
from markupsafe import escape
from werkzeug.utils import secure_filename

# Add scripts/ to sys.path for grader imports
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from grading_worker import (
    compute_summary_stats,
    convert_rubric_with_llm,
    run_dry_run,
    run_grading,
)
from jobs import job_manager

import config as grader_config

logger = logging.getLogger("osce_grader.web")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

# Directory for uploaded files and results
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _save_upload(file_storage, job_id: str, prefix: str) -> str:
    """Save an uploaded file to disk and return its path."""
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    safe_name = secure_filename(file_storage.filename) or "upload.xlsx"
    filename = f"{prefix}_{safe_name}"
    path = os.path.join(job_dir, filename)
    file_storage.save(path)
    return path


def _alert(level: str, message: str) -> str:
    """Return an escaped HTML alert div. Prevents XSS in error messages."""
    return f'<div class="alert alert-{escape(level)}">{escape(message)}</div>'


# -----------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("upload.html", active="grade",
                           default_temperature=grader_config.TEMPERATURE)


@app.route("/converter")
def converter():
    return render_template("converter.html", active="converter")


@app.route("/results/<job_id>")
def results_page(job_id):
    """Full-page results view (bookmarkable)."""
    job = job_manager.get(job_id)
    if not job:
        return render_template(
            "results.html", active="grade", error="Job not found or expired.",
            scores=None, stats=None
        )

    if job.status == "running":
        return render_template("upload.html", active="grade",
                               default_temperature=grader_config.TEMPERATURE)

    if job.status == "failed":
        return render_template(
            "results.html", active="grade", error=job.message,
            scores=None, stats=None
        )

    # Load results for display
    output_file = os.path.join(UPLOAD_DIR, job_id, f"results_{job_id}.xlsx")
    if not os.path.isfile(output_file):
        return render_template(
            "results.html", active="grade", error="Results file not found.",
            scores=None, stats=None
        )

    df = pd.read_excel(output_file)
    sections = grader_config.SECTIONS
    stats = compute_summary_stats(df, sections)
    scores = df.to_dict("records")

    return render_template(
        "results.html",
        active="grade",
        job_id=job_id,
        graded=len(df),
        stats=stats,
        scores=scores,
        sections=sections,
    )


# -----------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------

@app.route("/api/grade", methods=["POST"])
def api_grade():
    """Handle file uploads and start grading or dry-run."""
    action = request.form.get("action", "grade")

    # Validate uploads
    for field in ("rubric", "answer_key", "notes"):
        if field not in request.files or request.files[field].filename == "":
            return _alert("critical", f"Please upload all three files ({field} is missing).")

    # Create a job ID and save files
    job_id = job_manager.start("Uploading files...", total=0)

    try:
        rubric_path = _save_upload(request.files["rubric"], job_id, "rubric")
        answer_key_path = _save_upload(request.files["answer_key"], job_id, "key")
        notes_path = _save_upload(request.files["notes"], job_id, "notes")
    except Exception as e:
        job_manager.fail(job_id, str(e))
        return _alert("critical", f"Failed to save uploaded files: {e}")

    try:
        temperature = float(request.form.get("temperature", grader_config.TEMPERATURE))
    except (ValueError, TypeError):
        temperature = grader_config.TEMPERATURE

    if not 0.0 <= temperature <= 2.0:
        job_manager.fail(job_id, "Invalid temperature")
        return _alert("critical", "Temperature must be between 0.0 and 2.0.")

    # --- Dry Run (synchronous, fast) ---
    if action == "dry_run":
        try:
            info = run_dry_run(notes_path, rubric_path, answer_key_path)
            job_manager.complete(job_id, "")
            return render_template("_dry_run_result.html", **info)
        except (SystemExit, ValueError) as e:
            msg = str(e) if str(e) != "1" else "Failed to read input files. Check file format."
            return _alert("critical", msg)
        except Exception as e:
            return _alert("critical", f"Dry run failed: {e}")

    # --- Real Grading (background thread) ---
    output_dir = os.path.join(UPLOAD_DIR, job_id)

    # Count rows for progress bar
    try:
        df_temp = pd.read_excel(notes_path)
        total_rows = len(df_temp)
    except Exception:
        total_rows = 0

    job = job_manager.get(job_id)
    if job:
        job.total = total_rows
        job.message = f"Grading student 1 of {total_rows}..."

    def progress_callback(progress, total, message):
        job_manager.update_progress(job_id, progress, message)

    def grade_thread():
        try:
            result = run_grading(
                job_id=job_id,
                notes_path=notes_path,
                rubric_path=rubric_path,
                answer_key_path=answer_key_path,
                temperature=temperature,
                output_dir=output_dir,
                progress_callback=progress_callback,
            )

            # Build result HTML
            sections = grader_config.SECTIONS
            output_file = result["output_file"]
            df = pd.read_excel(output_file)
            scores = df.to_dict("records")

            with app.app_context():
                html = render_template(
                    "results_fragment.html",
                    job_id=job_id,
                    graded=result["graded"],
                    stats=result["stats"],
                    scores=scores,
                    sections=sections,
                )
            job_manager.complete(job_id, html)

        except (SystemExit, ValueError) as e:
            msg = str(e) if str(e) != "1" else "Grading failed. Check file format and columns."
            job_manager.fail(job_id, msg)
        except Exception as e:
            logger.exception("Grading failed for job %s", job_id)
            job_manager.fail(job_id, f"Grading failed: {e}")

    thread = threading.Thread(target=grade_thread, daemon=True)
    thread.start()

    # Return polling fragment immediately
    return render_template(
        "progress.html",
        job_id=job_id,
        message=f"Grading student 1 of {total_rows}...",
        progress=0,
        total=total_rows,
    )


@app.route("/api/jobs/<job_id>")
def api_job_status(job_id):
    """HTMX polling endpoint — returns progress or final result."""
    job = job_manager.get(job_id)
    if not job:
        return _alert("critical", "Job not found or expired.")

    if job.status == "running":
        return render_template(
            "progress.html",
            job_id=job_id,
            message=job.message,
            progress=job.progress,
            total=job.total,
        )

    if job.status == "failed":
        return _alert("critical", job.message)

    # Completed — return stored result HTML (no more hx-trigger = polling stops)
    return job.result_html or _alert("success", "Done!")


@app.route("/api/convert", methods=["POST"])
def api_convert():
    """Convert a PDF/DOCX rubric to structured Excel using the LLM."""
    if "rubric_file" not in request.files or request.files["rubric_file"].filename == "":
        return _alert("critical", "Please upload a rubric file.")

    upload = request.files["rubric_file"]
    safe_name = secure_filename(upload.filename) or "rubric.pdf"
    ext = os.path.splitext(safe_name)[-1].lower()

    if ext not in (".pdf", ".docx"):
        return _alert("critical", "Only .pdf and .docx files are supported.")

    # Save to temp location
    tmp_dir = tempfile.mkdtemp(prefix="osce_convert_")
    input_path = os.path.join(tmp_dir, safe_name)
    upload.save(input_path)

    # Step 1: Extract raw text from the document
    try:
        from convert_rubric import convert_pdf_to_text, convert_docx_to_text
        if ext == ".pdf":
            raw_text = convert_pdf_to_text(input_path)
        else:
            raw_text = convert_docx_to_text(input_path)
    except SystemExit:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return _alert("critical", "Failed to extract text from the file.")
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return _alert("critical", f"Text extraction failed: {e}")

    if not raw_text.strip():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return _alert("critical", "No text could be extracted from the file.")

    # Step 2: Use the LLM to parse into sections
    try:
        sections = convert_rubric_with_llm(raw_text)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return _alert("critical", f"LLM conversion failed: {e}")

    # Step 3: Build the structured Excel file
    output_name = os.path.splitext(safe_name)[0] + ".xlsx"
    output_path = os.path.join(tmp_dir, output_name)
    try:
        df = pd.DataFrame([sections])
        df.to_excel(output_path, index=False)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return _alert("critical", f"Failed to create Excel file: {e}")

    # Store the path so the download route can find it
    app.config.setdefault("CONVERT_FILES", {})[output_name] = {
        "path": output_path,
        "tmp_dir": tmp_dir,
    }

    # Build a preview of what was extracted
    found = [k.upper() for k, v in sections.items() if v.strip()]
    empty = [k.upper() for k, v in sections.items() if not v.strip()]

    return render_template("_convert_result.html",
                           output_name=output_name,
                           found=found,
                           empty=empty)


# -----------------------------------------------------------------------
# Download endpoints
# -----------------------------------------------------------------------

@app.route("/download/<job_id>/results")
def download_results(job_id):
    path = os.path.join(UPLOAD_DIR, job_id, f"results_{job_id}.xlsx")
    if not os.path.isfile(path):
        return _alert("critical", "Results file not found."), 404
    return send_file(path, as_attachment=True, download_name=f"osce_results_{job_id}.xlsx")


@app.route("/download/<job_id>/log")
def download_log(job_id):
    path = os.path.join(UPLOAD_DIR, job_id, f"results_{job_id}.log")
    if not os.path.isfile(path):
        return _alert("critical", "Log file not found."), 404
    return send_file(path, as_attachment=True, download_name=f"osce_log_{job_id}.log")


@app.route("/download/convert/<filename>")
def download_converted(filename):
    entries = app.config.get("CONVERT_FILES", {})
    entry = entries.get(filename)
    if not entry or not os.path.isfile(entry["path"]):
        return _alert("critical", "File not found."), 404
    path = entry["path"]
    tmp_dir = entry.get("tmp_dir")

    response = send_file(path, as_attachment=True, download_name=filename)

    # Clean up temp dir after download
    if tmp_dir:
        @response.call_on_close
        def cleanup():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            entries.pop(filename, None)

    return response
