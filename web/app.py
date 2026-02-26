"""OSCE Grader — Flask web application."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading

import pandas as pd
from flask import Flask, render_template, request, send_file

# Add scripts/ to sys.path for grader imports
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from grading_worker import compute_summary_stats, run_dry_run, run_grading
from jobs import job_manager

import config as grader_config

logger = logging.getLogger("osce_grader.web")

app = Flask(__name__)

# Directory for uploaded files and results
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _save_upload(file_storage, job_id: str, prefix: str) -> str:
    """Save an uploaded file to disk and return its path."""
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    filename = f"{prefix}_{file_storage.filename}"
    path = os.path.join(job_dir, filename)
    file_storage.save(path)
    return path


# -----------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("upload.html", active="grade")


@app.route("/converter")
def converter():
    return render_template("converter.html", active="converter")


@app.route("/results/<job_id>")
def results_page(job_id):
    """Full-page results view (bookmarkable)."""
    job = job_manager.get(job_id)
    if not job:
        return render_template(
            "results.html", active="grade", error="Job not found or expired.", scores=None, stats=None
        )

    if job.status == "running":
        # Redirect to home with polling
        return render_template("upload.html", active="grade")

    if job.status == "failed":
        return render_template(
            "results.html", active="grade", error=job.message, scores=None, stats=None
        )

    # Load results for display
    output_file = os.path.join(UPLOAD_DIR, job_id, f"results_{job_id}.xlsx")
    if not os.path.isfile(output_file):
        return render_template(
            "results.html", active="grade", error="Results file not found.", scores=None, stats=None
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
            return f'<div class="alert alert-critical">Please upload all three files ({field} is missing).</div>'

    # Create a job ID and save files
    job_id = job_manager.start("Uploading files...", total=0)

    try:
        rubric_path = _save_upload(request.files["rubric"], job_id, "rubric")
        answer_key_path = _save_upload(request.files["answer_key"], job_id, "key")
        notes_path = _save_upload(request.files["notes"], job_id, "notes")
    except Exception as e:
        job_manager.fail(job_id, str(e))
        return f'<div class="alert alert-critical">Failed to save uploaded files: {e}</div>'

    temperature = float(request.form.get("temperature", 0.1))
    if not 0.0 <= temperature <= 2.0:
        job_manager.fail(job_id, "Invalid temperature")
        return '<div class="alert alert-critical">Temperature must be between 0.0 and 2.0.</div>'

    # --- Dry Run (synchronous, fast) ---
    if action == "dry_run":
        try:
            info = run_dry_run(notes_path, rubric_path, answer_key_path)
            job_manager.complete(job_id, "")
            return render_template("_dry_run_result.html", **info)
        except (SystemExit, ValueError) as e:
            msg = str(e) if str(e) != "1" else "Failed to read input files. Check file format."
            return f'<div class="alert alert-critical">{msg}</div>'
        except Exception as e:
            return f'<div class="alert alert-critical">Dry run failed: {e}</div>'

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
        return '<div class="alert alert-critical">Job not found or expired.</div>'

    if job.status == "running":
        return render_template(
            "progress.html",
            job_id=job_id,
            message=job.message,
            progress=job.progress,
            total=job.total,
        )

    if job.status == "failed":
        return f'<div class="alert alert-critical">{job.message}</div>'

    # Completed — return stored result HTML (no more hx-trigger = polling stops)
    return job.result_html or '<div class="alert alert-success">Done!</div>'


@app.route("/api/convert", methods=["POST"])
def api_convert():
    """Convert a PDF/DOCX rubric to Excel."""
    if "rubric_file" not in request.files or request.files["rubric_file"].filename == "":
        return '<div class="alert alert-critical">Please upload a rubric file.</div>'

    upload = request.files["rubric_file"]
    filename = upload.filename
    ext = os.path.splitext(filename)[-1].lower()

    if ext not in (".pdf", ".docx"):
        return '<div class="alert alert-critical">Only .pdf and .docx files are supported.</div>'

    # Save to temp location
    tmp_dir = tempfile.mkdtemp(prefix="osce_convert_")
    input_path = os.path.join(tmp_dir, filename)
    upload.save(input_path)

    output_name = os.path.splitext(filename)[0] + ".xlsx"
    output_path = os.path.join(tmp_dir, output_name)

    try:
        from convert_rubric import convert_rubric
        convert_rubric(input_path, output_path)
    except SystemExit:
        return '<div class="alert alert-critical">Conversion failed. Check the file format.</div>'
    except Exception as e:
        return f'<div class="alert alert-critical">Conversion failed: {e}</div>'

    # Store the path so the download route can find it
    app.config.setdefault("CONVERT_FILES", {})[output_name] = output_path

    return f'''<div class="alert alert-success">
  Conversion complete! The output contains raw text in a single column.
  You will need to restructure it into section columns (hpi, pex, sum, ddx, support, plan) before grading.
</div>
<div class="action-bar">
  <a href="/download/convert/{output_name}" role="button">Download {output_name}</a>
</div>'''


# -----------------------------------------------------------------------
# Download endpoints
# -----------------------------------------------------------------------

@app.route("/download/<job_id>/results")
def download_results(job_id):
    path = os.path.join(UPLOAD_DIR, job_id, f"results_{job_id}.xlsx")
    if not os.path.isfile(path):
        return '<div class="alert alert-critical">Results file not found.</div>', 404
    return send_file(path, as_attachment=True, download_name=f"osce_results_{job_id}.xlsx")


@app.route("/download/<job_id>/log")
def download_log(job_id):
    path = os.path.join(UPLOAD_DIR, job_id, f"results_{job_id}.log")
    if not os.path.isfile(path):
        return '<div class="alert alert-critical">Log file not found.</div>', 404
    return send_file(path, as_attachment=True, download_name=f"osce_log_{job_id}.log")


@app.route("/download/convert/<filename>")
def download_converted(filename):
    paths = app.config.get("CONVERT_FILES", {})
    path = paths.get(filename)
    if not path or not os.path.isfile(path):
        return '<div class="alert alert-critical">File not found.</div>', 404
    return send_file(path, as_attachment=True, download_name=filename)
