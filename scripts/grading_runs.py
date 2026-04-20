# scripts/grading_runs.py
"""Writes to the grading_runs table — start / complete / cancel / store results."""

from __future__ import annotations

import json
import os
import sqlite3

import database


def begin_run(
    *,
    run_uuid: str,
    user_email: str,
    auth_session_id: str,
    assessment_type_id: str,
    provider: str,
    model: str,
    temperature: float,
    top_p: float,
    workers: int,
    max_tokens: int,
    sections: list[str],
    rubric_material_id: int | None,
    answer_key_material_id: int | None,
    student_notes_sha256: str,
    rubric_id: int | None = None,
    source_type: str = "uploaded",
    session_id: int | None = None,
) -> int:
    conn = sqlite3.connect(database.DB_PATH)
    try:
        cur = conn.execute(
            """
            INSERT INTO grading_runs
                (assessment_type_id, rubric_id, model_used, temperature,
                 source_type, session_id, status,
                 run_uuid, user_email, auth_session_id, provider, top_p,
                 workers, max_tokens, sections_json,
                 rubric_material_id, answer_key_material_id, student_notes_sha256)
            VALUES (?, ?, ?, ?, ?, ?, 'running',
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (assessment_type_id, rubric_id, model, temperature,
             source_type, session_id,
             run_uuid, user_email, auth_session_id, provider, top_p,
             workers, max_tokens, json.dumps(sections),
             rubric_material_id, answer_key_material_id, student_notes_sha256),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def complete_run(row_id: int, *, results_sha256: str, summary: dict) -> None:
    conn = sqlite3.connect(database.DB_PATH)
    try:
        conn.execute(
            """UPDATE grading_runs
               SET status='complete', completed_at=datetime('now'),
                   results_sha256=?, summary_json=?
               WHERE run_id=?""",
            (results_sha256, json.dumps(summary, default=str), row_id),
        )
        conn.commit()
    finally:
        conn.close()


def cancel_run(row_id: int, *, reason: str | None = None) -> None:
    conn = sqlite3.connect(database.DB_PATH)
    try:
        conn.execute(
            """UPDATE grading_runs
               SET status='cancelled', completed_at=datetime('now'),
                   summary_json=?
               WHERE run_id=?""",
            (json.dumps({"cancel_reason": reason}), row_id),
        )
        conn.commit()
    finally:
        conn.close()


def store_results_file(data: bytes) -> str:
    import hashlib
    sha = hashlib.sha256(data).hexdigest()
    base = os.environ.get("OSCE_STORAGE_DIR") or os.path.join(
        os.path.dirname(database.DB_PATH), "storage"
    )
    path = os.path.join(base, "results", f"{sha}.xlsx")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
    return sha
