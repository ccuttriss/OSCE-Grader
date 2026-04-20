# tests/test_grading_run_persistence.py
import os
import sys
import tempfile
import sqlite3
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


@pytest.fixture
def temp_env(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        import database
        monkeypatch.setattr(database, "DB_PATH", os.path.join(td, "t.db"))
        database.init_db()
        monkeypatch.setenv("OSCE_STORAGE_DIR", os.path.join(td, "storage"))
        yield td


def test_begin_run_inserts_row(temp_env):
    import grading_runs as gr
    row_id = gr.begin_run(
        run_uuid="run-1",
        user_email="u@x",
        auth_session_id="sess-1",
        assessment_type_id="uk_osce",
        provider="openai",
        model="gpt-4o",
        temperature=0.3,
        top_p=1.0,
        workers=4,
        max_tokens=4096,
        sections=["hpi", "pex"],
        rubric_material_id=None,
        answer_key_material_id=None,
        student_notes_sha256="abc",
    )
    assert isinstance(row_id, int)


def test_complete_run_updates_row(temp_env):
    import grading_runs as gr
    row_id = gr.begin_run(
        run_uuid="run-2", user_email="u@x", auth_session_id="s",
        assessment_type_id="uk_osce", provider="openai", model="gpt-4o",
        temperature=0.3, top_p=1.0, workers=4, max_tokens=4096,
        sections=["hpi"], rubric_material_id=None, answer_key_material_id=None,
        student_notes_sha256="h",
    )
    gr.complete_run(row_id, results_sha256="resulthash", summary={"n": 10})
    import database
    conn = sqlite3.connect(database.DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM grading_runs WHERE run_id=?", (row_id,)).fetchone()
    assert row["status"] == "complete"
    assert row["results_sha256"] == "resulthash"
