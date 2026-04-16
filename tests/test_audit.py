# tests/test_audit.py
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


@pytest.fixture
def temp_db(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "test.db")
        import database
        monkeypatch.setattr(database, "DB_PATH", db_path)
        database.init_db()
        yield db_path


def test_log_event_writes_row(temp_db):
    import audit
    audit.log_event(
        "app.start",
        stream="system",
        severity="info",
        detail={"version": "test"},
    )
    rows = audit.query_events(action="app.start")
    assert len(rows) == 1
    assert rows[0].stream == "system"
    assert rows[0].severity == "info"
    assert rows[0].outcome == "success"


def test_log_event_never_raises(temp_db, monkeypatch, capsys):
    import audit
    monkeypatch.setattr(audit, "_db_path", lambda: "/nonexistent/bad.db")
    audit.log_event("app.error", stream="system", severity="error")
    captured = capsys.readouterr()
    assert "audit.write_failure" in captured.err
