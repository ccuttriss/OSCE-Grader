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


from datetime import datetime, timedelta


def test_retention_sweep_deletes_old_rows_by_stream(temp_db):
    import audit
    import sqlite3
    conn = sqlite3.connect(temp_db)
    old_ts = (datetime.utcnow() - timedelta(days=100)).strftime("%Y-%m-%d %H:%M:%S")
    fresh_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    conn.executemany(
        """INSERT INTO audit_events (ts, stream, severity, action, outcome)
           VALUES (?, 'system', 'info', 'llm.call', 'success')""",
        [(old_ts,), (fresh_ts,)],
    )
    conn.commit()
    conn.close()

    deleted = audit.retention_sweep(user_days=2557, system_days=90)
    assert deleted["system"] == 1
    remaining = audit.query_events(stream="system", action="llm.call")
    assert len(remaining) == 1
