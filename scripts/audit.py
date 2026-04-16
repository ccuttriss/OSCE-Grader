# scripts/audit.py
"""Audit subsystem.

Single table, two logical streams ('user', 'system'). log_event() is
non-blocking and MUST NOT raise — failures go to stderr.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

import database


@dataclass(frozen=True)
class AuditEvent:
    id: int
    ts: str
    stream: str
    severity: str
    action: str
    actor_email: str | None
    actor_role: str | None
    session_id: str | None
    request_id: str | None
    target_kind: str | None
    target_id: str | None
    target_hash: str | None
    outcome: str
    error_code: str | None
    detail_json: str | None


def _db_path() -> str:
    return database.DB_PATH


def log_event(
    action: str,
    *,
    stream: Literal["user", "system"],
    actor=None,
    severity: Literal["info", "warn", "error"] = "info",
    outcome: Literal["success", "failure", "denied"] = "success",
    target_kind: str | None = None,
    target_id: str | None = None,
    target_hash: str | None = None,
    error_code: str | None = None,
    detail: dict | None = None,
    request_id: str | None = None,
) -> None:
    """Non-blocking audit write. Never raises."""
    try:
        actor_email = getattr(actor, "email", None) if actor else None
        actor_role = getattr(actor, "role", None) if actor else None
        session_id = getattr(actor, "session_id", None) if actor else None
        detail_json = json.dumps(detail, default=str) if detail else None

        conn = sqlite3.connect(_db_path())
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(
                """
                INSERT INTO audit_events
                    (stream, severity, action, actor_email, actor_role,
                     session_id, request_id, target_kind, target_id, target_hash,
                     outcome, error_code, detail_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (stream, severity, action, actor_email, actor_role,
                 session_id, request_id, target_kind, target_id, target_hash,
                 outcome, error_code, detail_json),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        print(
            f"audit.write_failure action={action} stream={stream} error={exc!r}",
            file=sys.stderr,
        )


def query_events(
    *,
    stream: str | None = None,
    actor_email: str | None = None,
    action: str | None = None,
    target_kind: str | None = None,
    target_id: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    severity: str | None = None,
    limit: int = 1000,
    offset: int = 0,
) -> list[AuditEvent]:
    clauses = []
    params: list[Any] = []
    for col, val in [
        ("stream", stream), ("actor_email", actor_email),
        ("action", action), ("target_kind", target_kind),
        ("target_id", target_id), ("severity", severity),
    ]:
        if val is not None:
            clauses.append(f"{col} = ?")
            params.append(val)
    if since is not None:
        clauses.append("ts >= ?")
        params.append(since.strftime("%Y-%m-%d %H:%M:%S"))
    if until is not None:
        clauses.append("ts <= ?")
        params.append(until.strftime("%Y-%m-%d %H:%M:%S"))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT id, ts, stream, severity, action, actor_email, actor_role,
               session_id, request_id, target_kind, target_id, target_hash,
               outcome, error_code, detail_json
        FROM audit_events
        {where}
        ORDER BY id DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [AuditEvent(**dict(r)) for r in rows]


def retention_sweep(
    *,
    user_days: int,
    system_days: int,
    now: datetime | None = None,
) -> dict:
    now = now or datetime.utcnow()
    conn = sqlite3.connect(_db_path())
    try:
        deleted = {}
        for stream, days in [("user", user_days), ("system", system_days)]:
            cutoff = (now - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
            cur = conn.execute(
                "DELETE FROM audit_events WHERE stream = ? AND ts < ?",
                (stream, cutoff),
            )
            deleted[stream] = cur.rowcount
        conn.commit()
    finally:
        conn.close()
    log_event(
        "storage.retention.sweep",
        stream="system",
        severity="info",
        detail={"deleted": deleted, "user_days": user_days, "system_days": system_days},
    )
    return deleted
