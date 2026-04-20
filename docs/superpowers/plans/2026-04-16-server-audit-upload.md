# Server-readiness, audit, and source-material library — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move OSCE-Grader from a local-workstation Python tool to an internal managed server environment with auditability, shared source-material uploads, role-gated tabs, and a SAML-ready identity seam.

**Architecture:** Three new service modules (`identity`, `audit`, `material_library`), a `RunContext` dataclass replacing mutating `config` globals on the grading path, new SQLite tables (`audit_events`, `materials`, `material_tags`) and additive columns on `grading_runs`, content-addressed filesystem storage under `storage/materials/<sha256>/`, env-driven paths so one artifact runs on VM or container.

**Tech Stack:** Python 3.8+, Streamlit ≥1.30, pandas, SQLite (via stdlib `sqlite3`), pytest ≥7.0, existing LLM SDKs (openai, anthropic, google-genai).

**Spec:** `docs/superpowers/specs/2026-04-16-server-audit-upload-design.md`.

---

## Conventions for every task

- **Branch:** one feature branch for the whole plan (`feat/server-audit-upload`), one commit per task unless a task explicitly says otherwise.
- **Test runner:** `pytest`.
- **Working directory:** repo root (`/Users/christopher/Projects/OSCE-Grader` or the worktree equivalent).
- **CLI regression check:** when a task touches `grader.py`, run `pytest tests/test_grader.py -v` before committing.
- **Never skip hooks** with `--no-verify`.
- **Commit message prefix:** `feat:`, `test:`, `refactor:`, `docs:`, `chore:` as appropriate.

---

## File structure

**New files:**

| Path | Responsibility |
| --- | --- |
| `scripts/run_context.py` | `RunContext` + `User` placeholder import seam |
| `scripts/identity.py` | `User` dataclass, sign_in, sign_out, get_current_user, is_admin, SAML seam |
| `scripts/audit.py` | `AuditEvent`, `log_event`, `query_events`, `retention_sweep` |
| `scripts/material_library.py` | `Material` dataclass, save/list/get/open/archive, content-addressed storage |
| `scripts/server_env.py` | Env-var parsing (`OSCE_DATA_DIR`, paths, retention windows, server-mode flag) |
| `tests/test_run_context.py` | Unit tests for RunContext |
| `tests/test_identity.py` | Unit tests for identity |
| `tests/test_audit.py` | Unit tests for audit writer + sweep + query |
| `tests/test_material_library.py` | Unit tests for material library |
| `tests/test_grading_run_persistence.py` | Unit tests for grading_runs row writes + results storage |
| `tests/test_server_integration.py` | End-to-end integration test with mock LLM |
| `Dockerfile` | Container image |
| `.env.example` | Env var documentation |
| `docs/server_deployment.md` | VM + container deployment notes |

**Modified files:**

| Path | Change |
| --- | --- |
| `scripts/database.py` | Add `audit_events`, `materials`, `material_tags`; ALTER `grading_runs`; bump `CURRENT_SCHEMA_VERSION` |
| `scripts/grader.py` | Accept `RunContext`; remove `config.MODEL` mutation in `main()`; emit system-stream audit events around LLM calls |
| `app.py` | Sign-in gate, header, role-gated tab dispatcher, Source Materials tab, Audit Log tab, material pickers in Grade Notes |
| `requirements.txt` | (Add `freezegun>=1.2` for time-frozen tests) |
| `README.md` | Add server-mode section pointing at `docs/server_deployment.md` |

---

## Phase 1 — `RunContext` refactor

Kill the shared mutation of `config.PROVIDER` / `config.MODEL` on the grading path. `config.py` keeps default values; callers construct a `RunContext` per run.

### Task 1: Create failing test for `RunContext`

**Files:**
- Create: `tests/test_run_context.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_context.py
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from run_context import RunContext


def test_run_context_is_frozen():
    ctx = RunContext(
        run_id="abc",
        actor_email="cli_local",
        actor_role="admin",
        auth_session_id="sess-1",
        provider="openai",
        model="gpt-4o",
        temperature=0.3,
        top_p=1.0,
        workers=4,
        max_tokens=4096,
        assessment_type="uk_osce",
        sections=["hpi", "pex"],
    )
    with pytest.raises(Exception):
        ctx.provider = "anthropic"  # frozen dataclass must reject mutation


def test_run_context_required_fields():
    # Every field must be provided — no silent defaults on the grading path
    with pytest.raises(TypeError):
        RunContext(run_id="abc")  # type: ignore[call-arg]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_context.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_context'`.

- [ ] **Step 3: Commit the test**

```bash
git add tests/test_run_context.py
git commit -m "test: failing tests for RunContext dataclass"
```

### Task 2: Implement `RunContext`

**Files:**
- Create: `scripts/run_context.py`

- [ ] **Step 1: Write minimal implementation**

```python
# scripts/run_context.py
"""RunContext: per-run configuration passed explicitly through the grading path.

Replaces module-global mutation of `config.PROVIDER` / `config.MODEL` /
`config.TEMPERATURE` etc. that the old CLI and webapp used to drive the grader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RunContext:
    run_id: str
    actor_email: str           # "cli_local" for CLI runs, otherwise user's email
    actor_role: Literal["end_user", "admin"]
    auth_session_id: str       # identity session id; distinct from any synthetic FK
    provider: str
    model: str
    temperature: float
    top_p: float
    workers: int
    max_tokens: int
    assessment_type: str
    sections: list[str]
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_run_context.py -v`
Expected: PASS for both tests.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_context.py
git commit -m "feat: add RunContext dataclass for per-run grading config"
```

### Task 3: Thread `RunContext` through `process_assessment`

**Files:**
- Modify: `scripts/grader.py:366` (`process_assessment` signature)

Goal: add an optional `ctx: RunContext | None = None` parameter. When provided, `ctx.temperature`, `ctx.top_p`, and `ctx.workers` override the current positional args; later tasks will make `ctx` required.

- [ ] **Step 1: Add the parameter**

At `scripts/grader.py` line 366, change the signature:

```python
def process_assessment(
    assessment_type,
    caller: LLMCaller,
    file_paths: dict,
    output_file: str,
    temperature: float,
    top_p: float,
    max_workers: int = 4,
    progress_callback=None,
    *,
    ctx: "RunContext | None" = None,
) -> pd.DataFrame:
```

Then near the top of the function body (just after the docstring), add:

```python
    if ctx is not None:
        temperature = ctx.temperature
        top_p = ctx.top_p
        max_workers = ctx.workers
```

Also add an import near the top of `grader.py` (after existing imports):

```python
from run_context import RunContext  # noqa: F401  — re-exported for typing
```

- [ ] **Step 2: Run existing grader tests to confirm no regression**

Run: `pytest tests/test_grader.py -v`
Expected: PASS (existing behavior preserved when callers pass positional args).

- [ ] **Step 3: Commit**

```bash
git add scripts/grader.py
git commit -m "refactor: accept RunContext in process_assessment (opt-in)"
```

### Task 4: Stop mutating `config.MODEL` in CLI main

**Files:**
- Modify: `scripts/grader.py:784-905` (`main()`)

- [ ] **Step 1: Replace the mutation at lines 860–864**

Current code (verify around line 860 with `grep -n "config.MODEL = args.model" scripts/grader.py`):

```python
        config.MODEL = args.model
    elif args.provider != config.PROVIDER:
        # Use the provider's default model instead of the config.py default.
        config.MODEL = config.DEFAULT_MODELS.get(args.provider, config.MODEL)
```

Replace with a local `model` variable passed downstream:

```python
    if args.model:
        model = args.model
    elif args.provider != config.PROVIDER:
        model = config.DEFAULT_MODELS.get(args.provider, config.MODEL)
    else:
        model = config.MODEL
```

Update the logging statement (was at approx. line 901):

```python
    logger.info("Provider: %s | Model: %s", args.provider, model)
```

And every subsequent reference to `config.MODEL` in `main()` — change to the local `model`. Pass `model` into `create_caller(...)` and into any `RunContext`/cost lookup.

- [ ] **Step 2: Build and use a `RunContext` in `main()`**

Just before the `process_assessment` call in `main()`, add:

```python
    import uuid
    ctx = RunContext(
        run_id=str(uuid.uuid4()),
        actor_email="cli_local",
        actor_role="admin",
        auth_session_id=str(uuid.uuid4()),
        provider=args.provider,
        model=model,
        temperature=args.temperature,
        top_p=args.top_p,
        workers=args.workers,
        max_tokens=config.MAX_TOKENS,
        assessment_type=args.assessment_type if hasattr(args, "assessment_type") else "uk_osce",
        sections=list(config.SECTIONS),
    )
```

Then pass `ctx=ctx` to `process_assessment(...)` (keyword).

- [ ] **Step 3: Run CLI-path test**

Run: `pytest tests/test_grader.py -v`
Expected: PASS. If the test uses `config.MODEL` directly, it should still work because `config.MODEL` is no longer being mutated — the default value is preserved.

- [ ] **Step 4: Run a CLI smoke check**

Run (with a provider key available):
```bash
python scripts/grader.py --provider openai --dry_run 2>&1 | tail -5
```
Expected: the existing dry-run output. No new errors.

- [ ] **Step 5: Commit**

```bash
git add scripts/grader.py
git commit -m "refactor: remove config.MODEL mutation from CLI main; use RunContext"
```

### Task 5: Add `RunContext.from_streamlit_state()` placeholder factory

**Files:**
- Modify: `scripts/run_context.py`

Goal: give `app.py` a single constructor to call. Identity fields use placeholders today; Phase 3 replaces them.

- [ ] **Step 1: Add the factory**

```python
# append to scripts/run_context.py
import uuid


def run_context_from_streamlit(
    *,
    provider: str,
    model: str,
    temperature: float,
    top_p: float,
    workers: int,
    max_tokens: int,
    assessment_type: str,
    sections: list[str],
    actor_email: str = "unknown",
    actor_role: str = "end_user",
    auth_session_id: str | None = None,
) -> RunContext:
    return RunContext(
        run_id=str(uuid.uuid4()),
        actor_email=actor_email,
        actor_role=actor_role,
        auth_session_id=auth_session_id or str(uuid.uuid4()),
        provider=provider,
        model=model,
        temperature=temperature,
        top_p=top_p,
        workers=workers,
        max_tokens=max_tokens,
        assessment_type=assessment_type,
        sections=list(sections),
    )
```

- [ ] **Step 2: Test it**

Append to `tests/test_run_context.py`:

```python
from run_context import run_context_from_streamlit


def test_run_context_from_streamlit_fills_defaults():
    ctx = run_context_from_streamlit(
        provider="google",
        model="gemini-2.5-flash",
        temperature=0.3,
        top_p=1.0,
        workers=4,
        max_tokens=4096,
        assessment_type="uk_osce",
        sections=["hpi"],
    )
    assert ctx.actor_email == "unknown"
    assert ctx.actor_role == "end_user"
    assert len(ctx.run_id) >= 32
    assert ctx.auth_session_id != ctx.run_id
```

- [ ] **Step 3: Run test**

Run: `pytest tests/test_run_context.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_context.py tests/test_run_context.py
git commit -m "feat: RunContext streamlit factory with placeholder identity"
```

### Task 6: Replace `config` mutation in `app.py` with `RunContext`

**Files:**
- Modify: `app.py` (search for `config.PROVIDER =` and `config.MODEL =` — replace each assignment with local vars passed into `run_context_from_streamlit`)

- [ ] **Step 1: Locate mutation sites**

Run: `grep -n "config\.\(PROVIDER\|MODEL\|TEMPERATURE\|TOP_P\|MAX_WORKERS\)\s*=" app.py`

For each hit, convert `config.X = value` into a local variable `x = value` that flows into the `run_context_from_streamlit(...)` call for that run.

- [ ] **Step 2: Replace the grading call**

Wherever `app.py` currently calls `process_assessment(...)`, build a `RunContext` first:

```python
from run_context import run_context_from_streamlit

ctx = run_context_from_streamlit(
    provider=provider,         # from the Streamlit selector
    model=model,               # local, not config.MODEL
    temperature=temperature,
    top_p=top_p,
    workers=workers,
    max_tokens=config.MAX_TOKENS,
    assessment_type=assessment_type,
    sections=sections,
)
results = process_assessment(
    assessment_type_instance,
    caller,
    file_paths,
    output_file,
    temperature=temperature,
    top_p=top_p,
    max_workers=workers,
    progress_callback=cb,
    ctx=ctx,
)
```

- [ ] **Step 3: Manual smoke test**

Run: `streamlit run app.py` in one terminal; open the browser; run a grading job using the existing examples. Expected: results identical to pre-refactor.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "refactor: app.py builds RunContext per run; no config mutation"
```

---

## Phase 2 — Audit foundation

One table, two streams. Start by wiring system-stream events only (no user concept yet). Identity wiring comes in Phase 3.

### Task 7: Add `audit_events` table DDL

**Files:**
- Modify: `scripts/database.py:51-128` (the `_SCHEMA_SQL` string) and `scripts/database.py:24` (`CURRENT_SCHEMA_VERSION`)

- [ ] **Step 1: Add the table to `_SCHEMA_SQL`**

Append to the `_SCHEMA_SQL` string (just before the closing `"""`):

```sql
CREATE TABLE IF NOT EXISTS audit_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL DEFAULT (datetime('now')),
    stream          TEXT    NOT NULL CHECK (stream IN ('user','system')),
    severity        TEXT    NOT NULL CHECK (severity IN ('info','warn','error')),
    action          TEXT    NOT NULL,
    actor_email     TEXT,
    actor_role      TEXT,
    session_id      TEXT,
    request_id      TEXT,
    target_kind     TEXT,
    target_id       TEXT,
    target_hash     TEXT,
    outcome         TEXT    NOT NULL CHECK (outcome IN ('success','failure','denied')),
    error_code      TEXT,
    detail_json     TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts            ON audit_events(ts);
CREATE INDEX IF NOT EXISTS idx_audit_actor         ON audit_events(actor_email, ts);
CREATE INDEX IF NOT EXISTS idx_audit_stream_action ON audit_events(stream, action, ts);
CREATE INDEX IF NOT EXISTS idx_audit_target        ON audit_events(target_kind, target_id);
```

- [ ] **Step 2: Bump `CURRENT_SCHEMA_VERSION`**

At `scripts/database.py:24`, change `CURRENT_SCHEMA_VERSION = 1` to `CURRENT_SCHEMA_VERSION = 2`.

- [ ] **Step 3: Run the existing DB tests**

Run: `pytest tests/ -k "database or loader" -v`
Expected: PASS (schema-additive changes don't break existing tables).

- [ ] **Step 4: Commit**

```bash
git add scripts/database.py
git commit -m "feat: add audit_events table to schema; bump to v2"
```

### Task 8: Failing test for `audit.log_event`

**Files:**
- Create: `tests/test_audit.py`

- [ ] **Step 1: Write the test**

```python
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
    # Force a DB error by pointing at an invalid path
    monkeypatch.setattr(audit, "_db_path", lambda: "/nonexistent/bad.db")
    audit.log_event("app.error", stream="system", severity="error")
    captured = capsys.readouterr()
    assert "audit.write_failure" in captured.err
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_audit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'audit'`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_audit.py
git commit -m "test: failing tests for audit.log_event and never-raise guarantee"
```

### Task 9: Implement `audit.py` minimum viable

**Files:**
- Create: `scripts/audit.py`

- [ ] **Step 1: Write the module**

```python
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
    actor=None,   # identity.User | None — avoids import cycle by duck-typing
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
    except Exception as exc:  # pragma: no cover  (written to stderr in tests)
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
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_audit.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add scripts/audit.py
git commit -m "feat: add audit.py — log_event, query_events, retention_sweep"
```

### Task 10: Test and implement retention sweep

**Files:**
- Modify: `tests/test_audit.py`

- [ ] **Step 1: Add the test**

```python
# tests/test_audit.py  (append)
from datetime import datetime, timedelta


def test_retention_sweep_deletes_old_rows_by_stream(temp_db):
    import audit
    import sqlite3
    # Seed: one old system row (100d old), one fresh system row
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
    assert len(remaining) == 1  # only the fresh row
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_audit.py::test_retention_sweep_deletes_old_rows_by_stream -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_audit.py
git commit -m "test: retention_sweep deletes rows past the per-stream horizon"
```

### Task 11: Emit `app.start` on Streamlit boot and CLI main

**Files:**
- Modify: `app.py` (near the top, right after `st.set_page_config(...)`)
- Modify: `scripts/grader.py` (top of `main()`)

- [ ] **Step 1: Add to `app.py`**

Right after the `st.set_page_config(...)` call:

```python
from audit import log_event
log_event("app.start", stream="system", severity="info", detail={"surface": "streamlit"})
```

- [ ] **Step 2: Add to `grader.py:main()`**

At the top of `main()`:

```python
from audit import log_event
log_event("app.start", stream="system", severity="info", detail={"surface": "cli"})
```

- [ ] **Step 3: Manual check**

Run: `python scripts/grader.py --help` (triggers argparse exit before `main()` body executes only if help is printed — use a real dry-run instead):
```bash
python scripts/grader.py --provider openai --dry_run
sqlite3 osce_grader.db "SELECT action, detail_json FROM audit_events WHERE action='app.start' ORDER BY id DESC LIMIT 1;"
```
Expected: one row with `surface=cli`.

- [ ] **Step 4: Commit**

```bash
git add app.py scripts/grader.py
git commit -m "feat: emit app.start audit event on app/CLI boot"
```

### Task 12: Wrap LLM calls with `llm.call` / `llm.retry` / `llm.failure` events

**Files:**
- Modify: `scripts/grader.py:111-152` (the `call_llm` function)

- [ ] **Step 1: Instrument `call_llm`**

Add audit events at the appropriate places in `call_llm`. Current shape (lines ~111–152) has a retry loop with `config.MAX_RETRIES` attempts. Wrap each attempt with timing, and on retry/failure log the event.

```python
# inside call_llm, replace the retry loop body with:
import time
from audit import log_event

for attempt in range(1, config.MAX_RETRIES + 1):
    t0 = time.time()
    try:
        response = caller(messages, temperature=temperature, top_p=top_p)
        log_event(
            "llm.call",
            stream="system",
            severity="info",
            detail={
                "provider": getattr(caller, "provider", "unknown"),
                "model": getattr(caller, "model", "unknown"),
                "latency_ms": int((time.time() - t0) * 1000),
                "attempt": attempt,
            },
        )
        return response
    except Exception as exc:
        if attempt < config.MAX_RETRIES:
            log_event(
                "llm.retry",
                stream="system",
                severity="warn",
                outcome="failure",
                error_code=type(exc).__name__,
                detail={"attempt": attempt, "error": str(exc)[:200]},
            )
            # existing backoff code stays here
            time.sleep(delay)
            delay *= 2
        else:
            log_event(
                "llm.failure",
                stream="system",
                severity="error",
                outcome="failure",
                error_code=type(exc).__name__,
                detail={"attempt": attempt, "error": str(exc)[:200]},
            )
            raise
```

Important: preserve all existing behavior (backoff delay doubling, log messages). The audit calls are additive.

- [ ] **Step 2: Run existing grader tests**

Run: `pytest tests/test_grader.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add scripts/grader.py
git commit -m "feat: emit llm.call/retry/failure system-stream audit events"
```

### Task 13: Top-level Streamlit exception handler → `app.error`

**Files:**
- Modify: `app.py` (wrap the bottom-level tab dispatch in try/except)

- [ ] **Step 1: Wrap**

At the bottom of `app.py`, wrap the `with tab1:` … `with tab6:` block (or whatever the final dispatch is) in a try/except that logs `app.error` on any uncaught exception and re-raises so Streamlit still shows the default error UI:

```python
import hashlib
import traceback
try:
    with tab1: tab_grade_notes()
    with tab2: tab_analysis()
    with tab3: tab_flagged()
    with tab4: tab_convert()
    with tab5: tab_gold_standard()
    with tab6: tab_synthetic_generator()
except Exception as exc:
    stack = traceback.format_exc()
    stack_hash = hashlib.sha256(stack.encode()).hexdigest()[:16]
    log_event(
        "app.error",
        stream="system",
        severity="error",
        outcome="failure",
        error_code=type(exc).__name__,
        detail={"stack_hash": stack_hash, "message": str(exc)[:200]},
    )
    raise
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: catch uncaught tab exceptions and emit app.error audit event"
```

---

## Phase 3 — Identity, sign-in, role gating

### Task 14: Failing test for `identity`

**Files:**
- Create: `tests/test_identity.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_identity.py
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


class FakeSessionState(dict):
    def __getattr__(self, k): return self[k]
    def __setattr__(self, k, v): self[k] = v


@pytest.fixture
def fake_streamlit(monkeypatch):
    import types
    fake_st = types.SimpleNamespace()
    fake_st.session_state = FakeSessionState()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    return fake_st


def test_sign_in_creates_user_and_session(fake_streamlit):
    import identity
    user = identity.sign_in("faculty@example.edu")
    assert user.email == "faculty@example.edu"
    assert user.role == "end_user"
    assert user.session_id
    assert identity.get_current_user() == user


def test_sign_in_rejects_bad_email(fake_streamlit):
    import identity
    with pytest.raises(ValueError):
        identity.sign_in("not-an-email")


def test_is_admin_matches_allowlist(fake_streamlit, monkeypatch):
    import identity
    monkeypatch.setenv("OSCE_ADMIN_EMAILS", "admin@example.edu,chris@osce.edu")
    user = identity.sign_in("admin@example.edu")
    assert identity.is_admin(user) is True
    user2 = identity.sign_in("faculty@example.edu")
    assert identity.is_admin(user2) is False


def test_sign_out_clears_session(fake_streamlit):
    import identity
    identity.sign_in("user@example.edu")
    identity.sign_out()
    assert identity.get_current_user() is None


def test_cli_stub_user_is_admin():
    import identity
    user = identity.cli_stub_user()
    assert user.email == "cli_local"
    assert user.role == "admin"
```

- [ ] **Step 2: Run test — expect fail**

Run: `pytest tests/test_identity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'identity'`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_identity.py
git commit -m "test: failing tests for identity module"
```

### Task 15: Implement `identity.py`

**Files:**
- Create: `scripts/identity.py`

- [ ] **Step 1: Write the module**

```python
# scripts/identity.py
"""Identity service.

Email-only sign-in today; SAML adapter replaces sign_in() later.
get_current_user() and is_admin() are the stable callers' interface.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from typing import Literal

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SESSION_KEY = "_osce_user"


@dataclass(frozen=True)
class User:
    email: str
    role: Literal["end_user", "admin"]
    session_id: str


def _admin_emails() -> set[str]:
    raw = os.environ.get("OSCE_ADMIN_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def _streamlit():
    import streamlit  # imported lazily so CLI doesn't need it
    return streamlit


def sign_in(email: str) -> User:
    email = email.strip().lower()
    if not _EMAIL_RE.match(email):
        raise ValueError("invalid email format")
    role: Literal["end_user", "admin"] = "admin" if email in _admin_emails() else "end_user"
    user = User(email=email, role=role, session_id=str(uuid.uuid4()))
    _streamlit().session_state[_SESSION_KEY] = user
    from audit import log_event
    log_event("sign_in", stream="user", actor=user)
    return user


def sign_out() -> None:
    st = _streamlit()
    user = st.session_state.get(_SESSION_KEY)
    if user is None:
        return
    from audit import log_event
    log_event("sign_out", stream="user", actor=user)
    del st.session_state[_SESSION_KEY]


def get_current_user() -> User | None:
    return _streamlit().session_state.get(_SESSION_KEY)


def is_admin(user: User | None) -> bool:
    if user is None:
        return False
    return user.role == "admin"


def cli_stub_user() -> User:
    """User stamped on audit rows emitted from CLI runs."""
    return User(email="cli_local", role="admin", session_id=str(uuid.uuid4()))
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_identity.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add scripts/identity.py
git commit -m "feat: add identity service with email sign-in and admin allowlist"
```

### Task 16: Render sign-in page in `app.py`

**Files:**
- Modify: `app.py` (insert near line 2524, before `st.title("OSCE Grader")`)

- [ ] **Step 1: Add sign-in rendering**

Replace the block starting at `st.title("OSCE Grader")` and the tab creation with:

```python
import identity

user = identity.get_current_user()
if user is None:
    st.title("OSCE Grader")
    st.caption("AI-powered grading for medical student post-encounter notes")
    with st.form("sign_in", clear_on_submit=False):
        email = st.text_input("Institutional email", placeholder="you@institution.edu")
        submitted = st.form_submit_button("Continue")
    if submitted:
        try:
            identity.sign_in(email)
            st.rerun()
        except ValueError:
            st.error("Please enter a valid email address.")
    st.caption("Your email is recorded in the audit log for this session only.")
    st.stop()

# Header
hdr_left, hdr_right = st.columns([3, 1])
with hdr_left:
    st.title("OSCE Grader")
    st.caption("AI-powered grading for medical student post-encounter notes")
with hdr_right:
    st.markdown(f"**{user.email}**  \n_role: {user.role}_")
    if st.button("Sign out"):
        identity.sign_out()
        st.rerun()
```

- [ ] **Step 2: Manual smoke**

Run: `streamlit run app.py`.
Expected: sign-in form shown first; entering a valid email reveals the tabs; clicking "Sign out" returns to the form.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: sign-in gate and header with user + sign-out in app.py"
```

### Task 17: Role-gated tab dispatcher

**Files:**
- Modify: `app.py` (the `st.tabs([...])` block near line 2527)

- [ ] **Step 1: Replace the tab list**

```python
ALL_TABS = [
    ("Grade Notes",        tab_grade_notes,          "end_user"),
    ("Analysis Dashboard", tab_analysis,             "end_user"),
    ("Flagged Items",      tab_flagged,              "end_user"),
    # Source Materials tab handler arrives in Phase 4 — stubbed below
    ("Source Materials",   lambda: st.info("Source Materials — coming in Phase 4"), "end_user"),
    ("Convert Rubric",     tab_convert,              "admin"),
    ("Gold Standard",      tab_gold_standard,        "admin"),
    ("Synthetic Data",     tab_synthetic_generator,  "admin"),
    # Audit Log tab arrives in Phase 6
    ("Audit Log",          lambda: st.info("Audit Log — coming in Phase 6"),        "admin"),
]

visible = [t for t in ALL_TABS if t[2] == "end_user" or identity.is_admin(user)]
tab_objects = st.tabs([t[0] for t in visible])
for tab_obj, (_name, handler, _role) in zip(tab_objects, visible):
    with tab_obj:
        handler()
```

- [ ] **Step 2: Manual check**

Run: `streamlit run app.py`.
- Sign in with a non-admin email → see 4 tabs.
- Set `OSCE_ADMIN_EMAILS=you@example.edu`, sign in with that → see all 8 tabs.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: role-gated tab dispatcher with admin allowlist"
```

### Task 18: Defense-in-depth guard inside each admin tab handler

**Files:**
- Modify: `app.py` (top of `tab_convert`, `tab_gold_standard`, `tab_synthetic_generator`, and the Audit Log handler once it exists)

- [ ] **Step 1: Add guard**

At the very top of each admin tab handler function:

```python
def tab_convert():
    user = identity.get_current_user()
    if not identity.is_admin(user):
        import audit
        audit.log_event(
            "tab.access.denied",
            stream="user",
            actor=user,
            severity="warn",
            outcome="denied",
            target_kind="tab",
            target_id="convert",
        )
        st.error("This tab requires admin privileges.")
        return
    # ... existing body
```

Repeat for `tab_gold_standard` (target_id="gold_standard") and `tab_synthetic_generator` (target_id="synthetic").

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: defense-in-depth is_admin guards in admin tab handlers"
```

### Task 19: Thread the signed-in user into `RunContext`

**Files:**
- Modify: `app.py` (wherever `run_context_from_streamlit(...)` is called from Task 6)

- [ ] **Step 1: Replace placeholder actor fields**

```python
user = identity.get_current_user()
ctx = run_context_from_streamlit(
    provider=provider,
    model=model,
    temperature=temperature,
    top_p=top_p,
    workers=workers,
    max_tokens=config.MAX_TOKENS,
    assessment_type=assessment_type,
    sections=sections,
    actor_email=user.email,
    actor_role=user.role,
    auth_session_id=user.session_id,
)
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: stamp RunContext with the signed-in user's identity"
```

### Task 20: CLI stub user in `grader.py:main()`

**Files:**
- Modify: `scripts/grader.py:main()` (the `RunContext` construction from Task 4)

- [ ] **Step 1: Use `identity.cli_stub_user()`**

Replace the hand-rolled `actor_email="cli_local"` in Task 4's `RunContext(...)` call with:

```python
from identity import cli_stub_user
stub = cli_stub_user()
ctx = RunContext(
    run_id=str(uuid.uuid4()),
    actor_email=stub.email,
    actor_role=stub.role,
    auth_session_id=stub.session_id,
    ...
)
```

- [ ] **Step 2: Commit**

```bash
git add scripts/grader.py
git commit -m "refactor: CLI grader uses identity.cli_stub_user for actor fields"
```

---

## Phase 4 — Source material library

### Task 21: Add `materials` and `material_tags` table DDL

**Files:**
- Modify: `scripts/database.py` (`_SCHEMA_SQL` + `CURRENT_SCHEMA_VERSION`)

- [ ] **Step 1: Append tables**

Append to `_SCHEMA_SQL`:

```sql
CREATE TABLE IF NOT EXISTS materials (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kind            TEXT    NOT NULL CHECK (kind IN (
                        'rubric','answer_key','student_notes','exemplar')),
    display_name    TEXT    NOT NULL,
    filename        TEXT    NOT NULL,
    content_sha256  TEXT    NOT NULL,
    size_bytes      INTEGER NOT NULL,
    mime_type       TEXT,
    assessment_type TEXT,
    uploaded_by     TEXT    NOT NULL,
    uploaded_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    archived_at     TEXT,
    notes           TEXT,
    UNIQUE (content_sha256, kind)
);
CREATE INDEX IF NOT EXISTS idx_materials_kind     ON materials(kind);
CREATE INDEX IF NOT EXISTS idx_materials_assess   ON materials(assessment_type);
CREATE INDEX IF NOT EXISTS idx_materials_archived ON materials(archived_at);

CREATE TABLE IF NOT EXISTS material_tags (
    material_id INTEGER NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
    tag         TEXT    NOT NULL,
    PRIMARY KEY (material_id, tag)
);
```

- [ ] **Step 2: Bump schema version**

Change `CURRENT_SCHEMA_VERSION = 2` → `3`.

- [ ] **Step 3: Commit**

```bash
git add scripts/database.py
git commit -m "feat: add materials and material_tags tables; schema v3"
```

### Task 22: Failing tests for `material_library`

**Files:**
- Create: `tests/test_material_library.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_material_library.py
import io
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


@pytest.fixture
def temp_env(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        import database
        monkeypatch.setattr(database, "DB_PATH", os.path.join(td, "test.db"))
        database.init_db()
        monkeypatch.setenv("OSCE_STORAGE_DIR", os.path.join(td, "storage"))
        yield td


def test_save_material_dedupes_identical_content(temp_env):
    import material_library as ml
    data = b"rubric body"
    m1 = ml.save_material(
        "rubric",
        file=io.BytesIO(data), filename="r.xlsx",
        display_name="R1", assessment_type="uk_osce",
        uploaded_by="u@x.edu",
    )
    m2 = ml.save_material(
        "rubric",
        file=io.BytesIO(data), filename="r.xlsx",
        display_name="R2",  # different label, same bytes
        assessment_type="uk_osce",
        uploaded_by="u2@x.edu",
    )
    assert m1.content_sha256 == m2.content_sha256
    # Same (sha, kind) pair is UNIQUE → save_material returns the existing row
    assert m1.id == m2.id


def test_list_materials_filters(temp_env):
    import material_library as ml
    ml.save_material(
        "rubric", file=io.BytesIO(b"a"), filename="a.xlsx",
        display_name="A", assessment_type="uk_osce", uploaded_by="u@x",
    )
    ml.save_material(
        "answer_key", file=io.BytesIO(b"b"), filename="b.xlsx",
        display_name="B", assessment_type="kpsom_osce", uploaded_by="u@x",
    )
    assert len(ml.list_materials(kind="rubric")) == 1
    assert len(ml.list_materials(assessment_type="kpsom_osce")) == 1


def test_open_material_returns_bytes(temp_env):
    import material_library as ml
    m = ml.save_material(
        "rubric", file=io.BytesIO(b"hello"), filename="r.xlsx",
        display_name="R", assessment_type="uk_osce", uploaded_by="u@x",
    )
    with ml.open_material(m) as f:
        assert f.read() == b"hello"


def test_archive_material_soft_deletes(temp_env):
    import material_library as ml
    from identity import User
    m = ml.save_material(
        "rubric", file=io.BytesIO(b"x"), filename="r.xlsx",
        display_name="R", assessment_type="uk_osce", uploaded_by="u@x",
    )
    ml.archive_material(m.id, by=User(email="a@x", role="admin", session_id="s"))
    assert len(ml.list_materials(kind="rubric")) == 0
    assert len(ml.list_materials(kind="rubric", include_archived=True)) == 1
```

- [ ] **Step 2: Run tests — expect fail**

Run: `pytest tests/test_material_library.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'material_library'`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_material_library.py
git commit -m "test: failing tests for material_library (save/list/open/archive)"
```

### Task 23: Implement `material_library.py`

**Files:**
- Create: `scripts/material_library.py`

- [ ] **Step 1: Write the module**

```python
# scripts/material_library.py
"""Content-addressed source-material library.

Files are hashed and sharded on disk; SQLite tracks metadata and tags.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import sqlite3
from dataclasses import dataclass
from typing import BinaryIO, Literal

import database


@dataclass(frozen=True)
class Material:
    id: int
    kind: str
    display_name: str
    filename: str
    content_sha256: str
    size_bytes: int
    mime_type: str | None
    assessment_type: str | None
    uploaded_by: str
    uploaded_at: str
    archived_at: str | None
    notes: str | None


def _storage_dir() -> str:
    base = os.environ.get("OSCE_STORAGE_DIR")
    if not base:
        base = os.path.join(os.path.dirname(database.DB_PATH), "storage")
    return os.path.join(base, "materials")


def _path_for(sha: str, ext: str) -> str:
    return os.path.join(_storage_dir(), sha[:2], sha[2:4], f"{sha}{ext}")


def _ext_from(filename: str) -> str:
    _, ext = os.path.splitext(filename)
    return ext.lower()


def save_material(
    kind: Literal["rubric", "answer_key", "student_notes", "exemplar"],
    *,
    file: BinaryIO,
    filename: str,
    display_name: str,
    assessment_type: str | None = None,
    tags: list[str] | None = None,
    uploaded_by: str,
    notes: str | None = None,
) -> Material:
    data = file.read()
    sha = hashlib.sha256(data).hexdigest()
    size = len(data)
    mime, _ = mimetypes.guess_type(filename)

    # Persist file if not already there
    path = _path_for(sha, _ext_from(filename))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)

    conn = sqlite3.connect(database.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        existing = conn.execute(
            "SELECT * FROM materials WHERE content_sha256=? AND kind=?",
            (sha, kind),
        ).fetchone()
        if existing:
            material = _row_to_material(existing)
        else:
            cur = conn.execute(
                """
                INSERT INTO materials
                    (kind, display_name, filename, content_sha256, size_bytes,
                     mime_type, assessment_type, uploaded_by, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (kind, display_name, filename, sha, size, mime,
                 assessment_type, uploaded_by, notes),
            )
            mid = cur.lastrowid
            for t in (tags or []):
                conn.execute(
                    "INSERT OR IGNORE INTO material_tags (material_id, tag) VALUES (?, ?)",
                    (mid, t),
                )
            conn.commit()
            row = conn.execute("SELECT * FROM materials WHERE id=?", (mid,)).fetchone()
            material = _row_to_material(row)
    finally:
        conn.close()

    from audit import log_event
    log_event(
        "material.upload",
        stream="user",
        actor=None,  # caller passes actor via audit if needed; v1: only email is on the row
        detail={
            "kind": kind, "size_bytes": size,
            "assessment_type": assessment_type, "uploaded_by": uploaded_by,
        },
        target_kind="material", target_id=str(material.id), target_hash=sha,
    )
    return material


def list_materials(
    *,
    kind: str | None = None,
    assessment_type: str | None = None,
    tag: str | None = None,
    include_archived: bool = False,
) -> list[Material]:
    clauses = []
    params: list = []
    if kind is not None:
        clauses.append("m.kind = ?")
        params.append(kind)
    if assessment_type is not None:
        clauses.append("m.assessment_type = ?")
        params.append(assessment_type)
    if not include_archived:
        clauses.append("m.archived_at IS NULL")
    join = ""
    if tag is not None:
        join = "JOIN material_tags t ON t.material_id = m.id"
        clauses.append("t.tag = ?")
        params.append(tag)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT m.* FROM materials m {join} {where} ORDER BY m.uploaded_at DESC"
    conn = sqlite3.connect(database.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [_row_to_material(r) for r in rows]


def get_material(
    material_id: int | None = None, *, sha256: str | None = None,
) -> Material | None:
    conn = sqlite3.connect(database.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        if material_id is not None:
            row = conn.execute(
                "SELECT * FROM materials WHERE id=?", (material_id,)
            ).fetchone()
        elif sha256 is not None:
            row = conn.execute(
                "SELECT * FROM materials WHERE content_sha256=?", (sha256,)
            ).fetchone()
        else:
            raise ValueError("must pass material_id or sha256")
    finally:
        conn.close()
    return _row_to_material(row) if row else None


def open_material(material: Material) -> BinaryIO:
    path = _path_for(material.content_sha256, _ext_from(material.filename))
    return open(path, "rb")


def archive_material(material_id: int, *, by) -> None:
    conn = sqlite3.connect(database.DB_PATH)
    try:
        conn.execute(
            "UPDATE materials SET archived_at = datetime('now') WHERE id=?",
            (material_id,),
        )
        conn.commit()
    finally:
        conn.close()
    from audit import log_event
    log_event(
        "material.archive",
        stream="user", actor=by,
        target_kind="material", target_id=str(material_id),
    )


def _row_to_material(row) -> Material:
    return Material(
        id=row["id"],
        kind=row["kind"],
        display_name=row["display_name"],
        filename=row["filename"],
        content_sha256=row["content_sha256"],
        size_bytes=row["size_bytes"],
        mime_type=row["mime_type"],
        assessment_type=row["assessment_type"],
        uploaded_by=row["uploaded_by"],
        uploaded_at=row["uploaded_at"],
        archived_at=row["archived_at"],
        notes=row["notes"],
    )
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_material_library.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add scripts/material_library.py
git commit -m "feat: add material_library (content-addressed storage + metadata)"
```

### Task 24: Source Materials tab (end-user)

**Files:**
- Modify: `app.py` (replace the `lambda: st.info("Source Materials — coming in Phase 4")` placeholder from Task 17)

- [ ] **Step 1: Add the tab handler**

Add a top-level function in `app.py`:

```python
def tab_source_materials():
    import material_library as ml
    user = identity.get_current_user()
    st.subheader("Source Materials")

    cols = st.columns([2, 1, 1, 1])
    with cols[0]:
        search = st.text_input("Search by name", key="sm_search")
    with cols[1]:
        kind_filter = st.selectbox(
            "Kind", ["(all)", "rubric", "answer_key", "exemplar", "student_notes"],
            key="sm_kind",
        )
    with cols[2]:
        from assessment_types import REGISTRY
        atypes = ["(all)"] + list(REGISTRY.keys())
        atype_filter = st.selectbox("Assessment type", atypes, key="sm_atype")
    with cols[3]:
        show_archived = st.checkbox("Include archived", value=False, key="sm_arch")

    items = ml.list_materials(
        kind=None if kind_filter == "(all)" else kind_filter,
        assessment_type=None if atype_filter == "(all)" else atype_filter,
        include_archived=show_archived,
    )
    if search:
        items = [m for m in items if search.lower() in m.display_name.lower()]

    import pandas as pd
    df = pd.DataFrame([
        {"id": m.id, "name": m.display_name, "kind": m.kind,
         "assessment": m.assessment_type, "uploader": m.uploaded_by,
         "uploaded_at": m.uploaded_at, "archived": bool(m.archived_at)}
        for m in items
    ])
    st.dataframe(df, use_container_width=True)

    with st.expander("Upload new material"):
        with st.form("upload_material"):
            kind = st.selectbox("Kind", ["rubric", "answer_key", "exemplar", "student_notes"])
            name = st.text_input("Display name")
            atype = st.selectbox("Assessment type", list(REGISTRY.keys()))
            tags_raw = st.text_input("Tags (comma-separated, optional)")
            notes = st.text_area("Notes (optional)")
            f = st.file_uploader("File", type=["xlsx", "csv", "pdf", "docx"])
            if st.form_submit_button("Upload") and f is not None and name:
                ml.save_material(
                    kind,  # type: ignore[arg-type]
                    file=f, filename=f.name,
                    display_name=name, assessment_type=atype,
                    tags=[t.strip() for t in tags_raw.split(",") if t.strip()],
                    uploaded_by=user.email, notes=notes or None,
                )
                st.success(f"Uploaded {f.name}")
                st.rerun()
```

Then replace the lambda at Task-17's `ALL_TABS` entry with `tab_source_materials`.

- [ ] **Step 2: Manual smoke**

Run: `streamlit run app.py`, sign in, click Source Materials, upload a sample file, verify the row appears and `osce_grader.db` contains it.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: Source Materials tab for upload/list/archive"
```

### Task 25: Material pickers in Grade Notes tab

**Files:**
- Modify: `app.py` (`tab_grade_notes` function near line 366 — replace the file uploaders for rubric and answer_key with pickers)

- [ ] **Step 1: Replace uploaders with pickers**

Inside `tab_grade_notes()`, for the rubric and answer-key inputs:

```python
import material_library as ml
atype = st.session_state.get("grade_assessment_type", "uk_osce")
rubric_options = ml.list_materials(kind="rubric", assessment_type=atype)
answer_options = ml.list_materials(kind="answer_key", assessment_type=atype)

rubric_choice = st.selectbox(
    "Rubric", options=rubric_options,
    format_func=lambda m: f"{m.display_name} ({m.uploaded_at[:10]})",
)
ak_choice = st.selectbox(
    "Answer key", options=answer_options,
    format_func=lambda m: f"{m.display_name} ({m.uploaded_at[:10]})",
)

# Inline shortcut upload
with st.expander("…or upload a new rubric / answer key"):
    col1, col2 = st.columns(2)
    with col1:
        nr = st.file_uploader("Rubric", key="gn_rubric_new", type=["xlsx", "csv", "pdf", "docx"])
        if nr is not None:
            user = identity.get_current_user()
            ml.save_material(
                "rubric", file=nr, filename=nr.name,
                display_name=f"(inline) {nr.name}", assessment_type=atype,
                uploaded_by=user.email,
            )
            st.rerun()
    with col2:
        na = st.file_uploader("Answer key", key="gn_ak_new", type=["xlsx", "csv"])
        if na is not None:
            user = identity.get_current_user()
            ml.save_material(
                "answer_key", file=na, filename=na.name,
                display_name=f"(inline) {na.name}", assessment_type=atype,
                uploaded_by=user.email,
            )
            st.rerun()
```

Where the grading call previously consumed `rubric_path`/`answer_key_path`, now open the selected Materials and write them to temp files (or pass the bytes directly if the grader accepts that). If the grader requires paths, use:

```python
import tempfile
def _material_to_path(m: ml.Material) -> str:
    with ml.open_material(m) as src:
        fd, path = tempfile.mkstemp(suffix=os.path.splitext(m.filename)[1])
        with os.fdopen(fd, "wb") as dst:
            dst.write(src.read())
    return path

rubric_path = _material_to_path(rubric_choice)
answer_key_path = _material_to_path(ak_choice)
```

- [ ] **Step 2: Opt-in student-notes-to-library checkbox**

Student notes remain a per-run uploader. Immediately after the uploader, add:

```python
keep_in_library = st.checkbox(
    "Keep this student-notes file in the library",
    value=False,
    help="Check this box only if you want to re-run this file later. Student notes often contain PII."
)
```

If checked after the uploader returns a file, call `ml.save_material("student_notes", ...)` once before the run starts.

- [ ] **Step 3: Smoke test**

Run: `streamlit run app.py`, sign in, go to Source Materials, upload a rubric and answer key, go back to Grade Notes, verify the pickers list them. Run grading on a tiny synthetic file; confirm results.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: Grade Notes uses material pickers for rubric/answer-key"
```

---

## Phase 5 — Grading run persistence

### Task 26: Add grading_runs columns via migration

**Files:**
- Modify: `scripts/database.py` — add a migration function for schema v3→v4; bump `CURRENT_SCHEMA_VERSION` to 4

- [ ] **Step 1: Add migration function**

Append to `scripts/database.py`:

```python
def _apply_migration_v4(conn: sqlite3.Connection) -> None:
    cols = [r[1] for r in conn.execute("PRAGMA table_info(grading_runs)").fetchall()]
    adds = [
        ("run_uuid",                "TEXT"),
        ("user_email",              "TEXT"),
        ("auth_session_id",         "TEXT"),
        ("provider",                "TEXT"),
        ("top_p",                   "REAL"),
        ("workers",                 "INTEGER"),
        ("max_tokens",              "INTEGER"),
        ("sections_json",           "TEXT"),
        ("rubric_material_id",      "INTEGER"),
        ("answer_key_material_id",  "INTEGER"),
        ("student_notes_sha256",    "TEXT"),
        ("summary_json",            "TEXT"),
        ("results_sha256",          "TEXT"),
    ]
    for name, typ in adds:
        if name not in cols:
            conn.execute(f"ALTER TABLE grading_runs ADD COLUMN {name} {typ}")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_grading_runs_run_uuid "
        "ON grading_runs(run_uuid) WHERE run_uuid IS NOT NULL"
    )
```

- [ ] **Step 2: Wire into `init_db()`**

Replace the body of `init_db()` with a migration-aware version:

```python
def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(_SCHEMA_SQL)
        current_row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version"
        ).fetchone()
        current = current_row[0]
        if current < 4:
            _apply_migration_v4(conn)
        if current < CURRENT_SCHEMA_VERSION:
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (CURRENT_SCHEMA_VERSION,),
            )
            from audit import log_event
            log_event(
                "db.migration",
                stream="system", severity="info",
                detail={"from": current, "to": CURRENT_SCHEMA_VERSION},
            )
```

Bump `CURRENT_SCHEMA_VERSION = 3` → `CURRENT_SCHEMA_VERSION = 4`.

- [ ] **Step 3: Smoke**

Run: `pytest tests/test_audit.py tests/test_material_library.py -v` (they both call `init_db`).
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/database.py
git commit -m "feat: migrate grading_runs to v4 with run_uuid and audit linkage"
```

### Task 27: Failing test for grading_runs row writes

**Files:**
- Create: `tests/test_grading_run_persistence.py`

- [ ] **Step 1: Write the test**

```python
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
    conn = sqlite3.connect(
        __import__("database").DB_PATH
    )
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM grading_runs WHERE run_id=?", (row_id,)).fetchone()
    assert row["status"] == "complete"
    assert row["results_sha256"] == "resulthash"
```

- [ ] **Step 2: Run — expect fail**

Run: `pytest tests/test_grading_run_persistence.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'grading_runs'`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_grading_run_persistence.py
git commit -m "test: failing tests for grading_runs persistence"
```

### Task 28: Implement `grading_runs.py`

**Files:**
- Create: `scripts/grading_runs.py`

- [ ] **Step 1: Write the module**

```python
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
    rubric_id: int | None = None,        # synthetic-only legacy path
    source_type: str = "uploaded",
    session_id: int | None = None,        # synthetic_sessions FK (legacy)
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
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_grading_run_persistence.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add scripts/grading_runs.py
git commit -m "feat: grading_runs row writer + results-file storage helper"
```

### Task 29: Emit run-lifecycle audit events from `app.py`

**Files:**
- Modify: `app.py` (the grading execution path inside `tab_grade_notes`)

- [ ] **Step 1: Wrap the run**

Around the existing `process_assessment(...)` call:

```python
import uuid, hashlib
import grading_runs as gr
import audit

user = identity.get_current_user()
run_uuid = ctx.run_id
# Hash student-notes file
with open(student_notes_path, "rb") as f:
    notes_bytes = f.read()
notes_sha = hashlib.sha256(notes_bytes).hexdigest()

row_id = gr.begin_run(
    run_uuid=run_uuid, user_email=user.email,
    auth_session_id=user.session_id,
    assessment_type_id=assessment_type,
    provider=ctx.provider, model=ctx.model,
    temperature=ctx.temperature, top_p=ctx.top_p,
    workers=ctx.workers, max_tokens=ctx.max_tokens,
    sections=ctx.sections,
    rubric_material_id=rubric_choice.id,
    answer_key_material_id=ak_choice.id,
    student_notes_sha256=notes_sha,
)
audit.log_event("grading.run.start", stream="user", actor=user,
                target_kind="run", target_id=run_uuid,
                detail={"provider": ctx.provider, "model": ctx.model})

try:
    results_df = process_assessment(
        assessment_type_instance, caller, file_paths, output_file,
        temperature=ctx.temperature, top_p=ctx.top_p, max_workers=ctx.workers,
        progress_callback=cb, ctx=ctx,
    )
    with open(output_file, "rb") as f:
        results_bytes = f.read()
    results_sha = gr.store_results_file(results_bytes)
    gr.complete_run(row_id, results_sha256=results_sha, summary={
        "rows": len(results_df),
    })
    audit.log_event("grading.run.complete", stream="user", actor=user,
                    target_kind="run", target_id=run_uuid,
                    detail={"rows": len(results_df), "results_sha": results_sha})
except Exception as exc:
    gr.cancel_run(row_id, reason=str(exc)[:200])
    audit.log_event("grading.run.cancel", stream="user", actor=user,
                    severity="error", outcome="failure",
                    target_kind="run", target_id=run_uuid,
                    error_code=type(exc).__name__,
                    detail={"error": str(exc)[:200]})
    raise
```

- [ ] **Step 2: Emit `results.download`**

When the user clicks the download button for the xlsx, emit:

```python
audit.log_event(
    "results.download", stream="user", actor=user,
    target_kind="run", target_id=run_uuid,
    target_hash=results_sha,
)
```

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: emit grading.run.start/complete/cancel and results.download"
```

---

## Phase 6 — Admin Audit Log tab + export

### Task 30: Audit Log tab handler

**Files:**
- Modify: `app.py` (replace the Task-17 `Audit Log` lambda)

- [ ] **Step 1: Add the handler**

```python
def tab_audit_log():
    user = identity.get_current_user()
    if not identity.is_admin(user):
        import audit
        audit.log_event(
            "tab.access.denied", stream="user", actor=user,
            severity="warn", outcome="denied",
            target_kind="tab", target_id="audit_log",
        )
        st.error("This tab requires admin privileges.")
        return

    import audit
    from datetime import datetime, timedelta

    st.subheader("Audit Log")
    cols = st.columns(4)
    with cols[0]:
        stream = st.selectbox("Stream", ["(all)", "user", "system"])
    with cols[1]:
        since = st.date_input("Since", value=(datetime.utcnow() - timedelta(days=7)).date())
    with cols[2]:
        until = st.date_input("Until", value=datetime.utcnow().date())
    with cols[3]:
        actor = st.text_input("Actor email (exact)")

    action = st.text_input("Action (exact, optional)")
    severity = st.selectbox("Severity", ["(all)", "info", "warn", "error"])

    events = audit.query_events(
        stream=None if stream == "(all)" else stream,
        actor_email=actor or None,
        action=action or None,
        severity=None if severity == "(all)" else severity,
        since=datetime.combine(since, datetime.min.time()),
        until=datetime.combine(until, datetime.max.time()),
        limit=2000,
    )

    import pandas as pd
    df = pd.DataFrame([{
        "ts": e.ts, "stream": e.stream, "severity": e.severity,
        "action": e.action, "actor_email": e.actor_email,
        "outcome": e.outcome, "target": f"{e.target_kind}:{e.target_id}" if e.target_kind else "",
        "error_code": e.error_code, "detail": e.detail_json,
    } for e in events])
    st.dataframe(df, use_container_width=True)

    if st.button("Export to CSV"):
        import io
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button(
            "Download audit_log.csv",
            data=buf.getvalue(),
            file_name=f"audit_log_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        audit.log_event(
            "audit.export", stream="user", actor=user,
            detail={"rows": len(df)},
        )
```

Wire it into `ALL_TABS` replacing the Task-17 lambda:
`("Audit Log", tab_audit_log, "admin"),`

- [ ] **Step 2: Smoke**

Run: `streamlit run app.py`, sign in as admin (`OSCE_ADMIN_EMAILS=you@…`), click Audit Log, verify rows from earlier actions appear. Click Export, download the CSV.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: admin Audit Log tab with filters and CSV export"
```

---

## Phase 7 — Server-mode deployment surface

### Task 31: `server_env.py` + `OSCE_SERVER_MODE` flag

**Files:**
- Create: `scripts/server_env.py`
- Modify: `scripts/providers.py` (read server-mode flag to disable key-file fallback)

- [ ] **Step 1: Write `server_env.py`**

```python
# scripts/server_env.py
"""Central parsing of server-environment variables."""

from __future__ import annotations

import os


def server_mode() -> bool:
    return os.environ.get("OSCE_SERVER_MODE", "0") == "1"


def log_json() -> bool:
    return os.environ.get("OSCE_LOG_JSON", "0") == "1"


def data_dir() -> str:
    return os.environ.get("OSCE_DATA_DIR", os.getcwd())


def db_path() -> str:
    return os.environ.get("OSCE_DB_PATH", os.path.join(data_dir(), "osce_grader.db"))


def storage_dir() -> str:
    return os.environ.get("OSCE_STORAGE_DIR", os.path.join(data_dir(), "storage"))


def audit_retention_user_days() -> int:
    return int(os.environ.get("OSCE_AUDIT_RETENTION_USER_DAYS", "2557"))


def audit_retention_system_days() -> int:
    return int(os.environ.get("OSCE_AUDIT_RETENTION_SYSTEM_DAYS", "90"))


def results_retention_days() -> int:
    return int(os.environ.get("OSCE_RESULTS_RETENTION_DAYS", "730"))


def admin_emails() -> set[str]:
    raw = os.environ.get("OSCE_ADMIN_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}
```

- [ ] **Step 2: Wire `OSCE_DATA_DIR` / `OSCE_DB_PATH` into `database.py`**

At `scripts/database.py` line ~19 where `DB_PATH` is defined, replace:

```python
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "osce_grader.db",
)
```

with:

```python
def _default_db_path() -> str:
    import server_env
    return server_env.db_path()

DB_PATH = _default_db_path()
```

- [ ] **Step 3: Wire server-mode key-file disable into `providers.py`**

At the point in `providers.py` that falls back to `api_key.txt`, guard it:

```python
import server_env
if server_env.server_mode():
    raise RuntimeError(
        f"{env_var} not set. In OSCE_SERVER_MODE=1, only env vars are consulted."
    )
# existing file-fallback code here
```

- [ ] **Step 4: Commit**

```bash
git add scripts/server_env.py scripts/database.py scripts/providers.py
git commit -m "feat: OSCE_SERVER_MODE + env-driven paths"
```

### Task 32: Enforce sign-in under server mode

**Files:**
- Modify: `app.py` (the sign-in block from Task 16)

- [ ] **Step 1: In server mode, make the form stricter**

Where the sign-in form is rendered, if `server_env.server_mode()`, display a plain message rather than the "Your email is recorded" caption:

```python
import server_env
if server_env.server_mode():
    st.info("This system is monitored. All activity is logged.")
```

No behavior change in non-server-mode. The enforcement is that the sign-in gate already blocks tab access; server mode simply surfaces the notice.

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: server-mode sign-in notice"
```

### Task 33: Startup retention sweep + daily background thread

**Files:**
- Modify: `app.py` and `scripts/grader.py` (schedule sweep at app start)

- [ ] **Step 1: Add a scheduler helper**

Append to `scripts/audit.py`:

```python
import threading


def schedule_daily_sweep(user_days: int, system_days: int) -> None:
    """Start a daemon thread that sweeps once per 24h."""
    def loop():
        import time
        while True:
            try:
                retention_sweep(user_days=user_days, system_days=system_days)
            except Exception as exc:  # pragma: no cover
                print(f"audit.sweep_failure: {exc}", file=sys.stderr)
            time.sleep(24 * 60 * 60)

    t = threading.Thread(target=loop, daemon=True, name="audit-retention-sweep")
    t.start()
```

- [ ] **Step 2: Call it at startup**

In `app.py` right after `log_event("app.start", ...)` from Task 11:

```python
if server_env.server_mode():
    from audit import schedule_daily_sweep, retention_sweep
    # One sweep on boot, then daily
    retention_sweep(
        user_days=server_env.audit_retention_user_days(),
        system_days=server_env.audit_retention_system_days(),
    )
    schedule_daily_sweep(
        user_days=server_env.audit_retention_user_days(),
        system_days=server_env.audit_retention_system_days(),
    )
```

Do the same at the top of `scripts/grader.py:main()` — **only** run the one-shot sweep (no thread for CLI):

```python
if server_env.server_mode():
    from audit import retention_sweep
    retention_sweep(
        user_days=server_env.audit_retention_user_days(),
        system_days=server_env.audit_retention_system_days(),
    )
```

- [ ] **Step 3: Commit**

```bash
git add scripts/audit.py app.py scripts/grader.py
git commit -m "feat: schedule audit retention sweep at startup in server mode"
```

### Task 34: JSON structured logging toggle

**Files:**
- Modify: `app.py`, `scripts/grader.py` (the `logging.basicConfig` calls)

- [ ] **Step 1: Add a formatter**

Create a small helper used by both entry points. Append to `scripts/server_env.py`:

```python
import logging


class JsonFormatter(logging.Formatter):
    def format(self, record):
        import json
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    if log_json():
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
```

- [ ] **Step 2: Use it**

At the top of `app.py` (replacing existing `logging.basicConfig(...)` from line ~78):

```python
import server_env
server_env.configure_logging()
```

At the top of `scripts/grader.py:main()`:

```python
import server_env
server_env.configure_logging()
```

- [ ] **Step 3: Commit**

```bash
git add scripts/server_env.py app.py scripts/grader.py
git commit -m "feat: JSON structured logging toggle via OSCE_LOG_JSON"
```

### Task 35: Integration test — end-to-end with mock LLM

**Files:**
- Create: `tests/test_server_integration.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_server_integration.py
"""End-to-end: sign in → upload materials → run grade → inspect audit."""

import io
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


@pytest.fixture
def server_env(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("OSCE_DATA_DIR", td)
        monkeypatch.setenv("OSCE_STORAGE_DIR", os.path.join(td, "storage"))
        monkeypatch.setenv("OSCE_ADMIN_EMAILS", "admin@x.edu")
        import database
        monkeypatch.setattr(database, "DB_PATH", os.path.join(td, "t.db"))
        database.init_db()
        yield td


class FakeSessionState(dict):
    def __getattr__(self, k): return self[k]
    def __setattr__(self, k, v): self[k] = v


@pytest.fixture
def fake_streamlit(monkeypatch):
    import types
    fake_st = types.SimpleNamespace(session_state=FakeSessionState())
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    return fake_st


def test_end_to_end_audit_sequence(server_env, fake_streamlit):
    import identity
    import material_library as ml
    import audit

    # Sign in
    user = identity.sign_in("admin@x.edu")
    assert identity.is_admin(user)

    # Upload a rubric
    rubric = ml.save_material(
        "rubric", file=io.BytesIO(b"rubric"), filename="r.xlsx",
        display_name="R", assessment_type="uk_osce", uploaded_by=user.email,
    )
    assert rubric.id > 0

    # Verify audit sequence
    events = audit.query_events()
    actions = [e.action for e in events]
    assert "sign_in" in actions
    assert "material.upload" in actions
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_server_integration.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_server_integration.py
git commit -m "test: end-to-end integration covering sign-in + upload + audit"
```

### Task 36: `Dockerfile` + `.env.example`

**Files:**
- Create: `Dockerfile`
- Create: `.env.example`

- [ ] **Step 1: Write `Dockerfile`**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OSCE_DATA_DIR=/data
ENV OSCE_SERVER_MODE=1
ENV OSCE_LOG_JSON=1
VOLUME ["/data"]

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

- [ ] **Step 2: Write `.env.example`**

```
# OSCE Grader server configuration
OSCE_SERVER_MODE=1
OSCE_LOG_JSON=1
OSCE_DATA_DIR=/var/lib/osce-grader
OSCE_ADMIN_EMAILS=admin@example.edu
OSCE_AUDIT_RETENTION_USER_DAYS=2557
OSCE_AUDIT_RETENTION_SYSTEM_DAYS=90
OSCE_RESULTS_RETENTION_DAYS=730
# LLM provider keys (set at least one)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
```

- [ ] **Step 3: Commit**

```bash
git add Dockerfile .env.example
git commit -m "chore: Dockerfile and .env.example for server deployment"
```

### Task 37: Deployment docs

**Files:**
- Create: `docs/server_deployment.md`
- Modify: `README.md` (add a link)

- [ ] **Step 1: Write the docs**

```markdown
# Server deployment

This application can run on a Linux VM (systemd) or a container platform
(Docker / Kubernetes / OpenShift). The same artifact works on either —
only env vars differ.

## Env vars

See `.env.example` for the full list. Minimum required for a working
server-mode deployment:

- `OSCE_SERVER_MODE=1` — enforces sign-in, disables key-file fallback
- `OSCE_DATA_DIR=/var/lib/osce-grader` (VM) or `/data` (container)
- `OSCE_ADMIN_EMAILS=comma,separated,list`
- At least one of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`

## Linux VM (systemd)

```ini
# /etc/systemd/system/osce-grader.service
[Unit]
Description=OSCE Grader
After=network.target

[Service]
User=osce
Group=osce
EnvironmentFile=/etc/osce-grader.env
WorkingDirectory=/opt/osce-grader
ExecStart=/opt/osce-grader/venv/bin/streamlit run app.py \
    --server.port=8501 --server.address=127.0.0.1
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Put a reverse proxy (NGINX / Apache) in front of `127.0.0.1:8501` for TLS
termination. Audit data lives at `$OSCE_DATA_DIR/osce_grader.db`; back it up.

## Container

```bash
docker build -t osce-grader .
docker run -d \
  --name osce-grader \
  -p 8501:8501 \
  --env-file /etc/osce-grader.env \
  -v /srv/osce-grader-data:/data \
  osce-grader
```

## Backup and retention

- Database: nightly `sqlite3 osce_grader.db .backup backup.db`.
- Materials: `$OSCE_STORAGE_DIR/materials` — rsync / object-store sync.
- Retention sweeps run automatically in server mode (startup + daily).
```

- [ ] **Step 2: Link from README**

Add to `README.md` after the "Resources" section:

```markdown
## Server Deployment
See [`docs/server_deployment.md`](docs/server_deployment.md) for VM and container deployment guides.
```

- [ ] **Step 3: Commit**

```bash
git add docs/server_deployment.md README.md
git commit -m "docs: server deployment guide (VM + container)"
```

---

## Finalization

### Task 38: Full test sweep

- [ ] Run: `pytest tests/ -v`
Expected: all pass.

### Task 39: CLI regression against gold-standard

- [ ] Run the CLI against a gold-standard set:

```bash
python scripts/grader.py \
  --rubric examples/sample_standard_rubric.xlsx \
  --answer_key examples/sample_flankpain_key.xlsx \
  --notes examples/sample_student_notes.xlsx \
  --output /tmp/regression.xlsx
```

Compare `/tmp/regression.xlsx` against the pre-refactor `results.xlsx` in the repo for score parity. Discrepancies larger than rounding indicate a regression.

### Task 40: Update `requirements.txt`

- [ ] Add `freezegun>=1.2` if retention tests rely on it (Task 10 uses relative timestamps; not strictly required — skip if unused).

### Task 41: Final review and PR

- [ ] Run: `git log --oneline main..HEAD` — verify logical commits, each phase represented.
- [ ] Open a PR from `feat/server-audit-upload` → `main` with a description referencing the spec file and the 7 phases.
