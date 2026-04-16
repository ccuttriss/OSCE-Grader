# Server-readiness, audit, and source-material library — design

**Date:** 2026-04-16
**Status:** Approved for implementation planning
**Scope:** One spec covering audit subsystem, end-user upload UX, and the app-side changes required to move OSCE-Grader from a local-workstation Python tool to an internal managed server environment. SAML2 / PingID integration is included as a concrete TODO section with pre-built integration seams — it is **not** built in this spec's implementation.

---

## 1. Goals and constraints

### Goals

- Make the existing Streamlit app (`app.py`) safe and useful in an internal managed server environment that may run on either a Linux VM or a container platform.
- Provide an audit trail suitable for institutional/FERPA-adjacent compliance review **and** for engineering observability, in a single store with two logical streams.
- Give end users (faculty) a shared library of source materials (rubrics, answer keys, exemplars) so they can reuse uploads across grading runs.
- Separate end-user and developer/calibration tabs via a lightweight two-role model.
- Leave clean seams so SAML2 / PingID integration is a single-module swap later.

### Non-goals (v1)

- Kubernetes manifests, Helm charts, CI/CD pipelines, reverse-proxy configuration, or base-image hardening (deployment infrastructure lives in a separate follow-on spec).
- Multi-tenant / multi-school support.
- SAML2 / PingID implementation itself.
- Concurrency beyond single-digit, mostly-sequential users.
- Per-user access control on material library items (role-only gating in v1).
- Versioning of "same" material (new upload = new row, tag it).

### User / concurrency profile

1–3 faculty at a time, a grading run per sitting. Streamlit single-process is sufficient; SQLite remains the database; grading runs inline with a progress bar. The design fixes existing shared-state bugs so scaling up later is a config/infra change, not a rewrite.

### Deployment target

The managed server environment is not yet decided between a Linux VM and a container platform. The design is **container-ready but VM-friendly**: state is externalized behind env-driven paths, secrets come from env vars, storage is behind an abstraction with a filesystem default, logging can be toggled to JSON stdout.

### Audit drivers and data sensitivity

- Both compliance (institutional / FERPA-adjacent, student-grade data) and operational observability.
- Data contains PII but not PHI.
- Default retention: 7 years for the compliance stream (configurable), 90 days for the operational stream.
- Audit rows are metadata-only; note bodies are never stored in audit tables. Uploaded files live in the material library and audit rows reference them by SHA-256 hash.

---

## 2. System architecture

### 2.1 Module layout

```
Presentation
  app.py (Streamlit)
    sign-in page
    tab dispatcher (role-gated)
    tab handlers (existing + "Source Materials" + "Audit Log")

Services (new)
  scripts/identity.py           — sign-in, current user, role, SAML seam
  scripts/audit.py              — log_event, query_events, retention_sweep
  scripts/material_library.py   — save/list/get/open/archive materials

Domain (existing, light refactor)
  scripts/grader.py             — takes RunContext; no more config-global mutation on the grading path
  scripts/providers.py          — unchanged
  scripts/evaluate.py           — unchanged
  scripts/assessment_types/     — unchanged

Persistence (existing SQLite + new tables/dirs)
  scripts/database.py           — existing, with new table creation
  osce_grader.db:
    existing tables (api_keys, rubrics, rubric_sections, synthetic_sessions, example_files, grading_runs)
      — rubrics/rubric_sections/synthetic_sessions/example_files stay as-is;
        they are the synthetic-generator domain and are not merged into materials
    + audit_events                                — new
    + materials, material_tags                    — new
    grading_runs: existing columns preserved; new columns added (§6.2)
  storage/materials/<ab>/<cd>/<sha256>.<ext>   — content-addressed uploads
  storage/results/<sha256>.xlsx                — grading-run output artifacts
```

### 2.2 High-level request flow

1. Every Streamlit session starts at a sign-in gate. `identity.get_current_user()` returns `None` until `identity.sign_in(email)` is called.
2. After sign-in, the tab dispatcher filters tabs by role via `identity.is_admin(user)`.
3. Actions of consequence (upload, grading-run start/complete, results download, config change, admin access, errors) call `audit.log_event(...)`.
4. File uploads flow through `material_library.save_material(...)` which content-addresses to `storage/materials/<sha256>/` and writes metadata to SQLite.
5. Grading runs construct a `RunContext` (model, temperature, workers, user_id, run_id, selected materials) and pass it to `grader.process_assessment(...)`. No module-level config state is mutated.

### 2.3 What stays the same

`providers.py`, `evaluate.py`, everything under `assessment_types/`, and all scoring logic. The CLI grader (`python scripts/grader.py ...`) still works — it constructs a `RunContext` from argparse and runs under a stub `User(email="cli_local", role="admin")`. Existing tabs (Grade Notes, Analysis Dashboard, Flagged Items, Convert Rubric, Gold Standard, Synthetic Data) keep their internal logic; their upload controls are swapped for material pickers where applicable, and a role gate wraps the admin tabs.

---

## 3. Identity, sign-in, and roles

### 3.1 User model

```python
# scripts/identity.py
@dataclass(frozen=True)
class User:
    email: str
    role: Literal["end_user", "admin"]
    session_id: str          # generated at sign-in, stamped on every audit row
```

### 3.2 Interface

```python
def get_current_user() -> User | None: ...   # reads Streamlit session_state
def sign_in(email: str) -> User: ...          # validates email format, creates session, logs sign_in
def sign_out() -> None: ...                   # clears session_state, logs sign_out
def is_admin(user: User) -> bool: ...         # checks OSCE_ADMIN_EMAILS
```

### 3.3 Sign-in page

A single-field form ("Enter your institutional email to start a session") is shown before any tab. Email format is validated; a `session_id` (UUID) is generated; the user record is written to `st.session_state` and the `sign_in` audit event is emitted. There is no password — the trust model is honor-system, traceable through audit, until SAML replaces `sign_in()`.

### 3.4 Header and sign-out

After sign-in, every page shows a header with the user's email, their role label, and a sign-out link. Sign-out clears session state and emits a `sign_out` audit event.

### 3.5 Role gating

Two roles: `end_user` and `admin`. Admin membership is configured via `OSCE_ADMIN_EMAILS` (comma-separated list). Tab visibility:

- **End-user tabs:** Grade Notes, Analysis Dashboard, Flagged Items, Source Materials.
- **Admin-only tabs:** Convert Rubric, Gold Standard, Synthetic Data, Audit Log.

Streamlit's tab UI is session-local, so hidden tabs are not URL-addressable; role gating at the dispatcher is the primary defense. Each admin tab handler additionally asserts `is_admin(user)` at entry for defense-in-depth, and any failure emits a `tab.access.denied` user-stream event with severity `warn`.

### 3.6 CLI identity

The CLI grader injects a stub `User(email="cli_local", role="admin", session_id=<uuid>)`. Audit rows from CLI runs are stamped with that actor.

### 3.7 SAML seam

`get_current_user()` and `is_admin()` are the only functions callers use. A future SAML implementation replaces `sign_in()` (and the sign-in page render) without touching callers. See §8 for details.

---

## 4. Audit subsystem

### 4.1 Schema

```sql
CREATE TABLE audit_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL DEFAULT (datetime('now')),  -- ISO-8601 UTC
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
CREATE INDEX idx_audit_ts            ON audit_events(ts);
CREATE INDEX idx_audit_actor         ON audit_events(actor_email, ts);
CREATE INDEX idx_audit_stream_action ON audit_events(stream, action, ts);
CREATE INDEX idx_audit_target        ON audit_events(target_kind, target_id);
```

One table, two logical streams distinguished by the `stream` column, so queries and retention windows differ per stream but storage is shared.

### 4.2 Event catalog

**User stream (compliance, 7-year default retention):**

- `sign_in`, `sign_out`
- `material.upload`, `material.delete`, `material.download`, `material.archive`
- `grading.run.start`, `grading.run.complete`, `grading.run.cancel`
- `results.download`, `flagged.export`, `audit.export`
- `admin.config.change` (admin-only; includes env-config or admin-allowlist visible changes made in-app)
- `tab.access.denied`

**System stream (operational, 90-day default retention):**

- `llm.call` — provider, model, latency, tokens, cost estimate (per-call)
- `llm.retry`, `llm.failure`
- `grading.section.grade` — one per (run, student, section) for debug reconstruction
- `db.migration`, `storage.retention.sweep`
- `app.start`, `app.error` (uncaught exceptions; stack hash only, full stack to app log)

Note: `llm.call` and `grading.section.grade` are high-volume. The 90-day default retention bounds storage. If volume becomes a concern in practice, aggregate at the run level via `grading.run.complete.detail_json` and drop per-section rows — revisit after measuring.

### 4.3 Writer interface

```python
# scripts/audit.py
def log_event(
    action: str,
    *,
    stream: Literal["user","system"],
    actor: User | None = None,
    severity: Literal["info","warn","error"] = "info",
    outcome: Literal["success","failure","denied"] = "success",
    target_kind: str | None = None,
    target_id: str | None = None,
    target_hash: str | None = None,
    error_code: str | None = None,
    detail: dict | None = None,
    request_id: str | None = None,
) -> None:
    """Non-blocking audit write. Never raises — failures logged to stderr."""

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
) -> list[AuditEvent]: ...

def retention_sweep(now: datetime | None = None) -> dict: ...
```

- `log_event` must never raise. Its own failures are written to stderr and a single `audit.write_failure` entry is emitted to stderr (not the DB) to avoid recursion.
- `query_events` streams rows via generator internally; the list return is paginated via `limit`/`offset`.

### 4.4 Retention and sweeps

- `OSCE_AUDIT_RETENTION_USER_DAYS` (default `2557` ≈ 7 years).
- `OSCE_AUDIT_RETENTION_SYSTEM_DAYS` (default `90`).
- `retention_sweep()` runs at app startup and once every 24 hours thereafter on a background timer thread. Deletions themselves emit a `storage.retention.sweep` system-stream event with counts.

### 4.5 Tamper-evidence (v1)

- Application code never UPDATEs or DELETEs `audit_events` except via `retention_sweep()`.
- The audit table has no UPDATE path exposed through the module API.
- **Future TODO:** add a hash-chain column if compliance requirements tighten. This is called out in the spec but not built.

### 4.6 Admin audit UI

A new admin-only "Audit Log" tab provides:

- Date range, stream, action, actor, and severity filters.
- Table view of matching rows with a "details" expander for `detail_json`.
- CSV export of the filtered result set (streams to a file to avoid memory issues at large ranges).
- Exporting itself emits an `audit.export` user-stream event.

---

## 5. Source material library

### 5.1 Schema

```sql
CREATE TABLE materials (
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

CREATE TABLE material_tags (
    material_id INTEGER NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
    tag         TEXT    NOT NULL,
    PRIMARY KEY (material_id, tag)
);
```

### 5.2 Filesystem layout

```
$OSCE_STORAGE_DIR/materials/<first-2>/<next-2>/<sha256>.<ext>
```

Sharded 2×2 hex levels to keep per-directory file counts small. Files are content-addressed, immutable, and never overwritten. Deduplication is automatic — identical bytes from two uploads live as one on-disk file.

### 5.3 Kinds and behavior differences

- `rubric`, `answer_key`, `exemplar`: canonical library items; reuse across runs is the default.
- `student_notes`: **not added to the library by default**. Per-run upload only, hashed, referenced by the `grading_runs` row, retained for the per-run artifact lifetime (§6.4). Faculty can **opt in** at upload time to keep a student-notes file in the library (e.g., for re-runs); the opt-in is per upload and audited.

### 5.3.1 Relationship to existing `example_files` and `rubrics` tables

The existing `example_files` table (used by the synthetic generator) and the existing `rubrics` / `rubric_sections` tables (structured rubric content for synthetic sessions) are **not merged** into `materials`. They serve the admin-only synthetic-generator workflow and use a different shape (BLOB storage, structured section metadata). `materials` is the user-facing library for uploaded grading artifacts. Keeping them separate avoids conflating two different domain concepts and requires no risky data migration; a future pass could consolidate if it proves warranted.

### 5.4 Interface

```python
# scripts/material_library.py
def save_material(
    kind: Literal["rubric","answer_key","student_notes","exemplar"],
    *,
    file: BinaryIO,
    filename: str,
    display_name: str,
    assessment_type: str,
    tags: list[str] | None = None,
    uploaded_by: str,
    notes: str | None = None,
) -> Material: ...                              # logs material.upload

def list_materials(
    *, kind: str | None = None, assessment_type: str | None = None,
    tag: str | None = None, include_archived: bool = False,
) -> list[Material]: ...

def get_material(material_id: int | None = None, *, sha256: str | None = None) -> Material: ...
def open_material(material: Material) -> BinaryIO: ...   # reads from storage
def archive_material(material_id: int, *, by: User) -> None: ...   # soft delete, logged
```

`save_material`, `open_material`, and `archive_material` are the three seams for a future S3 adapter — swap their file I/O bodies, keep the signatures.

### 5.5 UI surfaces

**Source Materials tab (end user and admin):** searchable, filterable table with columns for name, kind, assessment type, tags, uploader, date. Actions per row: download, archive. Top of page: search box, filter selects (kind, assessment type), "Upload new…" button opening a form (kind, display name, assessment type, tags, notes, file).

**Grade Notes tab (end user):** existing rubric / answer-key file uploaders become **material pickers** — dropdowns listing library items filtered by assessment type. Each picker has an "…or upload" shortcut that saves to the library inline. Student notes remain a per-run file uploader with an opt-in checkbox "keep this file in the library."

### 5.6 Non-goals

- No versioning of a "same" material — tag a new upload instead.
- No per-user ACLs — role-only access.
- No in-library preview — users download to inspect.

---

## 6. `RunContext` and grading run persistence

### 6.1 `RunContext`

```python
# scripts/grader.py (or a new scripts/run_context.py)
@dataclass(frozen=True)
class RunContext:
    run_id: str
    user: User
    provider: str
    model: str
    temperature: float
    top_p: float
    workers: int
    max_tokens: int
    assessment_type: str
    sections: list[str]
```

`grader.process_assessment(ctx, rubric, answer_key, notes_df)` takes a `RunContext` and returns results. No module-level state is mutated. `config.py` keeps default values only; callers build a `RunContext` from session state + library selections per run. The CLI grader builds one from argparse.

### 6.2 `grading_runs` table additions

The existing `grading_runs` table is preserved (see `scripts/database.py`: `run_id`, `assessment_type_id`, `rubric_id`, `model_used`, `temperature`, `source_type`, `session_id` FK to `synthetic_sessions`, `started_at`, `completed_at`, `status`, `results_json`, `log_text`). The existing `rubric_id` references the synthetic-generator `rubrics` table and stays as-is for synthetic-run compatibility. Add the following **new columns** via a schema migration (all nullable for backward compatibility with historical rows):

- `run_uuid` (TEXT, unique) — user-facing id, referenced by audit rows as `detail_json.run_id`
- `user_email` (TEXT) — actor for the run
- `auth_session_id` (TEXT) — identity session; distinct from the existing `session_id` column, which is a FK to `synthetic_sessions`
- `provider` (TEXT), `top_p` (REAL), `workers` (INTEGER), `max_tokens` (INTEGER) — supplement existing `model_used` / `temperature`
- `sections_json` (TEXT)
- `rubric_material_id` (INTEGER, nullable FK to `materials(id)`) — set for uploaded runs; left NULL for synthetic runs that use `rubrics.rubric_id`
- `answer_key_material_id` (INTEGER, nullable FK to `materials(id)`)
- `student_notes_sha256` (TEXT) — per-run hash, not a library FK unless the user opted in
- `summary_json` (TEXT) — counts, mean score, token totals, cost estimate
- `results_sha256` (TEXT) — hash of the output xlsx; the file lives at `$OSCE_STORAGE_DIR/results/<sha256>.xlsx`

Migration details (DB `CURRENT_SCHEMA_VERSION` bump, `ALTER TABLE ... ADD COLUMN` statements, and a `db.migration` system-stream audit row) are captured in the implementation plan.

### 6.3 Results artifacts

Grading output xlsx files are written to `$OSCE_STORAGE_DIR/results/<sha256>.xlsx`. The `grading_runs` row references them by hash. The Analysis Dashboard and Flagged Items tabs can resurrect a past run by its row.

### 6.4 Results retention

`OSCE_RESULTS_RETENTION_DAYS` (default `730` ≈ 2 years). Retention sweep deletes the file on disk and the `grading_runs.results_sha256` is nulled; the row itself is retained for audit cross-reference. Per-run `student_notes` files are co-swept on the same schedule.

### 6.5 Run lifecycle audit correlation

Every `audit_events.detail_json` emitted during a run includes the `run_id`. The admin audit UI's "details" expander surfaces `run_id` as a filter link.

---

## 7. Deployment surface

Configuration is env-driven. The same image / same binary runs on a Linux VM (systemd) or a container (Docker / k8s) by setting paths.

| Env var | Default | Purpose |
| --- | --- | --- |
| `OSCE_DATA_DIR` | `./` | Root for DB + storage (single-volume mount in container) |
| `OSCE_DB_PATH` | `$OSCE_DATA_DIR/osce_grader.db` | SQLite location |
| `OSCE_STORAGE_DIR` | `$OSCE_DATA_DIR/storage` | Materials + results files |
| `OSCE_ADMIN_EMAILS` | (empty) | Comma-separated admin allowlist |
| `OSCE_AUDIT_RETENTION_USER_DAYS` | `2557` | 7 years |
| `OSCE_AUDIT_RETENTION_SYSTEM_DAYS` | `90` | Ops log horizon |
| `OSCE_RESULTS_RETENTION_DAYS` | `730` | Per-run results artifact lifetime |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` | — | LLM secrets |
| `OSCE_SERVER_MODE` | `0` | Server-mode flag: disables key-file fallback, enables retention sweeps, enforces sign-in |
| `OSCE_LOG_JSON` | `0` | Structured JSON stdout logs when running behind a log aggregator |

Process model: single `streamlit run app.py` process. VM deployment: systemd unit with `OSCE_DATA_DIR=/var/lib/osce-grader`. Container deployment: same image, volume mounted at `/data`. A `Dockerfile` is added in the implementation plan (no orchestration opinion — the image works on either target).

When `OSCE_SERVER_MODE=1`:

- API-key-file fallback in `providers.py` is disabled; only env vars are consulted.
- Sign-in is enforced (the app will not run without a user session).
- Retention sweep is scheduled at startup.
- `OSCE_LOG_JSON=1` is the recommended companion so the log aggregator can parse events.

---

## 8. SAML2 / PingID integration — TODO hooks

**Not built in this spec.** The following hooks exist so a future pass can drop SAML in with a bounded diff.

### 8.1 Replacement points

- `identity.sign_in(email)` and the sign-in page render are replaced by a SAML assertion consumer. Two viable paths:
  - **App-resident:** `python3-saml` behind a Streamlit query-params callback URL handling the `SAMLResponse` POST.
  - **Proxy-resident:** Apache `mod_auth_mellon` or NGINX `auth_request` + a SAML sidecar (e.g., `pomerium`, `oauth2-proxy` with a SAML upstream). The proxy terminates SAML and injects `REMOTE_USER` / attributes as headers.
- `identity.get_current_user()` learns to read `REMOTE_USER` + attributes from the Streamlit request headers (proxy path) or from a SAML session created by the in-app consumer.
- `identity.is_admin(user)` switches from `OSCE_ADMIN_EMAILS` to a PingID group claim (e.g., `OSCE-Admins` in the SAML attribute statement).

### 8.2 Schema impact

None required — `actor_email`, `actor_role`, and `session_id` already carry what SAML will supply. `audit_events.detail_json` gains optional `saml_nameid` and `saml_session_index` fields for logout correlation.

### 8.3 Config additions (future)

- `OSCE_AUTH_MODE` (`email|saml`, default `email`).
- `OSCE_SAML_METADATA_URL`, `OSCE_SAML_ENTITY_ID`, `OSCE_SAML_ACS_URL`, and either `OSCE_SAML_IDP_CERT` or reliance on the IdP certificate fetched from the metadata URL (or proxy-equivalent env).
- `OSCE_ADMIN_GROUP` (SAML group claim name that maps to `admin` role).

### 8.4 Rollout notes (future)

- Run both auth modes in parallel during cutover by keying on `OSCE_AUTH_MODE`.
- A one-time migration stamps `actor_role` on historical rows based on the email allowlist at migration time.

---

## 9. Error handling

- `audit.log_event()` never raises; failures go to stderr.
- LLM failures already retry in `providers.py`; we add `llm.retry` (severity `warn`) and `llm.failure` (severity `error`) system-stream events per attempt.
- A top-level Streamlit exception handler logs `app.error` (system stream, severity `error`) with a stack hash only; the full stack goes to the app log (stdout) where the log aggregator picks it up.
- Uploaded-file validation (schema-check of Excel columns) fails loudly with a `material.upload` row of outcome `failure` and a specific `error_code` (`FILE_INVALID_COLUMNS`, `FILE_TOO_LARGE`, `FILE_UNREADABLE`).
- Sign-in with an invalid email format returns an inline form error and does **not** emit an audit row (no session exists).
- A denied access (non-admin hits admin tab) emits `tab.access.denied` with `outcome=denied` and severity `warn`.

---

## 10. Testing strategy

### 10.1 Unit

- `identity`: sign-in / sign-out, role check, session ID uniqueness, CLI stub user.
- `audit`: insert, retention sweep (with frozen clock), query filters, no-raise guarantee.
- `material_library`: dedupe on sha256, archive semantics, tag filtering, unique `(sha256, kind)` constraint.

All pure pytest, no Streamlit runtime required.

### 10.2 Integration

End-to-end "upload rubric → upload answer key → upload student notes → run grading → download results → inspect audit log" driven through the `RunContext` domain API, using a temp `OSCE_DATA_DIR` and a mocked LLM provider. Existing grader tests continue to work because `RunContext` maps one-to-one from today's CLI args.

### 10.3 Audit as test oracle

Each integration test asserts the expected audit event sequence (e.g., `[sign_in, material.upload × 3, grading.run.start, llm.call × N, grading.run.complete, results.download]`). Audit coverage is treated as a first-class correctness property, not a side channel.

### 10.4 Regression

The CLI grader path is tested to confirm the `RunContext`-refactored grader produces identical scores to the pre-refactor version on the existing gold-standard sets in `examples/`.

---

## 11. Migration and rollout

Ordered steps — each independently releasable, each keeps the CLI functional throughout.

1. **`RunContext` refactor.** Introduce `RunContext`; thread it through `grader.process_assessment`; remove config-module-global mutation on the grading path. No audit writes, no sign-in gate yet. Verify grader tests and regression parity.
2. **Audit foundation.** Add `audit.py`, `audit_events` table, startup retention sweep. Start writing system-stream events only (`app.start`, `llm.call`, `llm.retry`, `llm.failure`, `app.error`).
3. **Identity + user-stream events + role gating.** Add `identity.py`, sign-in page, header, sign-out. Emit user-stream events for sign-in/out, tab access denials. Add role-based tab filtering.
4. **Material library.** Add `material_library.py`, `materials` and `material_tags` tables, filesystem layout, Source Materials tab, material pickers in Grade Notes. Emit user-stream events for material operations.
5. **Grading-run persistence.** Extend `grading_runs` with the columns in §6.2; write `results_sha256` artifacts; add run-lifecycle user-stream events. Wire Analysis Dashboard and Flagged Items to read from stored runs.
6. **Admin Audit Log tab + export.** Admin UI on top of `audit.query_events`; CSV export; `audit.export` event.
7. **Server-mode deployment.** `OSCE_SERVER_MODE=1` enforcement, `OSCE_LOG_JSON=1` structured logging, `Dockerfile`, deploy notes (systemd sample + container sample) in `docs/`.

SAML (§8) is a follow-on spec after step 7 lands.

---

## 12. Open items / future TODOs

- Hash-chain column on `audit_events` if compliance tightens (§4.5).
- Swap filesystem material storage for an S3-compatible adapter (§5.4 hooks).
- Move from SQLite to Postgres if concurrency assumptions change (§2).
- Aggregate `llm.call` / `grading.section.grade` per-run if system-stream volume becomes a pain point (§4.2).
- SAML2 / PingID implementation (§8).
- Versioning of library materials (currently: tag a new upload).
- Per-user or per-course ACLs on library items (currently: role-only).
- Admin UI for rotating LLM API keys and adjusting admin allowlist without a restart (currently: env vars, restart required).
