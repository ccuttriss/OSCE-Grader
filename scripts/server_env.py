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
