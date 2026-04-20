# scripts/server_env.py
"""Central parsing of server-environment variables."""

from __future__ import annotations

import os


def server_mode() -> bool:
    """True unless explicitly disabled with ``OSCE_SERVER_MODE=0``.

    Server mode is the default: key files on disk are ignored, the API-key
    master key must come from ``OSCE_SECRET_KEY``, admin role requires an
    entry in ``OSCE_ADMIN_EMAILS``, and retention sweeps run. Set
    ``OSCE_SERVER_MODE=0`` only for a throwaway local-dev workstation.
    """
    return os.environ.get("OSCE_SERVER_MODE", "1") != "0"


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
