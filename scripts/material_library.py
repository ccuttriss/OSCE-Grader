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
        actor=None,
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
