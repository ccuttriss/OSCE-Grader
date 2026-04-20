# tests/test_server_integration.py
"""End-to-end: sign in → upload materials → inspect audit."""

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
    import passwords as pw
    import database as db

    # Provision the admin account with a real password and sign in.
    good_pw = "Str0ngPass!word"
    db.create_user(
        "admin@x.edu",
        role="admin",
        password_hash=pw.hash_password(good_pw),
        must_change_password=False,
    )
    user = identity.sign_in("admin@x.edu", good_pw)
    assert identity.is_admin(user)

    # Upload a rubric
    rubric = ml.save_material(
        "rubric", file=io.BytesIO(b"rubric"), filename="r.xlsx",
        display_name="R", assessment_type="uk_osce", uploaded_by=user.email,
    )
    assert rubric.id > 0

    # Verify audit trail
    events = audit.query_events()
    actions = [e.action for e in events]
    assert "sign_in" in actions
    assert "material.upload" in actions
