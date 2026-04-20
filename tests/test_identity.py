# tests/test_identity.py
import os
import sys
import tempfile
import importlib

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


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Point scripts/database.py at a throwaway DB and re-init it."""
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("OSCE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OSCE_DB_PATH", str(db_file))
    import database
    importlib.reload(database)
    database.DB_PATH = str(db_file)
    database.init_db()
    return database


GOOD_PW = "Str0ngPass!word"


def test_sign_in_rejects_bad_email(fake_streamlit, fresh_db):
    import identity
    with pytest.raises(identity.AuthError):
        identity.sign_in("not-an-email", GOOD_PW)


def test_sign_in_rejects_unknown_email(fake_streamlit, fresh_db):
    import identity
    with pytest.raises(identity.AuthError):
        identity.sign_in("nobody@example.edu", GOOD_PW)


def test_bootstrap_user_forces_password_change(fake_streamlit, fresh_db):
    import identity
    with pytest.raises(identity.PasswordChangeRequired):
        identity.sign_in("kpsomit@kp.org", "anything")
    # The force-change flow then finishes the sign-in.
    user = identity.complete_password_change("kpsomit@kp.org", GOOD_PW)
    assert user.email == "kpsomit@kp.org"
    assert user.role == "admin"
    # Next sign-in takes the normal path.
    user2 = identity.sign_in("kpsomit@kp.org", GOOD_PW)
    assert user2.role == "admin"


def test_complete_password_change_rejects_weak_password(fake_streamlit, fresh_db):
    import identity
    # Trigger the pending-change state first.
    with pytest.raises(identity.PasswordChangeRequired):
        identity.sign_in("kpsomit@kp.org", "anything")
    with pytest.raises(identity.AuthError):
        identity.complete_password_change("kpsomit@kp.org", "weak")


def test_admin_env_promotes_end_user(fake_streamlit, fresh_db, monkeypatch):
    import identity, passwords as pw
    fresh_db.create_user(
        "ops@example.edu",
        role="end_user",
        password_hash=pw.hash_password(GOOD_PW),
        must_change_password=False,
    )
    monkeypatch.setenv("OSCE_ADMIN_EMAILS", "ops@example.edu")
    user = identity.sign_in("ops@example.edu", GOOD_PW)
    assert user.role == "admin"


def test_wrong_password_fails(fake_streamlit, fresh_db):
    import identity, passwords as pw
    fresh_db.create_user(
        "u@example.edu",
        role="end_user",
        password_hash=pw.hash_password(GOOD_PW),
        must_change_password=False,
    )
    with pytest.raises(identity.AuthError):
        identity.sign_in("u@example.edu", "wrong-password-123!")


def test_sign_out_clears_session(fake_streamlit, fresh_db):
    import identity, passwords as pw
    fresh_db.create_user(
        "u@example.edu",
        role="end_user",
        password_hash=pw.hash_password(GOOD_PW),
        must_change_password=False,
    )
    identity.sign_in("u@example.edu", GOOD_PW)
    identity.sign_out()
    assert identity.get_current_user() is None


def test_cli_stub_user_is_admin():
    import identity
    user = identity.cli_stub_user()
    assert user.email == "cli_local"
    assert user.role == "admin"


def test_bootstrap_seed_is_idempotent(fresh_db):
    """Re-initialising doesn't duplicate the bootstrap user."""
    fresh_db.init_db()
    fresh_db.init_db()
    users = fresh_db.list_users()
    assert sum(1 for u in users if u["email"] == "kpsomit@kp.org") == 1
