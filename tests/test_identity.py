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
