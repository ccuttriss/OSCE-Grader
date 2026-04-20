# scripts/identity.py
"""Identity service.

Email + password auth backed by the ``users`` table in the SQLite DB.
Bootstrap admin (``kpsomit@kp.org``) is seeded on first DB init with a
blank password and ``must_change_password=1``; the UI forces a password
to be set on initial sign-in.

``get_current_user()`` and ``is_admin()`` remain the stable callers'
interface. A SAML / SSO adapter will replace the interactive sign-in
flow later — tracked in ``docs/TODO.md``.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from typing import Literal

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SESSION_KEY = "_osce_user"


class AuthError(Exception):
    """Raised by sign_in() on any auth failure. Message is user-safe."""


class PasswordChangeRequired(AuthError):
    """Raised when credentials are correct but a password reset is required.

    The UI catches this and renders the force-change form instead of
    completing the sign-in.
    """


@dataclass(frozen=True)
class User:
    email: str
    role: Literal["end_user", "admin"]
    session_id: str


def _admin_emails() -> set[str]:
    raw = os.environ.get("OSCE_ADMIN_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def _streamlit():
    import streamlit
    return streamlit


def sign_in(email: str, password: str) -> User:
    """Authenticate an email/password pair and start a session.

    Raises :class:`AuthError` on any failure (unknown email, wrong password).
    Raises :class:`PasswordChangeRequired` if the credentials are valid but
    the account is flagged ``must_change_password``; the caller should then
    run :func:`complete_password_change` once a new password is collected.
    """
    email = (email or "").strip().lower()
    if not _EMAIL_RE.match(email):
        raise AuthError("Please enter a valid email address.")

    import database as db
    row = db.get_user(email)
    if row is None:
        raise AuthError(
            "No account found for that email. Ask an admin to add you."
        )

    import passwords as pw
    if row["password_hash"]:
        if not pw.verify(password or "", row["password_hash"]):
            raise AuthError("Incorrect password.")
    else:
        # Bootstrap / freshly-reset account. Any typed password is accepted
        # for the auth step, but the caller must immediately set a real one.
        pass

    if row["must_change_password"]:
        # Stash the pending email in session so the UI can render the
        # force-change form without trusting client state.
        _streamlit().session_state["_osce_pending_change"] = email
        raise PasswordChangeRequired(
            "You must set a new password before continuing."
        )

    return _establish_session(row)


def complete_password_change(email: str, new_password: str) -> User:
    """Validate + store *new_password* for *email* and open a session.

    Called only from the forced-change flow where the caller has already
    passed the knowledge check (blank bootstrap password or an admin-set
    temporary password).
    """
    email = (email or "").strip().lower()
    import passwords as pw
    issues = pw.policy_violations(new_password)
    if issues:
        raise AuthError("Password " + "; ".join(issues) + ".")

    import database as db
    row = db.get_user(email)
    if row is None:
        raise AuthError("Account no longer exists.")

    db.set_user_password_hash(
        email, pw.hash_password(new_password), must_change_password=False
    )
    row = db.get_user(email)
    _streamlit().session_state.pop("_osce_pending_change", None)
    return _establish_session(row)


def _establish_session(row: dict) -> User:
    role = _resolve_role(row)
    user = User(email=row["email"], role=role, session_id=str(uuid.uuid4()))
    _streamlit().session_state[_SESSION_KEY] = user
    from audit import log_event
    log_event("sign_in", stream="user", actor=user)
    return user


def _resolve_role(row: dict) -> Literal["end_user", "admin"]:
    """Decide the effective role for a user row.

    DB role is authoritative. The legacy ``OSCE_ADMIN_EMAILS`` env var is
    still honoured as an escalation path (emails listed there are admin
    even if their DB role is end_user), so existing deployments that
    relied on the env var continue to work while we transition fully to
    DB-driven RBAC.
    """
    db_role = row.get("role") or "end_user"
    if db_role == "admin":
        return "admin"
    if row.get("email") in _admin_emails():
        return "admin"
    import server_env
    if not server_env.server_mode():
        return "admin"
    return "end_user"


def sign_out() -> None:
    st = _streamlit()
    user = st.session_state.get(_SESSION_KEY)
    if user is None:
        return
    from audit import log_event
    log_event("sign_out", stream="user", actor=user)
    del st.session_state[_SESSION_KEY]
    st.session_state.pop("_osce_pending_change", None)


def get_current_user() -> User | None:
    return _streamlit().session_state.get(_SESSION_KEY)


def is_admin(user: User | None) -> bool:
    if user is None:
        return False
    return user.role == "admin"


def cli_stub_user() -> User:
    """User stamped on audit rows emitted from CLI runs."""
    return User(email="cli_local", role="admin", session_id=str(uuid.uuid4()))
