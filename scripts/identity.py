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
    import streamlit
    return streamlit


def sign_in(email: str) -> User:
    email = email.strip().lower()
    if not _EMAIL_RE.match(email):
        raise ValueError("invalid email format")
    role: Literal["end_user", "admin"] = _resolve_role(email)
    user = User(email=email, role=role, session_id=str(uuid.uuid4()))
    _streamlit().session_state[_SESSION_KEY] = user
    from audit import log_event
    log_event("sign_in", stream="user", actor=user)
    return user


def _resolve_role(email: str) -> Literal["end_user", "admin"]:
    """Decide whether an email belongs to an admin.

    Rules:
      1. If the email is listed in OSCE_ADMIN_EMAILS, they're admin.
      2. Otherwise, if the process is NOT running in server mode, grant
         admin implicitly. This matches the local-workstation use case
         where the operator is always the admin and shouldn't need to
         pre-wire an env var just to reach the config tab.
      3. In server mode, unlisted emails stay as end_user.
    """
    import server_env
    if email in _admin_emails():
        return "admin"
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


def get_current_user() -> User | None:
    return _streamlit().session_state.get(_SESSION_KEY)


def is_admin(user: User | None) -> bool:
    if user is None:
        return False
    return user.role == "admin"


def cli_stub_user() -> User:
    """User stamped on audit rows emitted from CLI runs."""
    return User(email="cli_local", role="admin", session_id=str(uuid.uuid4()))
