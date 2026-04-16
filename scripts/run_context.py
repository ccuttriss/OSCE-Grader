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
