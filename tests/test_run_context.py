# tests/test_run_context.py
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from run_context import RunContext, run_context_from_streamlit


def test_run_context_is_frozen():
    ctx = RunContext(
        run_id="abc",
        actor_email="cli_local",
        actor_role="admin",
        auth_session_id="sess-1",
        provider="openai",
        model="gpt-4o",
        temperature=0.3,
        top_p=1.0,
        workers=4,
        max_tokens=4096,
        assessment_type="uk_osce",
        sections=["hpi", "pex"],
    )
    with pytest.raises(Exception):
        ctx.provider = "anthropic"  # frozen dataclass must reject mutation


def test_run_context_required_fields():
    # Every field must be provided — no silent defaults on the grading path
    with pytest.raises(TypeError):
        RunContext(run_id="abc")  # type: ignore[call-arg]


def test_run_context_from_streamlit_fills_defaults():
    ctx = run_context_from_streamlit(
        provider="google",
        model="gemini-2.5-flash",
        temperature=0.3,
        top_p=1.0,
        workers=4,
        max_tokens=4096,
        assessment_type="uk_osce",
        sections=["hpi"],
    )
    assert ctx.actor_email == "unknown"
    assert ctx.actor_role == "end_user"
    assert len(ctx.run_id) >= 32
    assert ctx.auth_session_id != ctx.run_id
