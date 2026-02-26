"""LLM provider abstraction for OSCE Grader.

Supports OpenAI, Anthropic (Claude), and Google (Gemini).  Each provider
is a callable that accepts a messages list and generation parameters,
and returns the model's text response as a string.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

import config

logger = logging.getLogger("osce_grader.providers")

# Type alias for the LLM call function signature
LLMCaller = Callable[[list[dict[str, str]], float, float], str]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# API key resolution (per-provider)
# ---------------------------------------------------------------------------

# Map of provider name -> (env var name, default key file name)
_KEY_CONFIG = {
    "openai":    ("OPENAI_API_KEY",    "openai_api_key.txt"),
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic_api_key.txt"),
    "google":    ("GOOGLE_API_KEY",    "google_api_key.txt"),
}


def get_api_key(provider: str) -> str:
    """Return the API key for the given provider.

    Resolution order:
      1. Provider-specific environment variable
      2. Contents of provider-specific key file in scripts/ directory
      3. Legacy: api_key.txt (for openai provider only, backward compat)

    Raises SystemExit if no valid key can be found.
    """
    env_var, key_file = _KEY_CONFIG[provider]

    # 1. Check provider-specific env var
    env_key = os.environ.get(env_var, "").strip()
    if env_key:
        return env_key

    # 2. Check provider-specific key file
    key_path = os.path.join(SCRIPT_DIR, key_file)
    key = _read_key_file(key_path)
    if key:
        return key

    # 3. Legacy fallback for openai only: api_key.txt
    if provider == "openai":
        legacy_path = config.API_KEY_FILE
        if not os.path.isabs(legacy_path):
            legacy_path = os.path.join(SCRIPT_DIR, legacy_path)
        key = _read_key_file(legacy_path)
        if key:
            return key

    logger.error(
        "No API key found for provider '%s'.\n"
        "Set the %s environment variable or create %s in the scripts/ directory.",
        provider, env_var, key_file,
    )
    raise SystemExit(1)


def _read_key_file(path: str) -> Optional[str]:
    """Read and return the stripped contents of a key file, or None."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            key = f.read().strip()
            return key if key else None
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def _extract_system_prompt(
    messages: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]]]:
    """Separate the system prompt from the messages list.

    Returns ``(system_prompt, remaining_messages)``.  If no system message
    is found, returns ``("", messages)``.
    """
    system_parts: list[str] = []
    other_messages: list[dict[str, str]] = []
    for msg in messages:
        if msg["role"] == "system":
            system_parts.append(msg["content"])
        else:
            other_messages.append(msg)
    return "\n".join(system_parts), other_messages


def _remap_roles(
    messages: list[dict[str, str]],
    role_map: dict[str, str],
) -> list[dict[str, str]]:
    """Return a copy of *messages* with roles remapped via *role_map*."""
    return [
        {**msg, "role": role_map.get(msg["role"], msg["role"])}
        for msg in messages
    ]


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _make_openai_caller(api_key: str) -> LLMCaller:
    """Create an OpenAI-backed LLM caller."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    def call(
        messages: list[dict[str, str]], temperature: float, top_p: float,
    ) -> str:
        response = client.chat.completions.create(
            model=config.MODEL,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content.strip()

    return call


def _make_anthropic_caller(api_key: str) -> LLMCaller:
    """Create an Anthropic Claude-backed LLM caller."""
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    def call(
        messages: list[dict[str, str]], temperature: float, top_p: float,
    ) -> str:
        system_prompt, user_messages = _extract_system_prompt(messages)

        # Anthropic temperature range is 0.0-1.0
        clamped_temp = min(temperature, 1.0)
        if clamped_temp != temperature:
            logger.debug(
                "Anthropic: temperature clamped from %.2f to %.2f",
                temperature, clamped_temp,
            )

        # Anthropic does not allow both temperature and top_p simultaneously.
        # Use temperature only (the primary tuning parameter).
        message = client.messages.create(
            model=config.MODEL,
            max_tokens=config.MAX_TOKENS,
            system=system_prompt,
            messages=user_messages,
            temperature=clamped_temp,
        )
        return message.content[0].text

    return call


def _make_google_caller(api_key: str) -> LLMCaller:
    """Create a Google Gemini-backed LLM caller."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    def call(
        messages: list[dict[str, str]], temperature: float, top_p: float,
    ) -> str:
        system_prompt, user_messages = _extract_system_prompt(messages)

        # Gemini uses "model" instead of "assistant" for the assistant role
        remapped = _remap_roles(user_messages, {"assistant": "model"})

        # Build Content objects for Gemini
        contents = [
            types.Content(
                role=msg["role"],
                parts=[types.Part.from_text(text=msg["content"])],
            )
            for msg in remapped
        ]

        response = client.models.generate_content(
            model=config.MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=config.MAX_TOKENS,
            ),
        )
        return response.text.strip()

    return call


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDER_FACTORIES = {
    "openai":    _make_openai_caller,
    "anthropic": _make_anthropic_caller,
    "google":    _make_google_caller,
}

SUPPORTED_PROVIDERS = list(_PROVIDER_FACTORIES.keys())


def create_caller(provider: str) -> LLMCaller:
    """Create and return an LLM caller for the given provider.

    The returned callable has signature::

        call(messages, temperature, top_p) -> str
    """
    if provider not in _PROVIDER_FACTORIES:
        logger.error(
            "Unknown provider '%s'. Supported providers: %s",
            provider, ", ".join(SUPPORTED_PROVIDERS),
        )
        raise SystemExit(1)

    api_key = get_api_key(provider)
    return _PROVIDER_FACTORIES[provider](api_key)
