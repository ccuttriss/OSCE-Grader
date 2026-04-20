"""Encrypt-at-rest support for sensitive values (API keys, etc.).

Uses Fernet (authenticated symmetric encryption from the cryptography
package). The master key is resolved in this order:

  1. ``OSCE_SECRET_KEY`` environment variable (base64-encoded 32 bytes).
  2. File ``.osce_secret_key`` inside the data directory (same dir that
     holds the SQLite DB). Created on first use with mode 0600.

If the ``cryptography`` package isn't importable at all, the module falls
back to plaintext storage with a single warning log line. Legacy
plaintext values are detected on decrypt and returned as-is so existing
databases keep working.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("osce_grader.secrets")

_FERNET_PREFIX = "gAAAAA"  # Fernet tokens always start with this.
_TOKEN_MARKER = "fernet:"  # Our explicit tag so we don't guess.

_cached_cipher = None  # initialized lazily, never re-read from disk after first use


def _fernet_available() -> bool:
    # Catch BaseException because some broken cryptography builds raise
    # pyo3_runtime.PanicException (which derives from BaseException, not
    # Exception). We still want the plaintext fallback in that case.
    try:
        from cryptography.fernet import Fernet  # noqa: F401
        return True
    except BaseException:
        return False


def _data_dir() -> str:
    import server_env
    return server_env.data_dir()


def _secret_key_file() -> str:
    return os.path.join(_data_dir(), ".osce_secret_key")


def _load_master_key() -> Optional[bytes]:
    env_val = os.environ.get("OSCE_SECRET_KEY", "").strip()
    if env_val:
        return env_val.encode("utf-8")

    path = _secret_key_file()
    if os.path.isfile(path):
        try:
            with open(path, "rb") as f:
                return f.read().strip()
        except OSError as exc:
            logger.warning("Could not read %s: %s", path, exc)
            return None

    # Generate a new key and persist it with restrictive permissions.
    import server_env
    if server_env.server_mode():
        logger.error(
            "No OSCE_SECRET_KEY set and server mode forbids auto-generating "
            "one. API keys will remain unencrypted at rest."
        )
        return None

    try:
        from cryptography.fernet import Fernet
    except BaseException:
        return None

    key = Fernet.generate_key()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        with open(path, "wb") as f:
            f.write(key)
        os.chmod(path, 0o600)
        logger.info("Generated new master encryption key at %s (0600)", path)
    except OSError as exc:
        logger.warning("Could not persist master key to %s: %s", path, exc)
        # Still return the key so encryption works for this process.
    return key


def _get_cipher():
    """Return a cached Fernet instance, or None if encryption is unavailable."""
    global _cached_cipher
    if _cached_cipher is not None:
        return _cached_cipher
    if not _fernet_available():
        logger.warning(
            "cryptography package not available; API keys will be stored "
            "in plaintext. Install 'cryptography' to enable encryption."
        )
        return None
    key = _load_master_key()
    if not key:
        return None
    try:
        from cryptography.fernet import Fernet
        _cached_cipher = Fernet(key)
    except BaseException as exc:
        logger.warning("Could not initialise Fernet cipher: %s", exc)
        return None
    return _cached_cipher


def is_encryption_active() -> bool:
    """True when stored values will be encrypted on write."""
    return _get_cipher() is not None


def encrypt(value: str) -> str:
    """Encrypt a string. Returns a prefixed token so we can detect format on read.

    If encryption is unavailable, returns the plaintext as-is.
    """
    cipher = _get_cipher()
    if cipher is None:
        return value
    token = cipher.encrypt(value.encode("utf-8")).decode("utf-8")
    return f"{_TOKEN_MARKER}{token}"


def decrypt(stored: str) -> str:
    """Return the plaintext for a stored value.

    Accepts values written by previous plaintext versions (no prefix). Raises
    ValueError if an encrypted token can't be decrypted (wrong master key).
    """
    if stored is None:
        return stored
    if not stored.startswith(_TOKEN_MARKER) and not stored.startswith(_FERNET_PREFIX):
        return stored  # legacy plaintext
    cipher = _get_cipher()
    if cipher is None:
        raise ValueError(
            "Stored value is encrypted, but no master key is available to "
            "decrypt it. Set OSCE_SECRET_KEY or restore the .osce_secret_key "
            "file in the data directory."
        )
    raw = stored[len(_TOKEN_MARKER):] if stored.startswith(_TOKEN_MARKER) else stored
    try:
        return cipher.decrypt(raw.encode("utf-8")).decode("utf-8")
    except Exception as exc:
        raise ValueError(f"Could not decrypt stored value: {exc}") from exc


def mask_tail(value: str, tail: int = 4) -> str:
    """Return a masked display string like ``••••abcd``."""
    if not value:
        return ""
    if len(value) <= tail:
        return "\u2022" * len(value)
    return "\u2022" * 4 + value[-tail:]
