"""Password hashing and policy validation.

Stdlib-only: uses ``hashlib.scrypt`` for KDF-hashing (no bcrypt dependency).
Stored hash format::

    scrypt$<N>$<r>$<p>$<base64-salt>$<base64-dk>

Passwords are never stored in plaintext and never recoverable by anyone
(hashes are one-way). ``verify()`` uses :func:`hmac.compare_digest` for a
constant-time comparison.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re

# KDF parameters. These are the OWASP-recommended minimums for interactive
# auth in 2024-2026; adjust upward when hardware improves.
_SCRYPT_N = 2 ** 15  # CPU/memory cost
_SCRYPT_R = 8
_SCRYPT_P = 1
_SALT_BYTES = 16
_DK_BYTES = 64

# OpenSSL's default scrypt maxmem is 32 MiB, which is exactly the memory
# footprint of these parameters and can be rejected on some builds. Bump
# the cap so hash/verify don't fall over from the default.
_SCRYPT_MAXMEM = 1 << 26  # 64 MiB

_ALGO_TAG = "scrypt"


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

MIN_LENGTH = 12


def policy_violations(password: str) -> list[str]:
    """Return a list of human-readable reasons *password* fails the policy.

    An empty list means the password satisfies every rule.
    """
    issues: list[str] = []
    if len(password) < MIN_LENGTH:
        issues.append(f"must be at least {MIN_LENGTH} characters")
    if not re.search(r"[A-Z]", password):
        issues.append("must contain at least one uppercase letter")
    if not re.search(r"[a-z]", password):
        issues.append("must contain at least one lowercase letter")
    if not re.search(r"\d", password):
        issues.append("must contain at least one digit")
    if not re.search(r"[^A-Za-z0-9]", password):
        issues.append("must contain at least one symbol")
    if re.search(r"\s", password):
        issues.append("must not contain spaces")
    return issues


def is_valid(password: str) -> bool:
    return not policy_violations(password)


# ---------------------------------------------------------------------------
# Hash / verify
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Return a storable hash string for *password*."""
    if not isinstance(password, str) or not password:
        raise ValueError("password must be a non-empty string")
    salt = os.urandom(_SALT_BYTES)
    dk = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_DK_BYTES,
        maxmem=_SCRYPT_MAXMEM,
    )
    return "$".join([
        _ALGO_TAG,
        str(_SCRYPT_N),
        str(_SCRYPT_R),
        str(_SCRYPT_P),
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(dk).decode("ascii"),
    ])


def verify(password: str, stored_hash: str) -> bool:
    """Constant-time check of *password* against a stored hash.

    Returns False for malformed or empty hashes (e.g. bootstrap users who
    haven't set a password yet). Never raises on input shape.
    """
    if not stored_hash or not password:
        return False
    try:
        tag, n_s, r_s, p_s, salt_b64, dk_b64 = stored_hash.split("$")
    except ValueError:
        return False
    if tag != _ALGO_TAG:
        return False
    try:
        n, r, p = int(n_s), int(r_s), int(p_s)
        salt = base64.b64decode(salt_b64)
        expected_dk = base64.b64decode(dk_b64)
    except (ValueError, base64.binascii.Error):
        return False
    try:
        candidate_dk = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(expected_dk),
            maxmem=_SCRYPT_MAXMEM,
        )
    except (ValueError, MemoryError):
        return False
    return hmac.compare_digest(candidate_dk, expected_dk)
