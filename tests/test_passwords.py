import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import passwords as pw


GOOD = "Str0ngPass!word"  # 15 chars, upper, lower, digit, symbol, no space


def test_good_password_passes_policy():
    assert pw.policy_violations(GOOD) == []
    assert pw.is_valid(GOOD)


@pytest.mark.parametrize(
    "password,expected_issue_substr",
    [
        ("short1!A", "at least 12 characters"),
        ("alllowercase1!a", "uppercase letter"),
        ("ALLUPPERCASE1!A", "lowercase letter"),
        ("NoDigitsHere!abc", "digit"),
        ("NoSymbolsHere1abc", "symbol"),
        ("Has Space 1!abcZ", "must not contain spaces"),
    ],
)
def test_policy_flags_specific_issues(password, expected_issue_substr):
    issues = pw.policy_violations(password)
    assert any(expected_issue_substr in i for i in issues), issues


def test_hash_roundtrip_verifies():
    h = pw.hash_password(GOOD)
    assert h.startswith("scrypt$")
    assert pw.verify(GOOD, h)
    assert not pw.verify(GOOD + "x", h)
    assert not pw.verify("", h)


def test_verify_rejects_malformed_hash():
    assert pw.verify(GOOD, "") is False
    assert pw.verify(GOOD, "not-a-hash") is False
    assert pw.verify(GOOD, "scrypt$1$2$3$bad$bad") is False


def test_hash_with_empty_password_raises():
    with pytest.raises(ValueError):
        pw.hash_password("")
