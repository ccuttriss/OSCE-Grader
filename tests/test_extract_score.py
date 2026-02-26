"""Tests for the extract_score() function in grader.py.

Covers all three regex fallback patterns and various edge cases
to catch regressions in score parsing.
"""

import sys
import os

# Allow importing from the scripts directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from grader import extract_score


class TestBareIntegerOnOwnLine:
    """Pattern 1: A bare integer on its own line."""

    def test_simple_score(self):
        assert extract_score("Great work!\n8\n") == 8

    def test_score_with_whitespace(self):
        assert extract_score("Good effort.\n  7  \n") == 7

    def test_score_at_end(self):
        assert extract_score("The student demonstrated understanding.\n9") == 9

    def test_multiple_bare_integers_returns_last(self):
        assert extract_score("Section score:\n3\nFinal:\n5\n") == 5

    def test_zero_score(self):
        assert extract_score("No attempt.\n0\n") == 0


class TestScorePattern:
    """Pattern 2: 'Score: N' or 'Score = N' patterns."""

    def test_score_colon(self):
        assert extract_score("The answer was incomplete. Score: 6") == 6

    def test_score_equals(self):
        assert extract_score("Well done. Score = 9") == 9

    def test_score_dash(self):
        assert extract_score("Adequate. Score - 7") == 7

    def test_total_colon(self):
        assert extract_score("Some points missed. Total: 5") == 5

    def test_case_insensitive(self):
        assert extract_score("SCORE: 8") == 8
        assert extract_score("score: 4") == 4


class TestLastIntegerFallback:
    """Pattern 3: Last standalone integer anywhere in the text."""

    def test_integer_in_prose(self):
        # When no bare line or Score: pattern, falls back to last integer
        assert extract_score("The student scored around 7 out of 10 points.") == 10

    def test_single_integer(self):
        assert extract_score("awarded 6 points for this section") == 6


class TestEdgeCases:
    """Edge cases and potential false positives."""

    def test_no_numbers_returns_none(self):
        assert extract_score("No numeric content here at all.") is None

    def test_empty_string_returns_none(self):
        assert extract_score("") is None

    def test_date_in_text_bare_score_wins(self):
        # Bare integer on own line should be preferred over date numbers
        text = "Submitted on 2026-02-15.\nGood work.\n8\n"
        assert extract_score(text) == 8

    def test_page_numbers_with_score_pattern(self):
        text = "See page 42 for reference. Score: 7"
        assert extract_score(text) == 7

    def test_multiline_explanation_with_final_score(self):
        text = (
            "The student correctly identified 3 of 5 key findings.\n"
            "They missed the cardiac auscultation and lung fields.\n"
            "Partial credit awarded for effort.\n"
            "6\n"
        )
        assert extract_score(text) == 6

    def test_json_response(self):
        """Score extraction from JSON-like text (fallback scenario)."""
        text = '{"explanation": "Good work", "score": 8}'
        result = extract_score(text)
        assert result == 8
