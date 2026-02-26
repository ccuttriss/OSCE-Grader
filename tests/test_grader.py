"""Tests for core grader.py functions beyond score extraction."""

import sys
import os

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from grader import (
    _row_already_graded,
    validate_input_columns,
)


class TestRowAlreadyGraded:
    """Tests for the resume-support helper."""

    def test_fully_graded_row(self):
        sections = ["hpi", "pex"]
        row = pd.Series({
            "hpi": "some note",
            "pex": "some note",
            "hpi_gpt_score": 8,
            "pex_gpt_score": 7,
        })
        assert _row_already_graded(row, sections) is True

    def test_partially_graded_row(self):
        sections = ["hpi", "pex"]
        row = pd.Series({
            "hpi": "some note",
            "pex": "some note",
            "hpi_gpt_score": 8,
            "pex_gpt_score": float("nan"),
        })
        assert _row_already_graded(row, sections) is False

    def test_ungraded_row(self):
        sections = ["hpi", "pex"]
        row = pd.Series({
            "hpi": "some note",
            "pex": "some note",
        })
        assert _row_already_graded(row, sections) is False

    def test_empty_section_counts_as_graded(self):
        """If a section has no content, it doesn't need a score."""
        sections = ["hpi", "pex"]
        row = pd.Series({
            "hpi": "some note",
            "pex": float("nan"),  # no content
            "hpi_gpt_score": 8,
        })
        assert _row_already_graded(row, sections) is True


class TestValidateInputColumns:
    """Tests for column validation."""

    def test_all_columns_present(self):
        df = pd.DataFrame({"hpi": [], "pex": [], "sum": []})
        assert validate_input_columns(df, ["hpi", "pex", "sum"]) == []

    def test_missing_columns(self):
        df = pd.DataFrame({"hpi": [], "pex": []})
        result = validate_input_columns(df, ["hpi", "pex", "sum"])
        assert result == ["sum"]

    def test_extra_columns_ok(self):
        df = pd.DataFrame({"hpi": [], "pex": [], "sum": [], "extra": []})
        assert validate_input_columns(df, ["hpi", "pex"]) == []
