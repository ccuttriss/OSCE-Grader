"""Tests for KPSOM OSCE assessment types.

Tests cover:
- derive_milestone() against known threshold values
- build_output_df() column names with and without faculty scores
- load_inputs() Q-number row skipping (using generated fixture files)
- Faculty scores trailing row filtering
- Score extraction helpers
"""

import os
import sys
import tempfile

import pandas as pd
import pytest
from openpyxl import Workbook

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from assessment_types.kpsom_osce import (
    KPSOMHandoffType,
    derive_milestone,
    _extract_float_score,
    _extract_int_score,
    _load_responses,
    _load_faculty_scores,
)
from assessment_types.kpsom_documentation import KPSOMDocumentationType
from assessment_types.base import GradingResult


# ---------------------------------------------------------------------------
# Fixtures: create minimal Excel files for testing
# ---------------------------------------------------------------------------

@pytest.fixture
def responses_xlsx(tmp_path):
    """Create a minimal KPSOM responses Excel file with Q-number row."""
    wb = Workbook()
    ws = wb.active

    # Row 0: Q-number metadata
    ws.append([None, "Q3", "Q4", "Q5", "Q6", "Q7"])
    # Row 1: Section headers
    ws.append(["Student", "Illness Severity", "Patient Summary",
               "Action List", "Situation Awareness", "Organization"])
    # Row 2+: Student data
    ws.append([1, "High acuity", "Patient is a 45yo...", "Check labs",
               "If worsens, call attending", "Clear handoff"])
    ws.append([2, "Low acuity", "Patient is a 30yo...", "Continue meds",
               "Stable", "Organized"])
    ws.append([3, "Medium", "Patient is a 55yo...", "Follow up imaging",
               "Watch for fever", "Good structure"])

    path = str(tmp_path / "responses.xlsx")
    wb.save(path)
    return path


@pytest.fixture
def faculty_scores_xlsx(tmp_path):
    """Create a minimal KPSOM faculty scores Excel file with 3 header rows
    and trailing template rows."""
    wb = Workbook()
    ws = wb.active

    # Row 0: Section group headers (merged)
    ws.append(["", "Illness Severity", "", "Patient Summary", "", "Total", "Milestone", "Comments"])
    # Row 1: Sub-group headers
    ws.append(["", "Item 1", "Item 2", "Item 3", "Item 4", "", "", ""])
    # Row 2: Real column headers
    ws.append(["Student", "IS_1", "IS_2", "PS_1", "PS_2", "Total", "Milestone", "Comments"])
    # Row 3+: Student data
    ws.append([1, 1.0, 0.5, 3.0, 2.0, 6.5, "Early Developing", "Good effort"])
    ws.append([2, 2.0, 1.0, 5.0, 4.0, 12.0, "Mid-Developing", "Well done"])
    ws.append([3, 1.5, 1.0, 4.0, 3.0, 9.5, "Early Developing", "Needs work"])
    # Trailing template rows (should be filtered out)
    ws.append([None, None, None, None, None, None, None, "Comment template 1"])
    ws.append([None, None, None, None, None, None, None, "Comment template 2"])

    path = str(tmp_path / "faculty_scores.xlsx")
    wb.save(path)
    return path


# ---------------------------------------------------------------------------
# Tests: derive_milestone()
# ---------------------------------------------------------------------------

class TestDeriveMilestone:
    """Test the milestone derivation formula against known thresholds."""

    def test_aspirational(self):
        assert derive_milestone(26, True, 2) == "Aspirational"

    def test_advanced_to_aspirational(self):
        assert derive_milestone(24, True, 2) == "Advanced Developing to Aspirational"

    def test_advanced_developing(self):
        assert derive_milestone(18, True, 1) == "Advanced Developing"

    def test_mid_to_advanced(self):
        assert derive_milestone(16, True, 1) == "Mid-Developing to Advanced Developing"

    def test_mid_developing(self):
        assert derive_milestone(14, False, 0) == "Mid-Developing"

    def test_early_to_mid(self):
        assert derive_milestone(12, False, 0) == "Early Developing to Mid-Developing"

    def test_early_developing(self):
        assert derive_milestone(9, False, 0) == "Early Developing"

    def test_entry_to_early(self):
        assert derive_milestone(7, False, 0) == "Entry to Early Developing"

    def test_boundary_25_with_sa_and_org2(self):
        # 25 is > 23 but not > 25
        assert derive_milestone(25, True, 2) == "Advanced Developing to Aspirational"

    def test_high_total_no_sa(self):
        # High total but no SA -> falls through to Mid-Developing
        assert derive_milestone(26, False, 2) == "Mid-Developing"

    def test_boundary_13(self):
        # 13 is not > 13
        assert derive_milestone(13, False, 0) == "Early Developing to Mid-Developing"


# ---------------------------------------------------------------------------
# Tests: Score extraction
# ---------------------------------------------------------------------------

class TestScoreExtraction:
    """Test float and integer score extraction from LLM output."""

    def test_float_on_last_line(self):
        text = "The student met criteria.\n3.5"
        assert _extract_float_score(text) == 3.5

    def test_integer_on_last_line(self):
        text = "Good handoff.\n4"
        assert _extract_float_score(text) == 4.0

    def test_score_with_trailing_whitespace(self):
        text = "Rationale here.\n  2.0  "
        assert _extract_float_score(text) == 2.0

    def test_score_pattern_fallback(self):
        text = "The student scored well. Score: 3"
        assert _extract_float_score(text) == 3.0

    def test_no_score_returns_none(self):
        text = "No numeric value here."
        assert _extract_float_score(text) is None

    def test_int_score_extraction(self):
        text = "Milestone level appropriate.\n4"
        assert _extract_int_score(text) == 4.0

    def test_int_score_from_float(self):
        text = "Good work.\n3.7"
        assert _extract_int_score(text) == 3.0  # truncated to int


# ---------------------------------------------------------------------------
# Tests: load_responses (Q-number row skipping)
# ---------------------------------------------------------------------------

class TestLoadResponses:
    """Test that _load_responses correctly handles the Q-number row."""

    def test_skips_q_number_row(self, responses_xlsx):
        df = _load_responses(responses_xlsx)
        # Should have 3 student rows, not 4 (Q-number row removed)
        assert len(df) == 3

    def test_correct_headers(self, responses_xlsx):
        df = _load_responses(responses_xlsx)
        assert "Student" in df.columns
        assert "Illness Severity" in df.columns
        assert "Patient Summary" in df.columns

    def test_q_numbers_stored_as_metadata(self, responses_xlsx):
        df = _load_responses(responses_xlsx)
        q_nums = df.attrs.get("q_numbers", {})
        assert q_nums.get("Illness Severity") == "Q3"
        assert q_nums.get("Patient Summary") == "Q4"

    def test_student_ids_are_correct(self, responses_xlsx):
        df = _load_responses(responses_xlsx)
        students = df["Student"].tolist()
        assert 1 in students
        assert 2 in students
        assert 3 in students


# ---------------------------------------------------------------------------
# Tests: load_faculty_scores (trailing row filtering)
# ---------------------------------------------------------------------------

class TestLoadFacultyScores:
    """Test that _load_faculty_scores filters trailing template rows."""

    def test_filters_trailing_rows(self, faculty_scores_xlsx):
        df = _load_faculty_scores(faculty_scores_xlsx)
        # Should have only 3 student rows, not 5 (2 template rows removed)
        assert len(df) == 3

    def test_student_ids_normalized_to_int(self, faculty_scores_xlsx):
        df = _load_faculty_scores(faculty_scores_xlsx)
        student_col = None
        for col in df.columns:
            if col.lower().strip() == "student":
                student_col = col
                break
        assert student_col is not None
        assert df[student_col].dtype in (int, "int64")

    def test_has_expected_columns(self, faculty_scores_xlsx):
        df = _load_faculty_scores(faculty_scores_xlsx)
        assert "Total" in df.columns
        assert "Milestone" in df.columns
        assert "Comments" in df.columns


# ---------------------------------------------------------------------------
# Tests: build_output_df columns
# ---------------------------------------------------------------------------

class TestBuildOutputDf:
    """Test that build_output_df produces expected columns."""

    def _make_result(self, with_faculty=False):
        sections = list(KPSOMHandoffType._sections.keys())
        scores = {s: float(i + 1) for i, s in enumerate(sections)}
        explanations = {s: f"Explanation for {s}" for s in sections}
        fac = None
        if with_faculty:
            fac = {s: float(i + 0.5) for i, s in enumerate(sections)}
            fac["total"] = 15.0
            fac["milestone"] = "Mid-Developing"
            fac["comments"] = "Good work"
        return GradingResult(
            student_id="1",
            section_scores=scores,
            section_explanations=explanations,
            total_score=sum(scores.values()),
            milestone="Early Developing",
            faculty_scores=fac,
        )

    def test_columns_without_faculty(self):
        at = KPSOMHandoffType()
        result = self._make_result(with_faculty=False)
        df = at.build_output_df([result])

        assert "student_id" in df.columns
        assert "ai_total" in df.columns
        assert "ai_milestone" in df.columns
        assert "illness_severity_ai_score" in df.columns
        assert "illness_severity_ai_explanation" in df.columns
        # No faculty columns
        assert "faculty_total" not in df.columns
        assert "total_delta" not in df.columns

    def test_columns_with_faculty(self):
        at = KPSOMHandoffType()
        result = self._make_result(with_faculty=True)
        df = at.build_output_df([result])

        assert "student_id" in df.columns
        assert "ai_total" in df.columns
        assert "faculty_total" in df.columns
        assert "total_delta" in df.columns
        assert "illness_severity_faculty_score" in df.columns
        assert "illness_severity_delta" in df.columns
        assert "faculty_comments" in df.columns

    def test_documentation_type_columns(self):
        at = KPSOMDocumentationType()
        sections = list(KPSOMDocumentationType._sections.keys())
        scores = {s: float(i + 1) for i, s in enumerate(sections)}
        explanations = {s: f"Explanation for {s}" for s in sections}
        result = GradingResult(
            student_id="1",
            section_scores=scores,
            section_explanations=explanations,
            total_score=sum(scores.values()),
        )
        df = at.build_output_df([result])

        assert "hpi_ai_score" in df.columns
        assert "social_hx_ai_score" in df.columns
        assert "ai_total" in df.columns
        assert "ai_pcig_total" in df.columns
        assert "ai_pcdp_total" in df.columns


# ---------------------------------------------------------------------------
# Tests: Assessment type interface
# ---------------------------------------------------------------------------

class TestAssessmentTypeInterface:
    """Test that assessment types implement the interface correctly."""

    def test_handoff_sections(self):
        at = KPSOMHandoffType()
        sections = at.get_sections()
        assert "illness_severity" in sections
        assert "patient_summary" in sections
        assert "organization" in sections
        assert len(sections) == 5

    def test_documentation_sections(self):
        at = KPSOMDocumentationType()
        sections = at.get_sections()
        assert "hpi" in sections
        assert "social_hx" in sections
        assert "summary_statement" in sections
        assert len(sections) == 5

    def test_handoff_required_files(self):
        at = KPSOMHandoffType()
        files = at.get_required_files()
        keys = [f["key"] for f in files]
        assert "rubric" in keys
        assert "responses" in keys
        assert "scores" in keys
        # Scores should be optional
        scores_spec = next(f for f in files if f["key"] == "scores")
        assert scores_spec["required"] is False

    def test_handoff_parse_llm_response(self):
        at = KPSOMHandoffType()
        text = "Student met criteria for items 1 and 3.\n2.5"
        explanation, score = at.parse_llm_response(text, "illness_severity")
        assert score == 2.5
        assert "criteria" in explanation

    def test_documentation_parse_llm_response(self):
        at = KPSOMDocumentationType()
        text = "Good clinical reasoning demonstrated.\n4"
        explanation, score = at.parse_llm_response(text, "hpi")
        assert score == 4.0

    def test_documentation_parse_clamps_score(self):
        at = KPSOMDocumentationType()
        text = "Excellent work.\n7"
        explanation, score = at.parse_llm_response(text, "hpi")
        assert score == 5.0  # clamped to max_score
