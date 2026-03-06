"""Tests for KPSOM Clinical Documentation assessment type (Case 3).

Tests cover:
- Responses loading with single header row
- Scores loading with student ID derived from row position
- None handling for non-submitters (students 11 and 32 pattern)
- build_output_df columns with and without faculty scores
- Milestone threshold derivation
"""

import os
import sys

import pandas as pd
import pytest
from openpyxl import Workbook

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from assessment_types.kpsom_documentation import (
    KPSOMDocumentationType,
    derive_documentation_milestone,
    _load_documentation_responses,
    _load_documentation_scores,
)
from assessment_types.base import GradingResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def doc_responses_xlsx(tmp_path):
    """Create a minimal Case 3 responses file."""
    wb = Workbook()
    ws = wb.active

    # Row 0: Section labels (header)
    ws.append([None, "HPI", "Social Hx", "Summary Statement", "Assessment", "Plan"])
    # Row 1+: Student data (col 0 = row number)
    ws.append([1, "Chest pain onset 2 hrs ago...", "Smoker, lives alone",
               "45yo M presenting with...", "ACS vs PE", "Troponin, ECG, CXR"])
    ws.append([2, "Abdominal pain for 3 days...", "Non-drinker, married",
               "62yo F presenting with...", "Cholecystitis vs pancreatitis", "US, lipase, CBC"])
    # Student 3 = non-submitter (all None)
    ws.append([3, None, None, None, None, None])
    ws.append([4, "Headache x 1 week...", "No PMH", "28yo M with...",
               "Tension HA vs migraine", "CT head, neuro exam"])

    path = str(tmp_path / "doc_responses.xlsx")
    wb.save(path)
    return path


@pytest.fixture
def doc_scores_xlsx(tmp_path):
    """Create a minimal Case 3 scores file.

    Simulates the flat single-header format with key columns at specific indices.
    We create a wide row with the right values at the right indices.
    """
    wb = Workbook()
    ws = wb.active

    # Create header row (row 0) with 82 columns
    header = [None] * 82
    header[0] = None  # No student ID column label
    header[29] = "HPI Score"
    header[34] = "Social Hx Score"
    header[49] = "Summary Statement Score"
    header[58] = "Assessment Score"
    header[72] = "Plan Score"
    header[73] = "Org/Lang Score"
    header[74] = "PCIG Total"
    header[75] = "PCIG Milestone"
    header[76] = "PCDP Total"
    header[77] = "PCDP Milestone"
    header[78] = "PCDO Score"
    header[79] = "PCDO Milestone"
    header[80] = "Total Score"
    header[81] = "Total Milestone"
    ws.append(header)

    # Student 1
    row1 = [None] * 82
    row1[29] = 4.0
    row1[34] = 3.5
    row1[49] = 3.0
    row1[58] = 4.0
    row1[72] = 4.5
    row1[73] = 3.5
    row1[74] = 7.5  # PCIG = HPI + Social Hx
    row1[75] = "Mid-Developing"
    row1[76] = 11.5  # PCDP
    row1[77] = "Mid-Developing"
    row1[78] = 3.5  # PCDO
    row1[79] = "Mid-Developing"
    row1[80] = 22.5  # Total
    row1[81] = "Advanced Developing to Aspirational"
    ws.append(row1)

    # Student 2
    row2 = [None] * 82
    row2[29] = 3.0
    row2[34] = 4.5
    row2[49] = 2.0
    row2[58] = 3.0
    row2[72] = 3.0
    row2[73] = 4.0
    row2[80] = 19.5
    row2[81] = "Advanced Developing"
    ws.append(row2)

    # Student 3 (non-submitter like students 11/32)
    row3 = [None] * 82
    row3[29] = 1.0  # Partial HPI score
    row3[34] = None
    row3[49] = 1.0  # Partial Summary Statement score
    row3[58] = None
    row3[72] = None
    row3[73] = None
    row3[80] = 2.0
    row3[81] = "Behavior requiring corrective response"
    ws.append(row3)

    # Student 4
    row4 = [None] * 82
    row4[29] = 3.5
    row4[34] = 3.0
    row4[49] = 4.0
    row4[58] = 4.5
    row4[72] = 4.0
    row4[73] = 3.0
    row4[80] = 22.0
    row4[81] = "Advanced Developing"
    ws.append(row4)

    path = str(tmp_path / "doc_scores.xlsx")
    wb.save(path)
    return path


# ---------------------------------------------------------------------------
# Tests: load_documentation_responses
# ---------------------------------------------------------------------------


class TestLoadDocumentationResponses:
    """Test that responses loading handles the single-header format."""

    def test_skips_header_row(self, doc_responses_xlsx):
        df = _load_documentation_responses(doc_responses_xlsx)
        # Should have 4 student rows (including non-submitter)
        assert len(df) == 4

    def test_correct_column_names(self, doc_responses_xlsx):
        df = _load_documentation_responses(doc_responses_xlsx)
        assert "student_id" in df.columns
        assert "hpi" in df.columns
        assert "social_hx" in df.columns
        assert "summary_statement" in df.columns
        assert "assessment" in df.columns
        assert "plan" in df.columns

    def test_nonsubmitter_has_none(self, doc_responses_xlsx):
        df = _load_documentation_responses(doc_responses_xlsx)
        student3 = df[df["student_id"] == 3].iloc[0]
        assert pd.isna(student3["hpi"])
        assert pd.isna(student3["social_hx"])
        assert pd.isna(student3["assessment"])
        assert pd.isna(student3["plan"])

    def test_student_ids_are_ints(self, doc_responses_xlsx):
        df = _load_documentation_responses(doc_responses_xlsx)
        assert list(df["student_id"]) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Tests: load_documentation_scores
# ---------------------------------------------------------------------------


class TestLoadDocumentationScores:
    """Test that scores loading derives student ID from position."""

    def test_derives_student_id_from_position(self, doc_scores_xlsx):
        scores = _load_documentation_scores(doc_scores_xlsx)
        assert 1 in scores
        assert 2 in scores
        assert 3 in scores
        assert 4 in scores

    def test_none_for_nonsubmitters(self, doc_scores_xlsx):
        scores = _load_documentation_scores(doc_scores_xlsx)
        s3 = scores[3]
        assert s3["social_hx"] is None
        assert s3["assessment"] is None
        assert s3["plan"] is None
        # But HPI and Summary Statement have partial scores
        assert s3["hpi"] == 1.0
        assert s3["summary_statement"] == 1.0

    def test_total_score_values(self, doc_scores_xlsx):
        scores = _load_documentation_scores(doc_scores_xlsx)
        assert scores[1]["total_score"] == 22.5
        assert scores[2]["total_score"] == 19.5
        assert scores[3]["total_score"] == 2.0

    def test_milestone_preserved_as_string(self, doc_scores_xlsx):
        scores = _load_documentation_scores(doc_scores_xlsx)
        assert scores[3]["total_milestone"] == "Behavior requiring corrective response"
        assert scores[1]["total_milestone"] == "Advanced Developing to Aspirational"

    def test_half_point_scores(self, doc_scores_xlsx):
        scores = _load_documentation_scores(doc_scores_xlsx)
        assert scores[1]["social_hx"] == 3.5
        assert scores[2]["social_hx"] == 4.5


# ---------------------------------------------------------------------------
# Tests: milestone derivation
# ---------------------------------------------------------------------------


class TestDocumentationMilestones:
    """Test milestone thresholds derived from observed data."""

    def test_entry(self):
        assert derive_documentation_milestone(2.0) == "Entry"

    def test_early_developing(self):
        assert derive_documentation_milestone(6.0) == "Early Developing"

    def test_mid_developing(self):
        assert derive_documentation_milestone(14.0) == "Mid-Developing"

    def test_mid_to_advanced(self):
        assert derive_documentation_milestone(19.0) == "Mid-Developing to Advanced Developing"

    def test_advanced_developing(self):
        assert derive_documentation_milestone(20.0) == "Advanced Developing"

    def test_advanced_to_aspirational(self):
        assert derive_documentation_milestone(23.0) == "Advanced Developing to Aspirational"

    def test_aspirational(self):
        assert derive_documentation_milestone(25.0) == "Aspirational"


# ---------------------------------------------------------------------------
# Tests: build_output_df
# ---------------------------------------------------------------------------


class TestDocumentationBuildOutputDf:
    """Test output DataFrame column structure."""

    def _make_result(self, with_faculty=False):
        at = KPSOMDocumentationType()
        sections = at.get_sections()
        scores = {"hpi": 4.0, "social_hx": 3.0, "summary_statement": 4.0,
                  "assessment": 3.0, "plan": 5.0}
        explanations = {s: f"Explanation for {s}" for s in sections}
        fac = None
        if with_faculty:
            fac = {
                "hpi": 3.5, "social_hx": 3.0, "summary_statement": 3.0,
                "assessment": 4.0, "plan": 4.5, "org_lang": 3.5,
                "pcig_total": 6.5, "pcdp_total": 11.5,
                "pcdo_score": 3.5, "total_score": 21.5,
                "total_milestone": "Advanced Developing",
            }
        return GradingResult(
            student_id="1",
            section_scores=scores,
            section_explanations=explanations,
            total_score=sum(scores.values()),
            faculty_scores=fac,
        )

    def test_columns_without_faculty(self):
        at = KPSOMDocumentationType()
        result = self._make_result(with_faculty=False)
        df = at.build_output_df([result])

        assert "student_id" in df.columns
        assert "hpi_ai_score" in df.columns
        assert "ai_pcig_total" in df.columns
        assert "ai_pcdp_total" in df.columns
        assert "ai_total" in df.columns
        assert "ai_total_milestone" in df.columns
        assert "faculty_total" not in df.columns

    def test_columns_with_faculty(self):
        at = KPSOMDocumentationType()
        result = self._make_result(with_faculty=True)
        df = at.build_output_df([result])

        assert "org_lang_faculty_score" in df.columns
        assert "faculty_pcig_total" in df.columns
        assert "faculty_pcdp_total" in df.columns
        assert "faculty_pcdo_score" in df.columns
        assert "faculty_total" in df.columns
        assert "total_delta" in df.columns
        assert "hpi_delta" in df.columns

    def test_domain_totals_computed(self):
        at = KPSOMDocumentationType()
        result = self._make_result(with_faculty=False)
        df = at.build_output_df([result])

        assert df.iloc[0]["ai_pcig_total"] == 7.0   # hpi(4) + social_hx(3)
        assert df.iloc[0]["ai_pcdp_total"] == 12.0   # sum(4) + assess(3) + plan(5)
        assert df.iloc[0]["ai_total"] == 19.0


# ---------------------------------------------------------------------------
# Tests: Interface
# ---------------------------------------------------------------------------


class TestDocumentationInterface:
    """Test that the type implements the interface correctly."""

    def test_sections(self):
        at = KPSOMDocumentationType()
        sections = at.get_sections()
        assert sections == ["hpi", "social_hx", "summary_statement", "assessment", "plan"]

    def test_required_files(self):
        at = KPSOMDocumentationType()
        files = at.get_required_files()
        keys = [f["key"] for f in files]
        assert "rubric" in keys
        assert "responses" in keys
        assert "scores" in keys

    def test_parse_llm_response_clamps(self):
        at = KPSOMDocumentationType()
        text = "Excellent.\n7"
        _, score = at.parse_llm_response(text, "hpi")
        assert score == 5.0  # clamped to max

    def test_parse_llm_response_floors(self):
        at = KPSOMDocumentationType()
        text = "Poor.\n0"
        _, score = at.parse_llm_response(text, "hpi")
        assert score == 1.0  # clamped to min
