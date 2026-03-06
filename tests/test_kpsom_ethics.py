"""Tests for KPSOM Ethics OSCE assessment type (Case 8).

Tests cover:
- Responses loading with Pro/Con concatenation
- "No PET submitted" filtering
- Scores loading with two header rows
- Milestone threshold derivation
- JSON response parsing for Q1, Q2, Q3
- build_output_df columns
"""

import json
import os
import sys

import pandas as pd
import pytest
from openpyxl import Workbook

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from assessment_types.kpsom_ethics import (
    KPSOMEthicsType,
    derive_ethics_milestone,
    _load_ethics_responses,
    _load_ethics_scores,
    _parse_q1_response,
    _parse_q2_response,
    _parse_q3_response,
    _concat_pro_con,
)
from assessment_types.base import GradingResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ethics_responses_xlsx(tmp_path):
    """Create a minimal Case 8 responses file."""
    wb = Workbook()
    ws = wb.active

    # Row 0: Full question text headers (15 columns)
    ws.append([
        "Student",
        "1. What is wrong with asking a student to consent a patient?",
        "Q2A Pro", "Q2A Con", "Q2A Combined",
        "Q2B Pro", "Q2B Con", "Q2B Combined",
        "Q2C Pro", "Q2C Con", "Q2C Combined",
        "Q3-1", "Q3-2", "Q3-3", "Q3-4",
    ])

    # Student 1: full submission
    ws.append([
        1,
        "Students lack training in consent process.",
        "Quick resolution", "May not be legal",
        "Quick resolution; May not be legal",
        "Assesses capacity", "Time consuming",
        "Assesses capacity; Time consuming",
        "Shifts responsibility", "May not resolve",
        "Shifts responsibility; May not resolve",
        "Can you tell me what procedure is planned?",
        "What are the risks?",
        "Are there other options?",
        "Why do you want this procedure?",
    ])

    # Student 2: full submission
    ws.append([
        2,
        "It is outside scope of practice for students.",
        "Family involved", "Daughter not patient",
        "Family involved; Daughter not patient",
        "Direct assessment", "Patient may refuse",
        "Direct assessment; Patient may refuse",
        "Experienced help", "Avoids responsibility",
        "Experienced help; Avoids responsibility",
        "What do you understand about your condition?",
        "What procedure is being suggested?",
        "What could go wrong?",
        "Do you have questions?",
    ])

    # Student 3: non-submitter
    ws.append([
        3,
        "No PET submitted",
        None, None, None,
        None, None, None,
        None, None, None,
        None, None, None, None,
    ])

    # Student 4: another non-submitter (empty)
    ws.append([
        4,
        None,
        None, None, None,
        None, None, None,
        None, None, None,
        None, None, None, None,
    ])

    # Student 5: full submission
    ws.append([
        5,
        "Consent requires physician-level understanding.",
        "Practical", "Ethically questionable",
        "Practical; Ethically questionable",
        "Proper process", "Stressful for patient",
        "Proper process; Stressful for patient",
        "Delegation", "Still unresolved",
        "Delegation; Still unresolved",
        "Do you know why you are here?",
        "Has anyone explained the surgery?",
        "What would happen if you decline?",
        "Are you comfortable proceeding?",
    ])

    path = str(tmp_path / "ethics_responses.xlsx")
    wb.save(path)
    return path


@pytest.fixture
def ethics_scores_xlsx(tmp_path):
    """Create a minimal Case 8 scores file."""
    wb = Workbook()
    ws = wb.active

    # Row 0: Section group labels (20 columns)
    ws.append([
        None,
        "Question 1 (4 points)",
        None, None, None,
        "Question 2 (6 points)",
        None, None, None, None, None,
        "Q2 Total",
        "Question 3 (8 points)",
        None, None, None,
        "Q3 Total",
        "Total Task Score",
        "Milestone",
        "Comments",
    ])

    # Row 1: Sub-item labels
    ws.append([
        None,  # student ID
        None,  # Q1 text embedded
        "Q1 Problematic",
        "Q1 Reasonable",
        "Q1 Total",
        "Q2A Option text",
        "Q2A Score",
        "Q2B Option text",
        "Q2B Score",
        "Q2C Option text",
        "Q2C Score",
        None,  # Q2 total
        "Enter question here:",
        "Enter question here:",
        "Enter question here:",
        "Enter question here:",
        None,  # Q3 total
        None,  # task total
        None,  # milestone
        None,  # comments
    ])

    # Student 1 data (row 2)
    ws.append([
        1,
        "Students lack training...",
        2, 1, 3,
        None, 2,
        None, 1,
        None, 2,
        5,
        "Can you tell me...", "What are the risks?",
        "Other options?", "Why do you want?",
        6,
        14,
        "Advanced Developing",
        "Good ethical reasoning",
    ])

    # Student 2 data
    ws.append([
        2,
        "Outside scope...",
        1, 2, 3,
        None, 1,
        None, 2,
        None, 1,
        4,
        "What do you understand...", "What procedure?",
        "What could go wrong?", "Do you have questions?",
        8,
        15,
        "Advanced Developing",
        "Strong capacity questions",
    ])

    # Student 5 data
    ws.append([
        5,
        "Consent requires physician...",
        2, 0, 2,
        None, 2,
        None, 1,
        None, 0,
        3,
        "Do you know why...", "Has anyone explained?",
        "Decline?", "Comfortable?",
        4,
        9,
        "Early Developing to Mid-Developing",
        "Needs improvement on Q3",
    ])

    path = str(tmp_path / "ethics_scores.xlsx")
    wb.save(path)
    return path


# ---------------------------------------------------------------------------
# Tests: load_ethics_responses
# ---------------------------------------------------------------------------


class TestLoadEthicsResponses:
    """Test responses loading with Pro/Con concatenation."""

    def test_concatenates_pro_con(self, ethics_responses_xlsx):
        df = _load_ethics_responses(ethics_responses_xlsx)
        s1 = df[df["student_id"] == 1].iloc[0]
        assert "Pro:" in s1["q2a"]
        assert "Con:" in s1["q2a"]
        assert "Quick resolution" in s1["q2a"]
        assert "May not be legal" in s1["q2a"]

    def test_filters_no_pet_submitted(self, ethics_responses_xlsx):
        df = _load_ethics_responses(ethics_responses_xlsx)
        # Students 3 and 4 are non-submitters
        s3 = df[df["student_id"] == 3].iloc[0]
        assert not s3["_is_submitter"]
        assert pd.isna(s3["q1"])

        s4 = df[df["student_id"] == 4].iloc[0]
        assert not s4["_is_submitter"]

    def test_q3_concatenated_as_list(self, ethics_responses_xlsx):
        df = _load_ethics_responses(ethics_responses_xlsx)
        s1 = df[df["student_id"] == 1].iloc[0]
        assert "1." in s1["q3"]
        assert "2." in s1["q3"]
        assert "3." in s1["q3"]
        assert "4." in s1["q3"]

    def test_all_students_present(self, ethics_responses_xlsx):
        df = _load_ethics_responses(ethics_responses_xlsx)
        assert set(df["student_id"].tolist()) == {1, 2, 3, 4, 5}


# ---------------------------------------------------------------------------
# Tests: load_ethics_scores
# ---------------------------------------------------------------------------


class TestLoadEthicsScores:
    """Test scores loading with two header rows."""

    def test_uses_two_header_rows(self, ethics_scores_xlsx):
        scores = _load_ethics_scores(ethics_scores_xlsx)
        # Student data starts at row 2 (after 2 header rows)
        assert 1 in scores
        assert 2 in scores
        assert 5 in scores

    def test_student_id_from_col0(self, ethics_scores_xlsx):
        scores = _load_ethics_scores(ethics_scores_xlsx)
        assert len(scores) == 3  # students 1, 2, 5

    def test_sub_scores_extracted(self, ethics_scores_xlsx):
        scores = _load_ethics_scores(ethics_scores_xlsx)
        s1 = scores[1]
        assert s1["q1_problematic"] == 2.0
        assert s1["q1_reasonable"] == 1.0
        assert s1["q1_total"] == 3.0
        assert s1["q2a_score"] == 2.0
        assert s1["q2b_score"] == 1.0
        assert s1["q2c_score"] == 2.0
        assert s1["q3_total"] == 6.0
        assert s1["task_total"] == 14.0

    def test_milestone_and_comments(self, ethics_scores_xlsx):
        scores = _load_ethics_scores(ethics_scores_xlsx)
        assert scores[1]["milestone"] == "Advanced Developing"
        assert scores[1]["comments"] == "Good ethical reasoning"


# ---------------------------------------------------------------------------
# Tests: milestone derivation
# ---------------------------------------------------------------------------


class TestEthicsMilestones:
    """Test milestone thresholds against rubric scoring table."""

    def test_entry(self):
        assert derive_ethics_milestone(2.0) == "Entry"

    def test_entry_to_early(self):
        assert derive_ethics_milestone(5.5) == "Entry to Early Developing"

    def test_early_developing(self):
        assert derive_ethics_milestone(7.0) == "Early Developing"

    def test_early_to_mid(self):
        assert derive_ethics_milestone(9.0) == "Early Developing to Mid-Developing"

    def test_mid_developing(self):
        assert derive_ethics_milestone(10.5) == "Mid-Developing"
        assert derive_ethics_milestone(11.5) == "Mid-Developing"

    def test_mid_to_advanced(self):
        assert derive_ethics_milestone(12.5) == "Mid-Developing to Advanced Developing"

    def test_advanced(self):
        assert derive_ethics_milestone(14.5) == "Advanced Developing"

    def test_advanced_to_aspirational(self):
        assert derive_ethics_milestone(15.5) == "Advanced Developing to Aspirational"

    def test_aspirational(self):
        assert derive_ethics_milestone(17.0) == "Aspirational"
        assert derive_ethics_milestone(18.0) == "Aspirational"


# ---------------------------------------------------------------------------
# Tests: JSON response parsing
# ---------------------------------------------------------------------------


class TestQ1Parsing:
    """Test Q1 dual sub-score JSON parsing."""

    def test_valid_json(self):
        response = '{"q1_problematic_score": 2, "q1_reasonable_score": 1, "rationale": "Good."}'
        explanation, score = _parse_q1_response(response)
        assert score == 3.0
        parsed = json.loads(explanation)
        assert parsed["q1_problematic_score"] == 2.0
        assert parsed["q1_reasonable_score"] == 1.0

    def test_clamped_scores(self):
        response = '{"q1_problematic_score": 5, "q1_reasonable_score": 3, "rationale": "N/A"}'
        explanation, score = _parse_q1_response(response)
        assert score == 4.0  # 2 + 2 (both clamped)

    def test_with_code_fences(self):
        response = '```json\n{"q1_problematic_score": 1, "q1_reasonable_score": 2, "rationale": "OK"}\n```'
        explanation, score = _parse_q1_response(response)
        assert score == 3.0


class TestQ2Parsing:
    """Test Q2 single score JSON parsing."""

    def test_valid_json(self):
        response = '{"score": 2, "rationale": "Both pro and con identified."}'
        explanation, score = _parse_q2_response(response)
        assert score == 2.0

    def test_clamped(self):
        response = '{"score": 5, "rationale": "N/A"}'
        _, score = _parse_q2_response(response)
        assert score == 2.0


class TestQ3Parsing:
    """Test Q3 per-question scores JSON parsing."""

    def test_valid_json(self):
        response = '{"scores": [2, 2, 1, 0], "elements_addressed": ["situation", "intervention", "alternatives", "orientation/other"], "rationale": "Good."}'
        explanation, score = _parse_q3_response(response)
        assert score == 5.0

    def test_max_8(self):
        response = '{"scores": [2, 2, 2, 2], "elements_addressed": ["a", "b", "c", "d"], "rationale": "Perfect."}'
        _, score = _parse_q3_response(response)
        assert score == 8.0

    def test_clamped_per_question(self):
        response = '{"scores": [3, 2, 2, 2], "elements_addressed": ["a", "b", "c", "d"], "rationale": "N/A"}'
        _, score = _parse_q3_response(response)
        assert score == 8.0  # 2+2+2+2 after clamping first to 2


# ---------------------------------------------------------------------------
# Tests: build_output_df
# ---------------------------------------------------------------------------


class TestEthicsBuildOutputDf:
    """Test output DataFrame column structure."""

    def _make_result(self, with_faculty=False):
        q1_explanation = json.dumps({
            "q1_problematic_score": 2.0,
            "q1_reasonable_score": 1.0,
            "rationale": "Good",
        })
        fac = None
        if with_faculty:
            fac = {
                "q1_problematic": 2.0,
                "q1_reasonable": 1.0,
                "q1_total": 3.0,
                "q2a_score": 2.0,
                "q2b_score": 1.0,
                "q2c_score": 2.0,
                "q2_total": 5.0,
                "q3_total": 6.0,
                "task_total": 14.0,
                "milestone": "Advanced Developing",
                "comments": "Well done",
            }
        return GradingResult(
            student_id="1",
            section_scores={"q1": 3.0, "q2a": 2.0, "q2b": 1.0, "q2c": 2.0, "q3": 6.0},
            section_explanations={
                "q1": q1_explanation,
                "q2a": '{"score": 2}', "q2b": '{"score": 1}',
                "q2c": '{"score": 2}', "q3": '{"scores": [2,2,1,1]}',
            },
            total_score=14.0,
            faculty_scores=fac,
        )

    def test_columns_without_faculty(self):
        at = KPSOMEthicsType()
        result = self._make_result(with_faculty=False)
        df = at.build_output_df([result])

        assert "student_id" in df.columns
        assert "q1_ai_total" in df.columns
        assert "q1_ai_problematic" in df.columns
        assert "q1_ai_reasonable" in df.columns
        assert "q2a_ai_score" in df.columns
        assert "q2_ai_total" in df.columns
        assert "q3_ai_total" in df.columns
        assert "ai_total" in df.columns
        assert "ai_milestone" in df.columns
        assert "faculty_total" not in df.columns

    def test_columns_with_faculty(self):
        at = KPSOMEthicsType()
        result = self._make_result(with_faculty=True)
        df = at.build_output_df([result])

        assert "q1_faculty_total" in df.columns
        assert "q1_faculty_problematic" in df.columns
        assert "q2a_faculty_score" in df.columns
        assert "q2_faculty_total" in df.columns
        assert "q3_faculty_total" in df.columns
        assert "faculty_total" in df.columns
        assert "total_delta" in df.columns
        assert "q1_delta" in df.columns
        assert "q2_delta" in df.columns
        assert "q3_delta" in df.columns
        assert "faculty_comments" in df.columns

    def test_q2_total_computed(self):
        at = KPSOMEthicsType()
        result = self._make_result(with_faculty=False)
        df = at.build_output_df([result])
        assert df.iloc[0]["q2_ai_total"] == 5.0  # 2 + 1 + 2

    def test_q1_subscores_extracted(self):
        at = KPSOMEthicsType()
        result = self._make_result(with_faculty=False)
        df = at.build_output_df([result])
        assert df.iloc[0]["q1_ai_problematic"] == 2.0
        assert df.iloc[0]["q1_ai_reasonable"] == 1.0


# ---------------------------------------------------------------------------
# Tests: Helpers
# ---------------------------------------------------------------------------


class TestConcatProCon:
    """Test the pro/con concatenation helper."""

    def test_both_present(self):
        result = _concat_pro_con("Good thing", "Bad thing")
        assert result == "Pro: Good thing\nCon: Bad thing"

    def test_pro_only(self):
        result = _concat_pro_con("Good thing", None)
        assert result == "Pro: Good thing"

    def test_both_empty(self):
        result = _concat_pro_con(None, None)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Interface
# ---------------------------------------------------------------------------


class TestEthicsInterface:
    """Test that the type implements the interface correctly."""

    def test_sections(self):
        at = KPSOMEthicsType()
        assert at.get_sections() == ["q1", "q2a", "q2b", "q2c", "q3"]

    def test_required_files(self):
        at = KPSOMEthicsType()
        files = at.get_required_files()
        keys = [f["key"] for f in files]
        assert "rubric" in keys
        assert "responses" in keys
        assert "scores" in keys
