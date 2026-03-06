"""UK-style OSCE assessment type — thin adapter over existing grader.py logic.

This wraps the existing UK-format grading pipeline so it participates in the
AssessmentType interface without duplicating any logic.
"""

from __future__ import annotations

import pandas as pd

import config
from grader import extract_score, read_rubric_and_key, _read_excel_safe
from .base import AssessmentType, GradingResult


class UKOSCEType(AssessmentType):
    """Standard UK-style OSCE with post-encounter notes."""

    name = "Standard OSCE (Post-Encounter Notes)"
    type_id = "uk_osce"

    _SECTIONS = ["hpi", "pex", "sum", "ddx", "support", "plan"]

    def get_sections(self) -> list[str]:
        return list(self._SECTIONS)

    def get_required_files(self) -> list[dict]:
        return [
            {
                "key": "rubric",
                "label": "Rubric (.xlsx)",
                "types": ["xlsx"],
                "required": True,
            },
            {
                "key": "answer_key",
                "label": "Answer Key (.xlsx)",
                "types": ["xlsx"],
                "required": True,
            },
            {
                "key": "responses",
                "label": "Student Notes (.xlsx)",
                "types": ["xlsx"],
                "required": True,
            },
        ]

    def load_inputs(self, **file_paths) -> tuple[pd.DataFrame, dict]:
        """Load rubric, answer key, and student notes.

        Returns ``(student_df, rubric_data)`` where rubric_data contains
        ``'rubric_content'`` and ``'answer_key_content'`` dicts.
        """
        rubric_path = file_paths["rubric"]
        answer_key_path = file_paths["answer_key"]
        responses_path = file_paths["responses"]

        rubric_content, answer_key_content = read_rubric_and_key(
            rubric_path, answer_key_path
        )
        df = _read_excel_safe(responses_path, "student notes")

        rubric_data = {
            "rubric_content": rubric_content,
            "answer_key_content": answer_key_content,
        }
        return df, rubric_data

    def build_grading_prompt(self, section: str, rubric_data: dict) -> str:
        return config.GRADING_PROMPT

    def build_user_message(
        self,
        section: str,
        student_response: str,
        rubric_data: dict,
    ) -> str:
        rubric_content = rubric_data["rubric_content"]
        answer_key_content = rubric_data["answer_key_content"]

        rubric_text = rubric_content.get(
            section.lower(), "No rubric available for this section."
        )
        answer_key_text = answer_key_content.get(section.lower(), "")

        return (
            f"Refer to the rubric: {rubric_text}.\n"
            f"Here is the answer key for {section}: {answer_key_text}.\n"
            f"Please evaluate the following {section} and provide a "
            f"score: {student_response}"
        )

    def parse_llm_response(
        self, response: str, section: str
    ) -> tuple[str, float | None]:
        score = extract_score(response)
        return response, float(score) if score is not None else None

    def build_output_df(self, results: list[GradingResult]) -> pd.DataFrame:
        rows = []
        for r in results:
            row: dict = {"student_id": r.student_id}
            for sec in self._SECTIONS:
                row[f"{sec}_gpt_score"] = r.section_scores.get(sec)
                row[f"{sec}_gpt_explanation"] = r.section_explanations.get(sec, "")
            rows.append(row)
        return pd.DataFrame(rows)
