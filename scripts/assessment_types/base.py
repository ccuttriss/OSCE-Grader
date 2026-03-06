"""Abstract base class for assessment types.

Each assessment type defines its own file format expectations, ingestion logic,
section definitions, grading prompts, and output schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class GradingResult:
    """Container for one student's grading results."""

    student_id: str
    section_scores: dict[str, float | None] = field(default_factory=dict)
    section_explanations: dict[str, str] = field(default_factory=dict)
    total_score: float | None = None
    milestone: str | None = None
    faculty_scores: dict[str, float] | None = None


class AssessmentType(ABC):
    """Interface that every assessment type must implement."""

    name: str       # display name, e.g. "KPSOM OSCE (I-PASS Handoff)"
    type_id: str    # slug, e.g. "kpsom_ipass"

    @abstractmethod
    def load_inputs(self, **file_paths) -> tuple[pd.DataFrame, dict]:
        """Load and validate all input files for this assessment type.

        Returns ``(student_responses_df, rubric_data)``.
        Raises ``ValueError`` with a descriptive message on bad input.
        """

    @abstractmethod
    def get_sections(self) -> list[str]:
        """Return the ordered list of section identifiers to grade."""

    @abstractmethod
    def build_grading_prompt(self, section: str, rubric_data: dict) -> str:
        """Return the system prompt for grading a specific section."""

    @abstractmethod
    def build_user_message(
        self,
        section: str,
        student_response: str,
        rubric_data: dict,
    ) -> str:
        """Return the user turn content for one section of one student."""

    @abstractmethod
    def parse_llm_response(
        self, response: str, section: str
    ) -> tuple[str, float | None]:
        """Parse the LLM response for a section.

        Returns ``(explanation_text, numeric_score)``.
        Score may be ``None`` if parsing fails.
        """

    @abstractmethod
    def build_output_df(self, results: list[GradingResult]) -> pd.DataFrame:
        """Convert a list of GradingResult objects into the output DataFrame."""

    def get_required_files(self) -> list[dict]:
        """Return metadata about required file uploads for the Streamlit UI.

        Each dict has keys: ``key``, ``label``, ``types`` (list of extensions),
        ``required`` (bool).
        """
        raise NotImplementedError
