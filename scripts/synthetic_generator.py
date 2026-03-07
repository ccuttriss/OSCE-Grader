"""Synthetic OSCE Data Generator — LLM agent-based simulation engine.

Generates realistic end-to-end OSCE data (rubrics, student notes, faculty
scores) using independent LLM agents with unique personas for students and
faculty.  Can optionally ground generation in uploaded de-identified examples.

The key design principle is **independence**: each student agent has a unique
persona (cultural background, academic level, writing style) and each faculty
agent has a unique rater persona (experience level, scoring tendency, specialty
focus).  This prevents clustering bias and produces data that mirrors the
realistic spread seen in actual OSCE administrations.
"""

from __future__ import annotations

import io
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from openpyxl import Workbook

logger = logging.getLogger("osce_grader.synthetic")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SyntheticRubric:
    """A generated rubric with per-section criteria and score level descriptors."""
    assessment_type_id: str
    case_title: str
    case_description: str
    sections: dict[str, SectionRubric]


@dataclass
class SectionRubric:
    """Rubric for a single section with score-level descriptors."""
    section_key: str
    display_name: str
    max_score: int | float
    criteria: str
    score_levels: dict[int, str]  # score -> descriptor text


@dataclass
class StudentPersona:
    """A synthetic student with demographic and academic characteristics."""
    student_id: int
    name: str
    background: str
    academic_level: str
    writing_style: str
    clinical_strengths: list[str]
    clinical_weaknesses: list[str]


@dataclass
class FacultyPersona:
    """A synthetic faculty rater with scoring tendencies."""
    faculty_id: str
    name: str
    specialty: str
    years_experience: int
    scoring_tendency: str  # lenient / moderate / strict
    focus_areas: list[str]
    background_note: str


@dataclass
class SyntheticSession:
    """A fully generated session: rubric + student notes + faculty scores."""
    label: str
    assessment_type_id: str
    rubric: SyntheticRubric
    faculty: FacultyPersona
    students: list[StudentPersona]
    student_notes: dict[int, dict[str, str]]     # student_id -> {section: text}
    faculty_scores: dict[int, dict[str, float]]   # student_id -> {section: score}
    sections: list[str]


# ---------------------------------------------------------------------------
# Type metadata
# ---------------------------------------------------------------------------

_TYPE_META: dict[str, dict] = {
    "uk_osce": {
        "name": "Standard OSCE (Post-Encounter Notes)",
        "sections": {
            "hpi": ("History of Present Illness", 5),
            "pex": ("Physical Examination", 5),
            "sum": ("Summary Statement", 5),
            "ddx": ("Differential Diagnosis", 5),
            "support": ("Supporting Evidence", 5),
            "plan": ("Diagnostic Workup / Plan", 5),
        },
        "case_context": "post-encounter note for a standardized patient case",
    },
    "kpsom_ipass": {
        "name": "KPSOM I-PASS Handoff",
        "sections": {
            "illness_severity": ("Illness Severity", 2),
            "patient_summary": ("Patient Summary", 14),
            "action_list": ("Action List", 5),
            "situation_awareness": ("Situation Awareness & Contingency Planning", 3),
            "organization": ("Organization", 3),
        },
        "case_context": "I-PASS handoff note for a clinical patient encounter",
    },
    "kpsom_documentation": {
        "name": "KPSOM Clinical Documentation",
        "sections": {
            "hpi": ("History of Present Illness", 5),
            "social_hx": ("Social History", 5),
            "summary_statement": ("Summary Statement", 5),
            "assessment": ("Assessment", 5),
            "plan": ("Plan", 5),
            "org_lang": ("Organization & Language", 4),
        },
        "case_context": "clinical documentation note following a patient encounter",
    },
    "kpsom_ethics": {
        "name": "KPSOM Ethics Open-Ended Questions",
        "sections": {
            "q1_total": ("Q1: Ethical Issue Identification", 4),
            "q2a_score": ("Q2A: Option Analysis - First Option", 4),
            "q2b_score": ("Q2B: Option Analysis - Second Option", 4),
            "q2c_score": ("Q2C: Option Analysis - Third Option", 4),
            "q3_total": ("Q3: Capacity Assessment Questions", 8),
        },
        "case_context": "ethics case analysis with open-ended question responses",
    },
}


# ---------------------------------------------------------------------------
# Persona generators
# ---------------------------------------------------------------------------

# Diverse backgrounds for student persona generation
_STUDENT_BACKGROUNDS = [
    "First-generation college student from a rural community in the Southern US",
    "International student from South Korea who completed pre-med in the US",
    "Career changer who previously worked as a nurse for 8 years",
    "Student from an urban underserved community with strong community health experience",
    "Second-generation immigrant from a Mexican-American family in Texas",
    "Student who grew up in West Africa and moved to the US for college",
    "Military veteran who served as a combat medic before medical school",
    "Student from an East Asian immigrant family with a research-heavy background",
    "First-generation medical student from an Appalachian community",
    "Student from a Native American reservation with IHS clinical exposure",
    "Student who grew up in the Middle East and completed undergrad in the UK",
    "Nontraditional student who worked in public health policy before med school",
    "Student from a South Asian immigrant family with dual cultural perspectives",
    "Student from a Caribbean island nation who completed pre-med in Canada",
    "First-generation Haitian-American student from an urban center",
    "Student with a PhD in biomedical engineering transitioning to clinical medicine",
    "Student from a Pacific Islander community with community health focus",
    "Eastern European immigrant student who completed undergrad in biochemistry",
    "Student from a multiracial family with extensive global health volunteer work",
    "Student who grew up in a refugee community and advocates for health equity",
]

_ACADEMIC_LEVELS = [
    "high-performing — thorough, organized, rarely misses key findings",
    "solid mid-range — covers most key points but occasionally misses nuances",
    "developing — captures main ideas but struggles with completeness and organization",
    "inconsistent — excels in some areas but has notable gaps in others",
    "strong clinically but weaker in written communication",
]

_WRITING_STYLES = [
    "concise and clinical — uses medical terminology precisely",
    "narrative and thorough — writes detailed descriptions, sometimes too verbose",
    "bullet-point oriented — structured but may lack connecting reasoning",
    "conversational tone — explains concepts clearly but sometimes informally",
    "terse and abbreviated — gets the key points across in minimal words",
]

_FACULTY_SPECIALTIES = [
    "Internal Medicine",
    "Family Medicine",
    "Emergency Medicine",
    "Pediatrics",
    "Surgery",
    "Psychiatry",
    "OB/GYN",
    "Neurology",
]


def _generate_student_personas(
    n_students: int,
    rng: np.random.Generator,
) -> list[StudentPersona]:
    """Generate diverse student personas."""
    personas = []
    backgrounds = list(_STUDENT_BACKGROUNDS)
    rng.shuffle(backgrounds)

    strengths_pool = [
        "history-taking", "physical exam documentation", "clinical reasoning",
        "differential diagnosis", "treatment planning", "patient communication",
        "organization", "summary writing", "ethical analysis", "evidence synthesis",
    ]
    weaknesses_pool = [
        "time management under pressure", "conciseness in writing",
        "including supporting evidence", "prioritizing differentials",
        "documenting negative findings", "structured organization",
        "specificity in plans", "social history completeness",
    ]

    for i in range(n_students):
        bg = backgrounds[i % len(backgrounds)]
        level = _ACADEMIC_LEVELS[int(rng.choice(len(_ACADEMIC_LEVELS)))]
        style = _WRITING_STYLES[int(rng.choice(len(_WRITING_STYLES)))]

        n_str = rng.integers(1, 4)
        n_wk = rng.integers(1, 3)
        str_indices = rng.choice(len(strengths_pool), size=n_str, replace=False)
        wk_indices = rng.choice(len(weaknesses_pool), size=n_wk, replace=False)

        personas.append(StudentPersona(
            student_id=i + 1,
            name=f"Student_{i + 1}",
            background=bg,
            academic_level=level,
            writing_style=style,
            clinical_strengths=[strengths_pool[j] for j in str_indices],
            clinical_weaknesses=[weaknesses_pool[j] for j in wk_indices],
        ))

    return personas


def _generate_faculty_persona(
    session_index: int,
    scoring_tendency: str,
    rng: np.random.Generator,
) -> FacultyPersona:
    """Generate a unique faculty rater persona for one session/year."""
    specialty = _FACULTY_SPECIALTIES[session_index % len(_FACULTY_SPECIALTIES)]
    years = int(rng.integers(3, 30))

    tendency_descriptions = {
        "lenient": "tends to give benefit of the doubt; focuses on what students got right",
        "moderate": "balanced scorer; applies rubric criteria consistently",
        "strict": "rigorous scorer; expects precise clinical language and completeness",
    }

    focus_areas_pool = [
        "clinical accuracy", "completeness of documentation",
        "organized presentation", "appropriate medical terminology",
        "evidence-based reasoning", "patient safety considerations",
        "communication clarity", "differential reasoning depth",
    ]
    n_focus = rng.integers(2, 4)
    focus_idx = rng.choice(len(focus_areas_pool), size=n_focus, replace=False)

    return FacultyPersona(
        faculty_id=f"faculty_{session_index + 1}",
        name=f"Dr. Faculty_{session_index + 1}",
        specialty=specialty,
        years_experience=years,
        scoring_tendency=scoring_tendency,
        focus_areas=[focus_areas_pool[j] for j in focus_idx],
        background_note=tendency_descriptions.get(scoring_tendency, ""),
    )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_rubric_generation_prompt(
    type_id: str,
    example_rubric_text: str | None = None,
) -> list[dict[str, str]]:
    """Build prompt for generating a detailed rubric."""
    meta = _TYPE_META[type_id]
    sections_desc = "\n".join(
        f"- **{display}** (max {max_s} points)"
        for _key, (display, max_s) in meta["sections"].items()
    )

    example_block = ""
    if example_rubric_text:
        example_block = (
            "\n\nHere is a de-identified example rubric from a real administration. "
            "Use this as a template for tone, detail level, and scoring criteria structure. "
            "Generate NEW content — do NOT copy verbatim:\n\n"
            f"--- EXAMPLE RUBRIC ---\n{example_rubric_text}\n--- END EXAMPLE ---\n"
        )

    system = (
        "You are a medical education assessment design expert. Generate a detailed, "
        "realistic OSCE grading rubric for the specified assessment type. The rubric "
        "must define clear, specific criteria for each score level so that different "
        "faculty members would arrive at similar scores for the same student response."
    )

    user = (
        f"Assessment type: {meta['name']}\n"
        f"Context: {meta['case_context']}\n\n"
        f"Sections to include:\n{sections_desc}\n"
        f"{example_block}\n"
        "Generate a complete rubric. For each section, provide:\n"
        "1. A brief description of what the section assesses\n"
        "2. Specific criteria at each score level (what distinguishes a 5 from a 4, "
        "a 3 from a 2, etc.)\n"
        "3. Key clinical concepts that must be present for each score level\n\n"
        "Also generate a brief clinical case vignette (2-3 sentences) that the "
        "student encountered.\n\n"
        "Respond with ONLY valid JSON:\n"
        "{\n"
        '  "case_title": "...",\n'
        '  "case_description": "2-3 sentence case vignette...",\n'
        '  "sections": {\n'
        '    "section_key": {\n'
        '      "criteria": "overall section description and grading criteria",\n'
        '      "score_levels": {"5": "descriptor...", "4": "...", ...}\n'
        "    }\n"
        "  }\n"
        "}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _build_student_note_prompt(
    type_id: str,
    rubric: SyntheticRubric,
    student: StudentPersona,
    example_notes: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Build prompt for a student agent to write their note."""
    meta = _TYPE_META[type_id]

    rubric_text = f"Case: {rubric.case_title}\n{rubric.case_description}\n\n"
    for sec_key, sec_rubric in rubric.sections.items():
        rubric_text += (
            f"## {sec_rubric.display_name} (max {sec_rubric.max_score} pts)\n"
            f"{sec_rubric.criteria}\n\n"
        )

    example_block = ""
    if example_notes:
        example_block = (
            "\n\nHere are de-identified example student responses from a real "
            "administration. Use these as a guide for the expected FORMAT, LENGTH, "
            "and LEVEL OF DETAIL — but write entirely NEW clinical content:\n\n"
        )
        for sec, text in example_notes.items():
            example_block += f"--- {sec} ---\n{text}\n\n"

    system = (
        "You are simulating a medical student completing an OSCE post-encounter "
        "writing task. You must write AS this specific student, reflecting their "
        "unique characteristics in how they approach the task.\n\n"
        f"YOUR PERSONA:\n"
        f"- Background: {student.background}\n"
        f"- Academic level: {student.academic_level}\n"
        f"- Writing style: {student.writing_style}\n"
        f"- Clinical strengths: {', '.join(student.clinical_strengths)}\n"
        f"- Clinical weaknesses: {', '.join(student.clinical_weaknesses)}\n\n"
        "IMPORTANT: Your response quality should reflect your academic level. "
        "If you are 'developing', your notes should realistically have gaps. "
        "If you are 'high-performing', they should be thorough but still natural — "
        "not robotically perfect. Write as a real student would."
    )

    user = (
        f"You are completing a {meta['name']} assessment.\n\n"
        f"RUBRIC AND CASE:\n{rubric_text}\n"
        f"{example_block}"
        "Write your response for EACH section. Respond with ONLY valid JSON:\n"
        "{\n"
    )
    for sec_key in meta["sections"]:
        user += f'  "{sec_key}": "your response text...",\n'
    user += "}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _build_faculty_scoring_prompt(
    type_id: str,
    rubric: SyntheticRubric,
    faculty: FacultyPersona,
    student_notes: dict[str, str],
    student_persona_summary: str,
    example_scores: dict[str, float] | None = None,
) -> list[dict[str, str]]:
    """Build prompt for a faculty agent to score a student's work."""
    meta = _TYPE_META[type_id]

    rubric_text = ""
    for sec_key, sec_rubric in rubric.sections.items():
        rubric_text += (
            f"## {sec_rubric.display_name} (max {sec_rubric.max_score} pts)\n"
            f"{sec_rubric.criteria}\n"
            f"Score levels:\n"
        )
        for level, desc in sorted(sec_rubric.score_levels.items(), key=lambda x: int(x[0]), reverse=True):
            rubric_text += f"  {level}: {desc}\n"
        rubric_text += "\n"

    notes_text = ""
    for sec_key, text in student_notes.items():
        display = meta["sections"].get(sec_key, (sec_key, 0))[0]
        notes_text += f"## {display}\n{text}\n\n"

    example_block = ""
    if example_scores:
        example_block = (
            "\nFor calibration reference, here are scores from a prior administration "
            "for a student of similar performance level (use as a general anchor, not "
            "an exact guide):\n"
        )
        for sec, score in example_scores.items():
            example_block += f"  {sec}: {score}\n"

    system = (
        "You are a medical faculty member grading a student's OSCE response. "
        "Score each section independently using the rubric criteria.\n\n"
        f"YOUR RATER PROFILE:\n"
        f"- Name: {faculty.name}\n"
        f"- Specialty: {faculty.specialty}\n"
        f"- Experience: {faculty.years_experience} years of clinical teaching\n"
        f"- Scoring tendency: {faculty.scoring_tendency} — {faculty.background_note}\n"
        f"- Focus areas: {', '.join(faculty.focus_areas)}\n\n"
        "IMPORTANT: Score realistically as this faculty member would. Your scoring "
        "tendency should subtly influence your scores — a lenient rater gives more "
        "benefit of the doubt, a strict rater expects precise completeness. "
        "But still follow the rubric — your tendency is a ~0.5 point bias, not a "
        "wholesale departure from the criteria."
    )

    user = (
        f"RUBRIC:\n{rubric_text}\n"
        f"STUDENT RESPONSE:\n{notes_text}\n"
        f"{example_block}\n"
        "Score each section. Respond with ONLY valid JSON — section keys mapped "
        "to numeric scores:\n"
        "{\n"
    )
    for sec_key, (_, max_s) in meta["sections"].items():
        user += f'  "{sec_key}": <score 0-{max_s}>,\n'
    user += "}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n", 1)
        text = lines[1] if len(lines) > 1 else ""
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _parse_json_response(text: str) -> dict:
    """Parse a JSON response from an LLM, handling common quirks."""
    text = _strip_fences(text)
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting JSON object from surrounding text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from LLM response:\n{text[:500]}")


def _parse_rubric_response(text: str, type_id: str) -> SyntheticRubric:
    """Parse LLM rubric generation response into a SyntheticRubric."""
    data = _parse_json_response(text)
    meta = _TYPE_META[type_id]

    sections = {}
    raw_sections = data.get("sections", {})
    for sec_key, (display, max_s) in meta["sections"].items():
        sec_data = raw_sections.get(sec_key, {})
        score_levels = {}
        raw_levels = sec_data.get("score_levels", {})
        for k, v in raw_levels.items():
            try:
                score_levels[int(k)] = str(v)
            except (ValueError, TypeError):
                pass

        sections[sec_key] = SectionRubric(
            section_key=sec_key,
            display_name=display,
            max_score=max_s,
            criteria=sec_data.get("criteria", ""),
            score_levels=score_levels,
        )

    return SyntheticRubric(
        assessment_type_id=type_id,
        case_title=data.get("case_title", "Synthetic Case"),
        case_description=data.get("case_description", ""),
        sections=sections,
    )


# ---------------------------------------------------------------------------
# Session generation orchestrator
# ---------------------------------------------------------------------------

def generate_synthetic_session(
    type_id: str,
    session_index: int,
    n_students: int,
    variability: float,
    llm_caller,
    *,
    temperature: float = 0.7,
    example_rubric_text: str | None = None,
    example_notes: dict[str, str] | None = None,
    example_scores: dict[int, dict[str, float]] | None = None,
    seed: int = 42,
    progress_callback=None,
) -> SyntheticSession:
    """Generate one complete synthetic session using LLM agents.

    Parameters
    ----------
    type_id : str
        Assessment type ID.
    session_index : int
        Index of this session (0-based), used for faculty persona variation.
    n_students : int
        Number of student agents to generate.
    variability : float
        0.0–1.0 controlling scoring tendency distribution and temperature.
    llm_caller : callable
        ``(messages, temperature, top_p) -> str`` LLM call function.
    temperature : float
        Base LLM temperature for generation.
    example_rubric_text : str, optional
        De-identified rubric text to ground generation.
    example_notes : dict, optional
        Example student notes ``{section: text}`` for grounding.
    example_scores : dict, optional
        Example faculty scores ``{student_id: {section: score}}`` for calibration.
    seed : int
        Random seed for persona generation.
    progress_callback : callable, optional
        ``(step_name: str, current: int, total: int) -> None``
    """
    meta = _TYPE_META.get(type_id)
    if meta is None:
        raise ValueError(f"Unknown assessment type: {type_id}")

    rng = np.random.default_rng(seed + session_index * 1000)
    sections = list(meta["sections"].keys())

    # Effective temperature scales with variability
    eff_temp = temperature * (0.5 + variability)

    # --- Step 1: Generate rubric ---
    if progress_callback:
        progress_callback("Generating rubric", 0, n_students + 2)

    rubric_messages = _build_rubric_generation_prompt(type_id, example_rubric_text)
    rubric_response = llm_caller(rubric_messages, eff_temp, 1.0)
    rubric = _parse_rubric_response(rubric_response, type_id)

    # --- Step 2: Generate faculty persona ---
    tendencies = ["lenient", "moderate", "strict"]
    # Variability affects the distribution: low var = mostly moderate
    if variability < 0.3:
        tendency = "moderate"
    elif variability < 0.7:
        tendency = tendencies[session_index % len(tendencies)]
    else:
        # High variability: more extreme tendencies
        weights = [0.35, 0.30, 0.35]
        tendency = rng.choice(tendencies, p=weights)

    faculty = _generate_faculty_persona(session_index, tendency, rng)

    # --- Step 3: Generate student personas ---
    students = _generate_student_personas(n_students, rng)

    # --- Step 4: Generate student notes (parallel) ---
    student_notes: dict[int, dict[str, str]] = {}
    # Pick one example student's scores for calibration reference if available
    example_score_ref = None
    if example_scores:
        first_key = next(iter(example_scores))
        example_score_ref = example_scores[first_key]

    if progress_callback:
        progress_callback("Generating student notes (parallel)", 1, n_students + 2)

    def _generate_one_note(student: StudentPersona) -> tuple[int, dict[str, str]]:
        note_messages = _build_student_note_prompt(
            type_id, rubric, student, example_notes
        )
        note_response = llm_caller(note_messages, eff_temp, 1.0)
        try:
            notes = _parse_json_response(note_response)
            for sec in sections:
                if sec not in notes:
                    notes[sec] = ""
            return student.student_id, {
                sec: str(notes.get(sec, "")) for sec in sections
            }
        except ValueError:
            logger.warning(
                "Failed to parse student %d notes, using empty.",
                student.student_id,
            )
            return student.student_id, {sec: "" for sec in sections}

    with ThreadPoolExecutor(max_workers=min(n_students, 8)) as pool:
        note_futures = {
            pool.submit(_generate_one_note, student): student
            for student in students
        }
        notes_done = 0
        for future in as_completed(note_futures):
            sid, parsed_notes = future.result()
            student_notes[sid] = parsed_notes
            notes_done += 1
            if progress_callback:
                progress_callback(
                    f"Student notes ({notes_done}/{n_students})",
                    notes_done,
                    n_students + 2,
                )

    # --- Step 5: Faculty scores all students (parallel) ---
    faculty_scores: dict[int, dict[str, float]] = {}

    if progress_callback:
        progress_callback("Scoring students (parallel)", n_students, n_students + 2)

    def _score_one_student(student: StudentPersona) -> tuple[int, dict[str, float]]:
        score_messages = _build_faculty_scoring_prompt(
            type_id,
            rubric,
            faculty,
            student_notes[student.student_id],
            f"Level: {student.academic_level}",
            example_score_ref,
        )
        score_response = llm_caller(score_messages, max(0.2, eff_temp * 0.5), 1.0)
        try:
            raw_scores = _parse_json_response(score_response)
            parsed: dict[str, float] = {}
            for sec in sections:
                val = raw_scores.get(sec)
                if val is not None:
                    try:
                        parsed[sec] = float(val)
                    except (ValueError, TypeError):
                        parsed[sec] = 0.0
                else:
                    parsed[sec] = 0.0
            return student.student_id, parsed
        except ValueError:
            logger.warning(
                "Failed to parse faculty scores for student %d, using zeros.",
                student.student_id,
            )
            return student.student_id, {sec: 0.0 for sec in sections}

    with ThreadPoolExecutor(max_workers=min(n_students, 8)) as pool:
        score_futures = {
            pool.submit(_score_one_student, student): student
            for student in students
        }
        scores_done = 0
        for future in as_completed(score_futures):
            sid, parsed_scores = future.result()
            faculty_scores[sid] = parsed_scores
            scores_done += 1
            if progress_callback:
                progress_callback(
                    f"Faculty scoring ({scores_done}/{n_students})",
                    n_students + 1,
                    n_students + 2,
                )

    label = f"{2020 + session_index} {'Spring' if session_index % 2 == 0 else 'Fall'}"

    return SyntheticSession(
        label=label,
        assessment_type_id=type_id,
        rubric=rubric,
        faculty=faculty,
        students=students,
        student_notes=student_notes,
        faculty_scores=faculty_scores,
        sections=sections,
    )


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def rubric_to_display_text(rubric: SyntheticRubric) -> str:
    """Format a rubric as readable text for UI display."""
    lines = [
        f"# {rubric.case_title}",
        f"\n{rubric.case_description}\n",
    ]
    for sec_key, sec in rubric.sections.items():
        lines.append(f"\n## {sec.display_name} (max {sec.max_score} pts)")
        lines.append(sec.criteria)
        if sec.score_levels:
            lines.append("\n**Score Levels:**\n")
            for level in sorted(sec.score_levels.keys(), reverse=True):
                lines.append(f"**{level}:** {sec.score_levels[level]}\n")
    return "\n".join(lines)


def rubric_to_excel(rubric: SyntheticRubric, type_id: str) -> bytes:
    """Export rubric as an Excel file matching the expected input format."""
    meta = _TYPE_META[type_id]
    wb = Workbook()
    ws = wb.active
    ws.title = "Rubric"

    if type_id == "uk_osce":
        # UK OSCE expects rubric.xlsx with section columns and one data row
        sections = list(meta["sections"].keys())
        ws.append(sections)
        row = []
        for sec in sections:
            sec_rubric = rubric.sections.get(sec)
            if sec_rubric:
                text = sec_rubric.criteria
                if sec_rubric.score_levels:
                    text += "\n" + "\n".join(
                        f"{lvl}) {desc}"
                        for lvl, desc in sorted(sec_rubric.score_levels.items(), reverse=True)
                    )
                row.append(text)
            else:
                row.append("")
        ws.append(row)
    else:
        # KPSOM types: .docx-style rubric exported as structured Excel
        ws.append(["Section", "Criteria", "Max Score", "Score Levels"])
        for sec_key, sec in rubric.sections.items():
            levels_text = "\n".join(
                f"{lvl}: {desc}"
                for lvl, desc in sorted(sec.score_levels.items(), reverse=True)
            )
            ws.append([sec.display_name, sec.criteria, sec.max_score, levels_text])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def answer_key_to_excel(rubric: SyntheticRubric, type_id: str) -> bytes:
    """Export an answer key derived from the rubric's top-level criteria."""
    meta = _TYPE_META[type_id]
    wb = Workbook()
    ws = wb.active
    ws.title = "Answer Key"

    sections = list(meta["sections"].keys())
    ws.append(sections)
    row = []
    for sec in sections:
        sec_rubric = rubric.sections.get(sec)
        if sec_rubric and sec_rubric.score_levels:
            # Use the highest score level as the answer key
            max_level = max(sec_rubric.score_levels.keys())
            row.append(sec_rubric.score_levels[max_level])
        else:
            row.append("")
    ws.append(row)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def student_notes_to_excel(session: SyntheticSession) -> bytes:
    """Export student notes as an Excel file matching the expected input format."""
    type_id = session.assessment_type_id
    meta = _TYPE_META[type_id]
    wb = Workbook()
    ws = wb.active
    ws.title = "Student Notes"

    sections = session.sections

    if type_id in ("kpsom_ipass", "kpsom_documentation"):
        # KPSOM format: row 0 = Q-numbers, row 1 = section headers, rows 2+ = data
        q_row = [""] + [f"Q{i+1}" for i in range(len(sections))]
        ws.append(q_row)
        header_row = ["Student"] + [
            meta["sections"][s][0] for s in sections
        ]
        ws.append(header_row)
        for sid in sorted(session.student_notes.keys()):
            row = [sid] + [session.student_notes[sid].get(s, "") for s in sections]
            ws.append(row)
    elif type_id == "kpsom_ethics":
        # Ethics: row 0 = question headers, rows 1+ = data
        header_row = ["Student"] + [
            meta["sections"][s][0] for s in sections
        ]
        ws.append(header_row)
        for sid in sorted(session.student_notes.keys()):
            row = [sid] + [session.student_notes[sid].get(s, "") for s in sections]
            ws.append(row)
    else:
        # UK OSCE: simple flat format
        headers = sections
        ws.append(headers)
        for sid in sorted(session.student_notes.keys()):
            row = [session.student_notes[sid].get(s, "") for s in sections]
            ws.append(row)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def faculty_scores_to_excel(session: SyntheticSession) -> bytes:
    """Export faculty scores matching the format load_faculty_session() expects."""
    # Reuse session_to_excel from gold_standard by constructing a SessionData
    from gold_standard import SessionData, session_to_excel

    sd = SessionData(
        label=session.label,
        assessment_type_id=session.assessment_type_id,
        sections=session.sections,
        scores=session.faculty_scores,
        student_count=len(session.faculty_scores),
    )
    return session_to_excel(sd)


def session_to_zip(session: SyntheticSession) -> bytes:
    """Bundle an entire session into a downloadable ZIP."""
    import zipfile

    buf = io.BytesIO()
    safe_label = session.label.replace(" ", "_")
    type_id = session.assessment_type_id

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Rubric
        zf.writestr(
            f"{safe_label}_rubric.xlsx",
            rubric_to_excel(session.rubric, type_id),
        )
        # Answer key (UK OSCE) or rubric detail
        if type_id == "uk_osce":
            zf.writestr(
                f"{safe_label}_answer_key.xlsx",
                answer_key_to_excel(session.rubric, type_id),
            )
        # Student notes
        zf.writestr(
            f"{safe_label}_student_notes.xlsx",
            student_notes_to_excel(session),
        )
        # Faculty scores
        zf.writestr(
            f"{safe_label}_faculty_scores.xlsx",
            faculty_scores_to_excel(session),
        )
        # Rubric as readable text
        zf.writestr(
            f"{safe_label}_rubric.txt",
            rubric_to_display_text(session.rubric),
        )

    return buf.getvalue()


def all_sessions_to_zip(sessions: list[SyntheticSession]) -> bytes:
    """Bundle multiple sessions into a single ZIP."""
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for session in sessions:
            safe_label = session.label.replace(" ", "_")
            type_id = session.assessment_type_id
            prefix = f"{safe_label}/"

            zf.writestr(
                f"{prefix}rubric.xlsx",
                rubric_to_excel(session.rubric, type_id),
            )
            if type_id == "uk_osce":
                zf.writestr(
                    f"{prefix}answer_key.xlsx",
                    answer_key_to_excel(session.rubric, type_id),
                )
            zf.writestr(
                f"{prefix}student_notes.xlsx",
                student_notes_to_excel(session),
            )
            zf.writestr(
                f"{prefix}faculty_scores.xlsx",
                faculty_scores_to_excel(session),
            )
            zf.writestr(
                f"{prefix}rubric.txt",
                rubric_to_display_text(session.rubric),
            )

    return buf.getvalue()
