# Implementation Plan: Closing the Gold Standard ↔ Grading Pipeline Gap

## Problem Statement

The OSCE-Grader has two completely disconnected systems:
1. **Gold Standard Pipeline** (Tab 5): Analyzes historical faculty data → produces benchmark artifacts (Excel/JSON), but these are never used for grading
2. **Grading Pipeline** (Tab 1): Grades student responses using only the parsed rubric text — no empirical score-level definitions, no benchmark constraints

**The gap**: Faculty consensus data that could define what "a 4 looks like" vs "a 2 looks like" is computed but never fed back into AI grading.

---

## Architecture: Two New Modules + Integration Points

### New Module 1: `scripts/faculty_analysis.py` — Deep Analysis of Past Faculty Grades

**Purpose**: Go beyond the current `gold_standard.py` (which only computes means/stds/CCT on session-level aggregates) to perform student-level, item-level psychometric analysis.

### New Module 2: `scripts/empirical_rubric.py` — LLM-Powered Empirical Score-Level Definition Generator

**Purpose**: Group student responses by faculty-assigned scores, use LLM to identify distinguishing features at each score level, and produce per-level definitions that feed directly into grading prompts.

---

## Phase 1: Enhanced Faculty Analysis (`faculty_analysis.py`)

### 1.1 Data Requirements

The current system loads faculty scores as `SessionData` (student_id → {section: score}). For deeper analysis, we need to also retain:
- Raw student responses (from the grading output files)
- Faculty comments (already partially loaded in handoff type)
- Student-level scores across multiple faculty raters (if available — currently each "session" is one faculty member's scores for one administration)

**Key insight**: The current `SessionData` represents one administration (e.g., "2020 Spring"). If multiple faculty graded the same students within one administration, that information is collapsed. We need to distinguish:
- **Cross-administration analysis** (what we have): Different cohorts graded in different years
- **Inter-rater analysis** (what we need): Multiple faculty grading the same students

We should support both. Inter-rater data requires a different upload format (multiple score files for the SAME student cohort).

### 1.2 Statistical Analyses to Implement

All implementable in Python — no SPSS/R needed:

| Analysis | Python Library | Notes |
|----------|---------------|-------|
| **Inter-rater reliability (ICC)** | `pingouin` | ICC(2,1) and ICC(3,1) — production-ready, matches SPSS |
| **Cohen's/Fleiss' kappa** | `statsmodels` or `sklearn.metrics` | For ordinal/categorical agreement |
| **Krippendorff's alpha** | `krippendorff` (PyPI) | Handles missing data, ordinal scales |
| **Item difficulty & discrimination** | `numpy`/`scipy` — manual formulas | p-values, point-biserial r, upper-lower discrimination index |
| **Distribution analysis** | `scipy.stats` | Shapiro-Wilk, skewness, kurtosis, ceiling/floor % |
| **Bland-Altman plots** | `matplotlib` — manual (simple) | Mean-difference plots for rater agreement |
| **Standard Error of Measurement** | Manual formula: `SEM = SD * sqrt(1 - reliability)` | |
| **Generalizability Theory** | Custom implementation using `statsmodels` mixed models or `numpy` variance decomposition | G-theory is a gap in Python — no mature package. But for our use case (single-facet or two-facet designs), we can implement the variance decomposition manually using ANOVA-based estimation. This is feasible and accurate for balanced designs. |
| **IRT (Rasch/2PL)** | `girth` or `py-irt` | Less mature than R's `mirt` or `ltm`, but adequate for our diagnostic use case. Not critical for the core goal. |
| **DIF analysis** | Manual (Mantel-Haenszel via `scipy.stats.chi2_contingency`) | R's `difR` package is superior, but MH-DIF is implementable in Python |

**Verdict: Python is sufficient for everything we need. SPSS/R would only be preferable for advanced IRT or complex G-theory designs, which are nice-to-have, not core.**

### 1.3 Functions to Implement

```python
# faculty_analysis.py

@dataclass
class InterRaterData:
    """Multiple raters scoring the same student cohort."""
    rater_labels: list[str]
    sections: list[str]
    scores: dict[int, dict[str, list[float | None]]]  # student_id → {section: [rater1_score, rater2_score, ...]}

@dataclass
class AnalysisReport:
    """Complete analysis output."""
    reliability: ReliabilityMetrics
    item_analysis: dict[str, ItemStats]
    distribution: dict[str, DistributionStats]
    faculty_comparison: FacultyComparisonStats
    warnings: list[str]
    recommendations: list[str]

# Core analysis functions:
def compute_icc(data: InterRaterData, section: str) -> ICCResult
def compute_kappa(data: InterRaterData, section: str) -> KappaResult
def compute_krippendorff_alpha(data: InterRaterData) -> float
def compute_item_difficulty(scores: list[float], max_score: float) -> float
def compute_item_discrimination(scores: list[float], total_scores: list[float]) -> float
def compute_distribution_stats(scores: list[float], max_score: float) -> DistributionStats
def compute_sem(sd: float, reliability: float) -> float
def compute_g_coefficient(data: InterRaterData, section: str) -> GTheoryResult
def run_full_analysis(sessions: list[SessionData], inter_rater: InterRaterData | None = None) -> AnalysisReport
```

### 1.4 Dependencies to Add

```
pingouin>=0.5.0
krippendorff>=0.6.0
scipy>=1.10.0
matplotlib>=3.7.0
```

(`numpy` and `pandas` are already available via other deps.)

---

## Phase 2: Empirical Score-Level Rubric Generator (`empirical_rubric.py`)

This is the high-value module that closes the gap.

### 2.1 Core Concept

Given:
- N students' written responses for a section (e.g., "HPI")
- Faculty-assigned scores for each response (e.g., scores of 1-5)
- The original rubric text

Produce:
- Empirically-derived score-level definitions: "A score of 5 typically includes X, Y, Z characteristics. A score of 3 typically includes..."
- These definitions are then injected into the grading prompt alongside the rubric

### 2.2 Algorithm

```
Step 1: COLLECT — Gather (response, faculty_score) pairs across all available sessions
Step 2: GROUP — Bucket responses by score level (all the 5s, all the 4s, etc.)
Step 3: FILTER — Require minimum N responses per level (≥3) for statistical validity
Step 4: SAMPLE — If too many responses at one level, sample representatively
Step 5: SYNTHESIZE — For each score level, send representative responses to LLM with prompt:
        "These N responses all received a score of [X] from faculty.
         The rubric criteria for this section is: [criteria].
         Identify the common characteristics that distinguish this score level.
         What do [X]-level responses typically include/exclude?"
Step 6: COMPARE — Send all level summaries to LLM for cross-level synthesis:
        "Here are the characteristics of each score level.
         Produce a single, coherent scoring guide that distinguishes levels."
Step 7: VALIDATE — Compare AI grades using new rubric vs. faculty grades (concordance)
Step 8: STORE — Save the empirical rubric as a reusable artifact
```

### 2.3 Data Structures

```python
@dataclass
class ScoreLevelExemplar:
    """A student response that received a known faculty score."""
    student_id: str
    response_text: str
    faculty_score: float
    session_label: str

@dataclass
class EmpiricalScoreLevel:
    """Empirically-derived definition for one score level."""
    score: float
    characteristics: list[str]       # What responses at this level typically include
    distinguishing_features: str     # What separates this level from adjacent levels
    exemplar_count: int              # How many exemplars informed this definition
    confidence: str                  # "high" (≥10 exemplars), "moderate" (5-9), "low" (3-4)

@dataclass
class EmpiricalRubric:
    """Complete empirical rubric for one section."""
    section: str
    assessment_type_id: str
    original_criteria: str           # The parsed rubric criteria
    score_levels: dict[float, EmpiricalScoreLevel]
    synthesis: str                   # LLM-generated holistic scoring guide
    validation_concordance: float | None  # Agreement rate with faculty scores
    created_from_sessions: list[str]
    total_exemplars: int
```

### 2.4 Key Functions

```python
def collect_exemplars(
    sessions: list[SessionData],
    grading_results: list[pd.DataFrame],  # Past grading output files with student responses
    section: str,
) -> list[ScoreLevelExemplar]

def generate_level_definitions(
    exemplars: list[ScoreLevelExemplar],
    section: str,
    original_criteria: str,
    caller: LLMCaller,
) -> dict[float, EmpiricalScoreLevel]

def synthesize_scoring_guide(
    levels: dict[float, EmpiricalScoreLevel],
    section: str,
    original_criteria: str,
    caller: LLMCaller,
) -> str

def validate_empirical_rubric(
    rubric: EmpiricalRubric,
    test_exemplars: list[ScoreLevelExemplar],
    caller: LLMCaller,
    assessment_type: AssessmentType,
) -> float  # Returns concordance rate

def generate_empirical_rubric(
    sessions: list[SessionData],
    grading_results: list[pd.DataFrame],
    section: str,
    original_criteria: str,
    assessment_type_id: str,
    caller: LLMCaller,
) -> EmpiricalRubric
```

### 2.5 LLM Prompts

**Level Definition Prompt** (per score level):
```
You are a psychometrician analyzing faculty-graded student responses.
Below are {N} student responses for the "{section}" section of a medical OSCE,
all scored {score}/{max_score} by faculty using this rubric:

{original_criteria}

Responses:
{numbered_responses}

Identify the COMMON CHARACTERISTICS that led faculty to assign this score:
1. What clinical content is typically present at this level?
2. What clinical content is typically missing at this level?
3. What quality indicators (specificity, organization, terminology) are common?
4. What distinguishes this level from one score point higher or lower?

Respond with JSON: {"characteristics": [...], "distinguishing_features": "..."}
```

**Cross-Level Synthesis Prompt**:
```
You are creating an empirical scoring guide for "{section}" based on analysis
of {total} faculty-graded responses. Here are the empirically-observed
characteristics at each score level:

{level_summaries}

The original rubric criteria:
{original_criteria}

Create a single coherent scoring guide that a grader can use. For each score level,
provide a 2-3 sentence description of what responses at that level typically look like.
Focus on OBSERVABLE DIFFERENCES between adjacent levels. Be specific about clinical
content expectations, not just quality adjectives.

Format:
{score}: [description]
```

---

## Phase 3: Integration into the Grading Pipeline

### 3.1 Modify `AssessmentType.build_grading_prompt()` and `build_user_message()`

Add an optional `benchmark_data` parameter to `rubric_data` that includes the empirical scoring guide. The grading prompt is augmented:

```python
# In kpsom_osce.py, modified build_user_message():
def build_user_message(self, section, student_response, rubric_data):
    msg = f"Rubric criteria:\n{rubric_data['parsed_rubric'][section]['criteria']}\n\n"

    # NEW: Include empirical scoring guide if available
    empirical = rubric_data.get("empirical_rubric", {}).get(section)
    if empirical:
        msg += f"\n\nEMPIRICAL SCORING GUIDE (based on {empirical.total_exemplars} "
        msg += f"faculty-graded responses):\n{empirical.synthesis}\n\n"

    msg += f"Student response:\n{student_response}"
    return msg
```

### 3.2 Modify `process_assessment()` in `grader.py`

Add a new optional parameter `benchmark_id` (or `empirical_rubric_path`) that loads the empirical rubric artifact and merges it into `rubric_data`:

```python
# In grader.py, process_assessment():
empirical_path = file_paths.get("empirical_rubric")
if empirical_path:
    import json
    with open(empirical_path) as f:
        empirical_data = json.load(f)
    rubric_data["empirical_rubric"] = empirical_data
```

### 3.3 Modify `GoldStandardBenchmark` dataclass

Add the empirical rubric as part of the benchmark artifact:

```python
@dataclass
class GoldStandardBenchmark:
    # ... existing fields ...
    empirical_rubrics: dict[str, EmpiricalRubric] = field(default_factory=dict)
```

---

## Phase 4: Streamlit UI Integration

### 4.1 Tab 5 (Gold Standard) — Enhanced Analysis Panel

Add new expandable sections:
- **Psychometric Analysis**: ICC, kappa, SEM, item difficulty/discrimination
- **Distribution Analysis**: Histograms, ceiling/floor effects, normality tests
- **Empirical Rubric Generation**: Button to generate empirical score-level definitions
  - Shows progress as each section is analyzed
  - Displays the generated scoring guide for faculty review/editing
  - "Approve & Save" button to persist the empirical rubric

### 4.2 Tab 1 (Grading) — Benchmark Integration

Add optional file upload:
- "Gold Standard Benchmark (optional)": Upload a benchmark JSON file
- When present, the empirical scoring guide is included in grading prompts
- UI shows: "Grading with empirical rubric from {date} ({N} exemplars)"

### 4.3 New Tab 6 (or sub-tab): Validation Dashboard

- Run validation: Grade a held-out set of faculty-scored responses using the empirical rubric
- Show concordance matrix (AI score vs. faculty score)
- Show per-section agreement rates
- Bland-Altman plots
- "This rubric achieves X% exact agreement, Y% ±1 agreement with faculty"

---

## Phase 5: File Changes Summary

| File | Change Type | Description |
|------|------------|-------------|
| `scripts/faculty_analysis.py` | **NEW** | Psychometric analysis module |
| `scripts/empirical_rubric.py` | **NEW** | Empirical score-level rubric generator |
| `scripts/gold_standard.py` | MODIFY | Add `empirical_rubrics` to `GoldStandardBenchmark`; add export support |
| `scripts/grader.py` | MODIFY | Accept and pass empirical rubric data in `process_assessment()` |
| `scripts/assessment_types/base.py` | MODIFY | Add optional `benchmark_data` parameter to `build_user_message` signature |
| `scripts/assessment_types/kpsom_osce.py` | MODIFY | Inject empirical scoring guide into grading prompts |
| `scripts/assessment_types/kpsom_documentation.py` | MODIFY | Same prompt injection |
| `scripts/assessment_types/kpsom_ethics.py` | MODIFY | Same prompt injection |
| `app.py` | MODIFY | Add UI for analysis, empirical rubric generation, validation |
| `requirements.txt` | MODIFY | Add `pingouin`, `krippendorff`, `scipy`, `matplotlib` |

---

## Phase 6: Implementation Order

1. **`faculty_analysis.py`** — The statistical analysis module (standalone, testable)
2. **`empirical_rubric.py`** — The empirical rubric generator (depends on LLM caller)
3. **Integration into `grader.py` and assessment types** — Wire the empirical rubric into prompts
4. **`gold_standard.py` modifications** — Add empirical rubric to benchmark artifacts
5. **`app.py` UI** — Add analysis panels, rubric generation UI, validation dashboard
6. **Tests** — Unit tests for statistical functions, integration tests for rubric generation

---

## Tool Assessment: Python vs. SPSS vs. R

| Capability | Python Feasibility | SPSS/R Advantage |
|-----------|-------------------|-----------------|
| ICC, Kappa, Krippendorff's alpha | Full (`pingouin`, `krippendorff`) | None |
| Item analysis (difficulty, discrimination) | Full (manual, straightforward) | None |
| Distribution tests | Full (`scipy.stats`) | None |
| G-theory (basic 1-2 facet) | Implementable (ANOVA-based) | R's `gtheory` package is more complete for complex designs |
| IRT (Rasch, 2PL) | Adequate (`girth`) | R's `mirt` is clearly superior for advanced models |
| DIF (Mantel-Haenszel) | Implementable | R's `difR` is more complete |
| Visualization | Full (`matplotlib`, `plotly` for Streamlit) | None |
| LLM integration | Full (already built) | Not applicable |

**Recommendation: Stay in Python.** The two areas where R is clearly superior (advanced IRT, complex G-theory) are nice-to-have analytics, not core functionality. The core value — empirical rubric generation via LLM — is only possible in Python within this codebase. Adding SPSS/R would create a fragile pipeline with manual handoffs for marginal analytical gains.

If advanced IRT or G-theory becomes a priority later, consider `rpy2` (Python→R bridge) rather than separate environments.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Insufficient exemplars at some score levels | Require minimum 3; show confidence levels; allow faculty to supplement |
| LLM-generated scoring guide is too vague | Use structured prompts; require specific clinical content, not adjectives |
| Empirical rubric causes AI scores to diverge MORE from faculty | Validation step (Phase 3) catches this before deployment |
| Faculty data format variability | Current loaders already handle this; extend as needed |
| Context window limits with many exemplars | Sample 5-10 per level; use chunked synthesis if needed |
