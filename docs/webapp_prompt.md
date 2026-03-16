# Claude Code Prompt — OSCE Grader Web Application

## Context

You are working in the `/Users/cuttriss/Documents/GitHub Repos/OSCE-Grader/` repository. This is an existing Python CLI tool that uses LLMs to grade medical student OSCE post-encounter notes. It currently works entirely via command-line scripts (`scripts/grader.py`, `scripts/evaluate.py`, `scripts/providers.py`, `scripts/config.py`).

The grader supports three LLM providers (OpenAI, Anthropic, Google) with these models:
- **OpenAI:** gpt-4o, gpt-4o-mini
- **Anthropic:** claude-sonnet-4-6, claude-haiku-4-5
- **Google:** gemini-2.5-flash, gemini-2.5-pro

The evaluation system compares AI scores against two human graders on a 1–4 rubric scale per section (e.g., HPI, Physical Exam, Summary, DDx, Supporting Evidence, Plan — though sections are configurable).

Read all files in `scripts/` and `docs/` to fully understand the existing codebase before starting.

---

## Task

Build a **web application** (using Streamlit or a lightweight Python web framework of your choice) that wraps the existing grading and evaluation functionality into an interactive UI. The app should reuse the existing Python modules (`grader.py`, `evaluate.py`, `providers.py`, `config.py`) as the backend — do NOT rewrite the grading logic.

---

## Requirements

### 1. File Upload & Configuration

- Allow users to upload three Excel files: **rubric**, **answer key**, and **student notes**
- Show a preview of each uploaded file (first few rows) so users can verify they uploaded the right files
- Auto-detect the section columns from the uploaded files

### 2. LLM Model Selection with Recommendations

- Present a model selector that shows all 6 supported models grouped by provider
- For each model, display a recommendation card showing:
  - **Accuracy** (within-1 agreement from our benchmarks): gpt-4o: 99%, claude-sonnet-4-6: 99%, gemini-2.5-flash: 98%, gemini-2.5-pro: 96%, claude-haiku-4-5: 95%, gpt-4o-mini: ~95%
  - **Cost per 1,000 students**: gemini-2.5-flash: $5.19, claude-haiku-4-5: $12.30, gemini-2.5-pro: $21.00, gpt-4o: $27.00, claude-sonnet-4-6: $36.90
  - **Bias tendency**: claude-haiku-4-5: -0.04 (best), gemini-2.5-pro: -0.06, gemini-2.5-flash: -0.13, claude-sonnet-4-6: -0.35, gpt-4o: -0.48 (harshest)
  - A **recommended badge** on gemini-2.5-flash (best cost-accuracy ratio) and a **most accurate badge** on gpt-4o / claude-sonnet-4-6
- Allow temperature adjustment (slider, default 0.3)
- Allow worker count adjustment (1–8, default 4)
- API key input fields (per provider, with option to use environment variables)

### 3. Grading Execution

- A "Run Grading" button that kicks off the grading process
- **Real-time progress** — show which student is being graded (e.g., "Grading student 5 of 42...")
- Allow the user to download the raw results Excel file when complete

### 4. Analysis Dashboard (Post-Grading)

After grading completes (or when a user uploads an existing results file), display an analysis dashboard with:

#### 4a. Overall Metrics Panel
- **Within-1 Agreement** (% of scores within 1 point of human average) — with pass/fail indicator (≥90% = pass)
- **MAE** (Mean Absolute Error vs human average)
- **Bias** (signed, with interpretation: negative = AI harsher, positive = AI more lenient)
- **Exact Agreement** (% of identical scores)
- **Cost estimate** for the run

#### 4b. Per-Section Analysis Table
For each section, show:
- Section name
- AI mean score, Human mean score
- Within-1 %, MAE, Bias, Exact agreement %
- A classification tag:
  - 🟢 **Easy** — Within-1 ≥ 98% AND MAE ≤ 0.30
  - 🟡 **Moderate** — Within-1 ≥ 90% AND MAE ≤ 0.60
  - 🔴 **Difficult** — Within-1 < 90% OR MAE > 0.60
  - ⚠️ **High-Variability** — Standard deviation of (AI score - human avg) across students is > 0.75, indicating inconsistent grading even if the mean looks OK

#### 4c. Section Difficulty Summary
- A visual summary (bar chart or similar) showing which sections are easy, moderate, difficult, or high-variability
- Brief interpretation text explaining which sections the AI handles well and which may need prompt tuning or manual review

### 5. Outlier & Manual Review Flagging

This is a critical feature. Implement a flagging system that identifies scores needing manual review:

#### Flag Criteria (flag if ANY of the following are true):
1. **Score gap ≥ 2** — The AI score differs from the human average by 2 or more points
2. **AI score = 1 when humans averaged ≥ 3** — AI gave the lowest score but humans thought the student did well (likely AI grading error)
3. **AI score = 4 when humans averaged ≤ 2** — AI gave the highest score but humans thought the student did poorly (likely AI grading error)
4. **High section variability** — The score is in a section classified as "High-Variability" or "Difficult" AND the individual score gap is ≥ 1
5. **Cross-grader disagreement** — Human grader 1 and grader 2 differ by ≥ 2 points on this section (the humans themselves disagreed, so the "ground truth" is uncertain)

#### Flagged Items View
- A filterable, sortable table of all flagged scores showing: student identifier, section, AI score, G1 score, G2 score, human average, gap, flag reason(s)
- Count of total flags and breakdown by flag type
- Ability to download the flagged items as a separate Excel/CSV for distribution to human reviewers
- A summary stat: "X of Y total scores (Z%) flagged for manual review"

### 6. Answer Key Variability Analysis

Add an analysis that examines the answer key itself for potential issues:
- For each section, calculate the **human inter-rater agreement** (how often G1 and G2 agree or are within 1 point)
- Flag sections where human graders frequently disagree (inter-rater within-1 < 85%) — these sections may have ambiguous rubric criteria
- Show this alongside the AI analysis so users can distinguish "the AI is bad at this section" from "even humans can't agree on this section"

---

## Technical Notes

- Reuse the existing `grader.py` functions (especially `process_excel_file_with_key`) and `evaluate.py` functions — import them, don't rewrite them
- Reuse the existing `providers.py` factory pattern for LLM creation
- The `config.py` module uses module-level globals that get mutated (e.g., `config.MODEL = "gpt-4o"`). You may need to handle this carefully in the webapp context to avoid shared state between runs
- The evaluate.py `evaluate()` function returns a dict with all the metrics — use that as the data source for the dashboard
- Student notes Excel files have columns like `hpi`, `pex`, `sum` for student text plus `hpi_grader_1`, `hpi_grader_2` for human scores and `hpi_gpt_score` for AI scores after grading
- All scoring is on a 1–4 integer scale
- Add the webapp to the existing project (e.g., `app.py` in the root or `scripts/app.py`) rather than creating a separate repository
- Update `requirements.txt` with any new dependencies
