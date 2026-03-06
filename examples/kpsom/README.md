# KPSOM OSCE Example Files

This directory documents the expected file formats for KPSOM OSCE assessment types.

## Assessment Types

### I-PASS Handoff (`kpsom_ipass`)
Checklist-based scoring for I-PASS handoff notes.

**Sections** (with max scores):
| Section | Max Score |
|---------|-----------|
| Illness Severity | 2 |
| Patient Summary | 14 |
| Action List | 5 |
| Situation Awareness | 3 |
| Organization | 3 |
| **Total** | **27** |

### Clinical Documentation (`kpsom_documentation`)
Milestone-based scoring (1-5 scale) for clinical documentation.

**Sections** (with max scores):
| Section | Max Score |
|---------|-----------|
| HPI | 5 |
| Social History | 5 |
| Summary Statement | 5 |
| Assessment | 5 |
| Plan | 5 |
| Written Communication | 4 |
| **Total** | **29** |

## File Formats

### Student Responses (.xlsx)

The student responses file has a special two-header-row structure:

- **Row 0 (Q-number row)**: Contains question numbers (e.g., `Q3`, `Q4`, `Q5`). This row is metadata and is stored but not used for grading. The Q-numbers may not start at Q1 — for example, P-OSCE 1 files use Q3–Q7.
- **Row 1 (Section headers)**: Contains section names (e.g., `Student`, `Illness Severity`, `Patient Summary`, etc.). This is the real header row used for column naming.
- **Rows 2+**: One row per student with free-text responses.

**Column name mapping**: The loader maps common header variations to canonical section keys:

| Header Text (case-insensitive) | Section Key |
|-------------------------------|-------------|
| Illness Severity | `illness_severity` |
| Patient Summary | `patient_summary` |
| Action List | `action_list` |
| Situation Awareness | `situation_awareness` |
| Situational Awareness | `situation_awareness` |
| Organization | `organization` |
| HPI / History of Present Illness | `hpi` |
| Social History / Social Hx | `social_history` |
| Summary Statement / Summary | `summary_statement` |
| Assessment | `assessment` |
| Plan | `plan` |
| Written Communication / Written Comm | `written_communication` |

### Faculty Scores (.xlsx, optional)

The faculty scores file has a three-header-row structure with merged cells:

- **Row 0**: Section group headers (merged, spanning multiple columns)
- **Row 1**: Sub-group headers
- **Row 2**: Item-level column headers — this is the real header row
- **Rows 3+**: Student score data

**Important notes**:
- Load with `data_only=True` to resolve Excel formula values
- Student IDs are integers (deidentified; real IDs would be NUID format like `a123456`)
- Filter out trailing rows where `Student ID` is `None` — these contain comment templates
- Empty score cells are treated as 0.0
- Key columns: `Student`, `Total`, `Milestone`, `Comments`

### Rubric (.docx)

The rubric is a Word document containing scoring criteria. It is parsed via an LLM call (not manually structured). The `convert_docx_to_text()` function extracts the raw text, which is then sent to the LLM with a parsing prompt specific to the task type (checklist vs. milestone).

## Known Data Quirks

1. **All-None columns**: Some rubric items (e.g., "Trend lactate") may have no awarded scores across all students. The loader warns about this but does not drop the column.
2. **SA section inversion**: The "other contingency plan" bonus column may be awarded more frequently than primary SA items. This is valid data.
3. **Student ID normalization**: IDs appear as strings like `'10'`, `'1'`. They are normalized to `int` for joining with the responses file.
4. **Q-number offset**: Row 0 may show Q3–Q7 instead of Q1–Q5. This is stored as metadata (`df.attrs['q_numbers']`) but does not affect grading.
