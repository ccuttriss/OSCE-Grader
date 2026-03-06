"""OSCE Grader — Streamlit Web Application.

A comprehensive web UI for grading medical student OSCE post-encounter notes
using LLMs. Provides grading, analysis dashboards, outlier flagging, and
rubric conversion.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — make scripts/ importable
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import config
from grader import (
    _read_excel_safe,
    _save_results,
    call_llm,
    compute_summary_stats,
    grade_section_with_key,
    read_rubric_and_key,
    run_dry_run,
    validate_input_columns,
)
from providers import create_caller
from convert_rubric import convert_docx_to_text, convert_pdf_to_text
from assessment_types import REGISTRY, get_type
from grader import process_assessment
from gold_standard import (
    load_faculty_session,
    validate_sessions,
    compute_cross_session_stats,
    compute_consensus_analysis,
    generate_example_sessions,
    session_to_dataframe,
    session_to_excel,
    build_bias_prompt,
    parse_bias_response,
    generate_benchmark_excel,
    generate_benchmark_json,
    GoldStandardBenchmark,
    BiasAnalysisResult,
    ConsensusResult,
)
from synthetic_generator import (
    generate_synthetic_session,
    rubric_to_display_text,
    rubric_to_excel,
    answer_key_to_excel,
    student_notes_to_excel,
    faculty_scores_to_excel,
    session_to_zip,
    all_sessions_to_zip,
    _TYPE_META as SYNTH_TYPE_META,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("osce_grader.streamlit")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="OSCE Grader",
    page_icon="\U0001f3eb",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Model benchmark data
# ---------------------------------------------------------------------------
MODEL_INFO = {
    "gpt-4o": {
        "provider": "openai",
        "accuracy": 99,
        "cost_1k": 27.00,
        "bias": -0.48,
        "badges": ["Most Accurate"],
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "accuracy": 95,
        "cost_1k": 1.62,
        "bias": None,
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "accuracy": 99,
        "cost_1k": 36.90,
        "bias": -0.35,
        "badges": ["Most Accurate"],
    },
    "claude-haiku-4-5": {
        "provider": "anthropic",
        "accuracy": 95,
        "cost_1k": 12.30,
        "bias": -0.04,
        "badges": ["Least Biased"],
    },
    "gemini-2.5-flash": {
        "provider": "google",
        "accuracy": 98,
        "cost_1k": 5.19,
        "bias": -0.13,
        "badges": ["Recommended"],
    },
    "gemini-2.5-pro": {
        "provider": "google",
        "accuracy": 96,
        "cost_1k": 21.00,
        "bias": -0.06,
    },
}

PROVIDER_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
}

# Environment variable names per provider for API keys
API_KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}

# ---------------------------------------------------------------------------
# Persistent API key storage (.env file, already gitignored)
# ---------------------------------------------------------------------------
_ENV_FILE = os.path.join(REPO_ROOT, ".env")


def _load_env_file() -> None:
    """Load API keys from .env into os.environ (does not overwrite existing)."""
    if not os.path.isfile(_ENV_FILE):
        return
    with open(_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and value and not os.environ.get(key):
                os.environ[key] = value


def _save_api_key_to_env(env_var: str, value: str) -> None:
    """Persist an API key to the .env file."""
    lines: list[str] = []
    found = False
    if os.path.isfile(_ENV_FILE):
        with open(_ENV_FILE) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith(env_var + "="):
                    lines.append(f"{env_var}={value}\n")
                    found = True
                else:
                    lines.append(line if line.endswith("\n") else line + "\n")
    if not found:
        lines.append(f"{env_var}={value}\n")
    with open(_ENV_FILE, "w") as f:
        f.writelines(lines)
    os.environ[env_var] = value


# Load persisted keys on startup
_load_env_file()


# ---------------------------------------------------------------------------
# Helper: save uploaded file to a temp path
# ---------------------------------------------------------------------------
def _save_upload(uploaded_file) -> str:
    """Write an uploaded file to a temp directory and return the path."""
    tmp_dir = os.path.join(REPO_ROOT, "uploads_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


# ---------------------------------------------------------------------------
# Persistent example file storage per assessment type
# ---------------------------------------------------------------------------
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")

_EXAMPLE_SLOTS = ("rubric", "student_notes", "faculty_scores")

_EXAMPLE_EXTENSIONS = {
    "rubric": (".docx", ".xlsx"),
    "student_notes": (".xlsx",),
    "faculty_scores": (".xlsx",),
}


def _examples_type_dir(type_id: str) -> str:
    """Return the persistent examples directory for a given assessment type."""
    d = os.path.join(EXAMPLES_DIR, type_id)
    os.makedirs(d, exist_ok=True)
    return d


def _save_example(type_id: str, slot: str, uploaded_file) -> str:
    """Persist an uploaded example file to disk, replacing any previous file in that slot."""
    d = _examples_type_dir(type_id)
    # Remove any existing file for this slot
    _remove_example(type_id, slot)
    # Save with a prefixed name so we can identify the slot
    filename = f"{slot}__{uploaded_file.name}"
    path = os.path.join(d, filename)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def _get_example_path(type_id: str, slot: str) -> str | None:
    """Return the path of the currently saved example for a slot, or None."""
    d = _examples_type_dir(type_id)
    prefix = f"{slot}__"
    for fname in os.listdir(d):
        if fname.startswith(prefix):
            return os.path.join(d, fname)
    return None


def _get_example_display_name(type_id: str, slot: str) -> str | None:
    """Return the original filename (without slot prefix) of a saved example."""
    d = _examples_type_dir(type_id)
    prefix = f"{slot}__"
    for fname in os.listdir(d):
        if fname.startswith(prefix):
            return fname[len(prefix):]
    return None


def _remove_example(type_id: str, slot: str) -> None:
    """Delete the saved example file for a slot."""
    path = _get_example_path(type_id, slot)
    if path and os.path.exists(path):
        os.remove(path)


# ---------------------------------------------------------------------------
# Helper: detect section columns from a DataFrame
# ---------------------------------------------------------------------------
def _detect_sections(df: pd.DataFrame) -> list[str]:
    """Return section column names found in the DataFrame."""
    known = ["hpi", "pex", "sum", "ddx", "support", "plan", "org"]
    return [s for s in known if s in df.columns]


# =========================================================================
# TAB 1: GRADE NOTES
# =========================================================================
def tab_grade_notes():
    st.header("Grade Notes")
    st.markdown("Upload your rubric, answer key, and student notes to grade with AI.")

    # --- Assessment type selector ---
    type_options = {
        "uk_osce": "Standard OSCE (HPI / PEX / SUM / DDX / Plan)",
        "kpsom_ipass": "KPSOM \u2014 I-PASS Handoff",
        "kpsom_documentation": "KPSOM \u2014 Clinical Documentation",
        "kpsom_ethics": "KPSOM \u2014 Ethics Open-Ended Questions",
    }

    selected_type_id = st.selectbox(
        "Assessment Type",
        options=list(type_options.keys()),
        format_func=lambda k: type_options[k],
        key="assessment_type",
    )
    assessment_type = get_type(selected_type_id)
    is_uk = selected_type_id == "uk_osce"

    st.divider()

    # --- Dynamic file uploads ---
    required_files = assessment_type.get_required_files()
    uploaded_files = {}
    file_cols = st.columns(len(required_files))
    for i, file_spec in enumerate(required_files):
        with file_cols[i]:
            label = file_spec["label"]
            if not file_spec.get("required", True):
                label += " (optional)"
            uploaded_files[file_spec["key"]] = st.file_uploader(
                label,
                type=file_spec["types"],
                key=f"upload_{file_spec['key']}",
            )

    # --- Previews ---
    for file_spec in required_files:
        fkey = file_spec["key"]
        uf = uploaded_files.get(fkey)
        if uf and file_spec["types"] == ["xlsx"]:
            with st.expander(f"{file_spec['label']} preview", expanded=False):
                try:
                    df_preview = pd.read_excel(uf)
                    st.dataframe(df_preview.head(5), use_container_width=True)
                except Exception:
                    st.warning("Could not preview this file.")
                uf.seek(0)

    # UK-specific section detection
    detected_sections = []
    if is_uk:
        notes_uf = uploaded_files.get("responses")
        if notes_uf:
            try:
                df_n = pd.read_excel(notes_uf)
                detected_sections = _detect_sections(df_n)
                if detected_sections:
                    st.info(f"Detected sections: **{', '.join(s.upper() for s in detected_sections)}**")
                notes_uf.seek(0)
            except Exception:
                pass

    st.divider()

    # --- Model selection ---
    st.subheader("Model Selection")

    # Group models by provider
    providers_grouped: dict[str, list[str]] = {}
    for model_name, info in MODEL_INFO.items():
        prov = info["provider"]
        providers_grouped.setdefault(prov, []).append(model_name)

    # Radio selector for model
    model_options = list(MODEL_INFO.keys())
    # Default to gemini-2.5-flash (Recommended)
    default_idx = model_options.index("gemini-2.5-flash") if "gemini-2.5-flash" in model_options else 0

    selected_model = st.radio(
        "Choose a model",
        model_options,
        index=default_idx,
        format_func=lambda m: (
            f"{m} ({PROVIDER_LABELS[MODEL_INFO[m]['provider']]}) — "
            f"{MODEL_INFO[m]['accuracy']}% accuracy, ${MODEL_INFO[m]['cost_1k']:.2f}/1K students"
            + (f" — {', '.join(MODEL_INFO[m].get('badges', []))}" if MODEL_INFO[m].get("badges") else "")
        ),
        key="model_select",
    )

    info = MODEL_INFO[selected_model]
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Accuracy", f"{info['accuracy']}%")
    mcol2.metric("Cost / 1K students", f"${info['cost_1k']:.2f}")
    mcol3.metric("Bias", f"{info['bias']:+.2f}" if info['bias'] is not None else "N/A")
    if info.get("badges"):
        mcol4.metric("Badge", ", ".join(info["badges"]))

    st.divider()

    # --- Parameters ---
    st.subheader("Parameters")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        max_temp = 1.0 if info["provider"] == "anthropic" else 2.0
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=max_temp,
            value=min(0.3, max_temp),
            step=0.1,
            key="temperature",
        )
    with pcol2:
        workers = st.slider(
            "Parallel workers",
            min_value=1,
            max_value=8,
            value=4,
            key="workers",
        )

    # --- API key ---
    provider = info["provider"]
    env_var = API_KEY_ENV_VARS[provider]
    has_env_key = bool(os.environ.get(env_var, "").strip())

    if has_env_key:
        st.success(f"API key found in environment (`{env_var}`)")
        api_key_input = None
    else:
        api_key_input = st.text_input(
            f"{PROVIDER_LABELS[provider]} API Key",
            type="password",
            key=f"api_key_{provider}",
            help=f"Set {env_var} environment variable to avoid entering this each time.",
        )

    st.divider()

    # --- Buttons ---
    bcol1, bcol2 = st.columns(2)
    dry_run_clicked = bcol1.button("Dry Run", use_container_width=True, type="secondary")
    grade_clicked = bcol2.button("Grade", use_container_width=True, type="primary")

    # --- Validation ---
    def _validate():
        for file_spec in required_files:
            if file_spec.get("required", True) and not uploaded_files.get(file_spec["key"]):
                st.error(f"Please upload: {file_spec['label']}")
                return False
        if not has_env_key and not api_key_input:
            st.error(f"Please provide a {PROVIDER_LABELS[provider]} API key.")
            return False
        return True

    # --- Set API key if provided ---
    def _set_api_key():
        if api_key_input and not has_env_key:
            _save_api_key_to_env(env_var, api_key_input)

    # --- Set config for the selected model ---
    def _set_config():
        config.MODEL = selected_model
        config.PROVIDER = provider
        config.TEMPERATURE = temperature
        if detected_sections:
            config.SECTIONS = detected_sections

    # --- Dry Run ---
    if dry_run_clicked:
        if not _validate():
            return

        _set_api_key()
        _set_config()

        if is_uk:
            rubric_path = _save_upload(uploaded_files["rubric"])
            key_path = _save_upload(uploaded_files["answer_key"])
            notes_path = _save_upload(uploaded_files["responses"])

            try:
                result = run_dry_run(notes_path, rubric_path, key_path)
                st.success("Dry run complete!")
                dcol1, dcol2, dcol3, dcol4 = st.columns(4)
                dcol1.metric("Students", result["total_rows"])
                dcol2.metric("API Calls", result["api_calls"])
                dcol3.metric("Est. Cost", result["est_cost"])
                dcol4.metric("Model", result["model"])
                st.info(f"Sections: {', '.join(s.upper() for s in result['sections'])}")
            except (SystemExit, ValueError) as e:
                st.error(f"Dry run failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
        else:
            # KPSOM dry run: count students and estimate API calls
            try:
                saved_paths = {}
                for file_spec in required_files:
                    uf = uploaded_files.get(file_spec["key"])
                    if uf:
                        saved_paths[file_spec["key"]] = _save_upload(uf)
                df_dry, _ = assessment_type.load_inputs(**saved_paths)
                sections = assessment_type.get_sections()
                n_students = len(df_dry)
                api_calls = n_students * len(sections) + 1  # +1 for rubric parsing
                cost_per_call = config.MODEL_COSTS.get(config.MODEL, (0.15, 0.60))
                est_cost = api_calls * (cost_per_call[0] * 1.0 / 1000 + cost_per_call[1] * 0.5 / 1000)
                st.success("Dry run complete!")
                dcol1, dcol2, dcol3, dcol4 = st.columns(4)
                dcol1.metric("Students", n_students)
                dcol2.metric("API Calls", api_calls)
                dcol3.metric("Est. Cost", f"${est_cost:.2f}")
                dcol4.metric("Model", config.MODEL)
                st.info(f"Sections: {', '.join(s.upper().replace('_', ' ') for s in sections)}")
            except (SystemExit, ValueError) as e:
                st.error(f"Dry run failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    # --- Grade ---
    if grade_clicked:
        if not _validate():
            return

        _set_api_key()
        _set_config()

        # Output paths
        output_dir = os.path.join(REPO_ROOT, "uploads_tmp")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "results_streamlit.xlsx")
        log_file = os.path.join(output_dir, "results_streamlit.log")

        try:
            caller = create_caller(provider)
        except SystemExit:
            st.error(f"Failed to create {PROVIDER_LABELS[provider]} caller. Check your API key.")
            return

        if is_uk:
            # ---- UK grading path (unchanged) ----
            rubric_path = _save_upload(uploaded_files["rubric"])
            key_path = _save_upload(uploaded_files["answer_key"])
            notes_path = _save_upload(uploaded_files["responses"])

            try:
                rubric_content, answer_key_content = read_rubric_and_key(rubric_path, key_path)
            except SystemExit:
                st.error("Failed to read rubric or answer key files. Check format.")
                return

            try:
                df = pd.read_excel(notes_path)
            except Exception as e:
                st.error(f"Failed to read student notes: {e}")
                return

            sections = config.SECTIONS
            missing = validate_input_columns(df, sections)
            if missing:
                st.error(f"Missing columns in student notes: {', '.join(missing)}")
                return

            total_rows = len(df)
            if total_rows == 0:
                st.warning("Student notes file has no data rows.")
                return

            progress_bar = st.progress(0, text="Starting grading...")
            status = st.status("Grading in progress...", expanded=True)
            effective_workers = min(workers, len(sections))

            graded = 0
            for index, row in df.iterrows():
                pct = graded / total_rows
                progress_bar.progress(pct, text=f"Student {graded + 1} of {total_rows}...")
                status.update(label=f"Grading student {graded + 1} of {total_rows}...")
                status.write(f"Processing student {graded + 1}...")

                gradable = []
                for section in sections:
                    section_content = row[section]
                    if pd.notna(section_content):
                        gradable.append((section, str(section_content)))

                if effective_workers <= 1 or len(gradable) <= 1:
                    for section, content in gradable:
                        try:
                            explanation, numeric_score = grade_section_with_key(
                                caller, rubric_content, answer_key_content,
                                content, section, log_file, temperature, config.TOP_P,
                            )
                            df.at[index, f"{section}_gpt_explanation"] = explanation
                            df.at[index, f"{section}_gpt_score"] = numeric_score
                        except SystemExit:
                            st.error(f"API error grading section '{section}' for student {graded + 1}")
                            return
                        except Exception as e:
                            st.error(f"Error grading {section} for student {graded + 1}: {e}")
                            return
                else:
                    futures = {}
                    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                        for section, content in gradable:
                            future = pool.submit(
                                grade_section_with_key,
                                caller, rubric_content, answer_key_content,
                                content, section, log_file, temperature, config.TOP_P,
                            )
                            futures[section] = future

                    for section, future in futures.items():
                        try:
                            explanation, numeric_score = future.result()
                            df.at[index, f"{section}_gpt_explanation"] = explanation
                            df.at[index, f"{section}_gpt_score"] = numeric_score
                        except SystemExit:
                            st.error(f"API error grading section '{section}' for student {graded + 1}")
                            return
                        except Exception as e:
                            st.error(f"Error grading {section} for student {graded + 1}: {e}")
                            return

                graded += 1
                _save_results(df, output_file)

            progress_bar.progress(1.0, text="Grading complete!")
            status.update(label="Grading complete!", state="complete")

            st.session_state["results_df"] = df
            st.session_state["results_sections"] = sections

            stats = compute_summary_stats(df, sections)
            if stats:
                st.subheader("Summary Statistics")
                st.dataframe(pd.DataFrame(stats), use_container_width=True)

            dcol1, dcol2 = st.columns(2)
            with dcol1:
                with open(output_file, "rb") as f:
                    st.download_button(
                        "Download Results (.xlsx)",
                        data=f,
                        file_name="osce_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            with dcol2:
                if os.path.isfile(log_file):
                    with open(log_file, "rb") as f:
                        st.download_button(
                            "Download Log (.log)",
                            data=f,
                            file_name="osce_grading.log",
                            mime="text/plain",
                        )

        else:
            # ---- KPSOM grading path ----
            saved_paths = {}
            for file_spec in required_files:
                uf = uploaded_files.get(file_spec["key"])
                if uf:
                    saved_paths[file_spec["key"]] = _save_upload(uf)

            sections = assessment_type.get_sections()

            progress_bar = st.progress(0, text="Starting grading...")
            status = st.status("Grading in progress...", expanded=True)

            def _kpsom_progress(current, total):
                if total > 0:
                    pct = current / total
                    progress_bar.progress(pct, text=f"Student {current + 1} of {total}...")
                    status.update(label=f"Grading student {current + 1} of {total}...")
                    status.write(f"Processing student {current + 1}...")

            try:
                result_df = process_assessment(
                    assessment_type,
                    caller,
                    saved_paths,
                    output_file,
                    temperature,
                    config.TOP_P,
                    max_workers=workers,
                    progress_callback=_kpsom_progress,
                )
            except (SystemExit, ValueError) as e:
                st.error(f"Grading failed: {e}")
                return
            except Exception as e:
                st.error(f"Unexpected error during grading: {e}")
                return

            progress_bar.progress(1.0, text="Grading complete!")
            status.update(label="Grading complete!", state="complete")

            st.session_state["results_df"] = result_df
            st.session_state["results_sections"] = sections

            # --- KPSOM Results Display ---
            if not result_df.empty:
                # Score summary table
                st.subheader("Score Summary")
                summary_rows = []
                for sec in sections:
                    ai_col = f"{sec}_ai_score"
                    fac_col = f"{sec}_faculty_score"
                    row_data = {"Section": sec.upper().replace("_", " ")}
                    if ai_col in result_df.columns:
                        ai_scores = pd.to_numeric(result_df[ai_col], errors="coerce").dropna()
                        row_data["AI Mean"] = round(ai_scores.mean(), 2) if len(ai_scores) > 0 else None
                    if fac_col in result_df.columns:
                        fac_scores = pd.to_numeric(result_df[fac_col], errors="coerce").dropna()
                        row_data["Faculty Mean"] = round(fac_scores.mean(), 2) if len(fac_scores) > 0 else None
                    delta_col = f"{sec}_delta"
                    if delta_col in result_df.columns:
                        deltas = pd.to_numeric(result_df[delta_col], errors="coerce").dropna()
                        row_data["Mean Delta"] = round(deltas.mean(), 2) if len(deltas) > 0 else None
                    summary_rows.append(row_data)
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

                # Milestone distribution
                if "ai_milestone" in result_df.columns:
                    st.subheader("Milestone Distribution")
                    milestone_counts = result_df["ai_milestone"].value_counts()
                    st.dataframe(milestone_counts.reset_index().rename(
                        columns={"index": "Milestone", "ai_milestone": "Milestone", "count": "Count"}
                    ), use_container_width=True, hide_index=True)

                # Faculty comparison
                if "faculty_total" in result_df.columns and "ai_total" in result_df.columns:
                    with st.expander("Faculty vs AI Comparison (delta > 2 points)"):
                        comparison = result_df[["student_id", "ai_total", "faculty_total", "total_delta"]].copy()
                        comparison = comparison.dropna(subset=["total_delta"])
                        outliers = comparison[comparison["total_delta"].abs() > 2]
                        if len(outliers) > 0:
                            st.dataframe(outliers, use_container_width=True, hide_index=True)
                        else:
                            st.success("No students differ by more than 2 points.")

            # Download buttons
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                with open(output_file, "rb") as f:
                    st.download_button(
                        "Download Results (.xlsx)",
                        data=f,
                        file_name="osce_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            with dcol2:
                if os.path.isfile(log_file):
                    with open(log_file, "rb") as f:
                        st.download_button(
                            "Download Log (.log)",
                            data=f,
                            file_name="osce_grading.log",
                            mime="text/plain",
                        )


# =========================================================================
# TAB 2: ANALYSIS DASHBOARD
# =========================================================================
def tab_analysis():
    st.header("Analysis Dashboard")
    st.markdown("Evaluate AI grading accuracy against human graders.")

    # Option to use just-graded results or upload
    source = st.radio(
        "Results source",
        ["Upload results file", "Use results from grading tab"],
        index=0 if "results_df" not in st.session_state else 1,
        horizontal=True,
        key="analysis_source",
    )

    df = None
    if source == "Use results from grading tab":
        if "results_df" in st.session_state:
            df = st.session_state["results_df"]
            st.success("Using results from the grading tab.")
        else:
            st.warning("No results from the grading tab. Please grade notes first or upload a file.")
            return
    else:
        uploaded = st.file_uploader("Upload graded results (.xlsx)", type=["xlsx"], key="analysis_upload")
        if uploaded:
            df = pd.read_excel(uploaded)
        else:
            st.info("Upload a graded results Excel file to see the analysis dashboard.")
            return

    if df is None:
        return

    # Auto-detect sections with both GPT and human grader columns
    sections = []
    for col in df.columns:
        if col.endswith("_gpt_score"):
            sec = col.replace("_gpt_score", "")
            if f"{sec}_grader_1" in df.columns and f"{sec}_grader_2" in df.columns:
                sections.append(sec)

    if not sections:
        st.warning(
            "No sections found with both AI scores (`_gpt_score`) and human grader "
            "scores (`_grader_1`, `_grader_2`). Upload a file with human grader columns."
        )
        return

    # Run evaluate() and suppress stdout
    tmp_path = os.path.join(REPO_ROOT, "uploads_tmp", "_analysis_temp.xlsx")
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    df.to_excel(tmp_path, index=False)

    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture):
        try:
            from scripts.evaluate import evaluate
            results = evaluate(tmp_path, sections=sections, model=config.MODEL)
        except SystemExit:
            st.error("Evaluation failed. Check that the file has the required columns.")
            return

    overall = results.get("_overall", {})

    # --- Overall metrics ---
    st.subheader("Overall Metrics")
    ocol1, ocol2, ocol3, ocol4, ocol5 = st.columns(5)
    ocol1.metric("Within-1 %", f"{overall.get('within1_pct', 0):.0f}%")
    ocol2.metric("MAE", f"{overall.get('mae', 0):.2f}")
    ocol3.metric("Bias", f"{overall.get('bias', 0):+.2f}")
    ocol4.metric("Total Comparisons", overall.get("total", 0))
    if "est_cost" in overall:
        ocol5.metric("Est. Cost", f"${overall['est_cost']:.4f}")

    # Pass/fail verdict
    w1 = overall.get("within1_pct", 0)
    if w1 >= 90:
        st.success(f"TARGET MET: {w1:.0f}% of scores within 1 point of human average")
    elif w1 >= 80:
        st.warning(f"CLOSE: {w1:.0f}% within 1 point (target: 90%)")
    else:
        st.error(f"NOT MET: {w1:.0f}% within 1 point (target: 90%)")

    st.divider()

    # --- Per-section table ---
    st.subheader("Per-Section Breakdown")
    section_rows = []
    for sec in sections:
        if sec not in results or sec == "_overall":
            continue
        r = results[sec]

        # Difficulty classification
        w1_pct = min(r.get("within1_g1", 0), r.get("within1_g2", 0))
        mae = r.get("mae_vs_avg", 0)
        if w1_pct >= 95 and mae <= 0.3:
            difficulty = "Easy"
            diff_color = "\U0001f7e2"
        elif w1_pct >= 85 and mae <= 0.6:
            difficulty = "Moderate"
            diff_color = "\U0001f7e1"
        elif w1_pct < 80 or mae > 1.0:
            difficulty = "Difficult"
            diff_color = "\U0001f534"
        else:
            difficulty = "Moderate"
            diff_color = "\U0001f7e1"

        # Check for high variability (human graders disagree a lot)
        human_range = abs(r.get("g1_mean", 0) - r.get("g2_mean", 0))
        if human_range > 0.5:
            difficulty = "High-Variability"
            diff_color = "\u26a0\ufe0f"

        section_rows.append({
            "Section": sec.upper(),
            "AI Mean": round(r.get("gpt_mean", 0), 2),
            "Human Mean": round(r.get("human_avg_mean", 0), 2),
            "Within-1 %": f"{w1_pct:.0f}%",
            "MAE": round(mae, 2),
            "Bias": f"{r.get('bias', 0):+.2f}",
            "Exact %": f"{max(r.get('exact_agree_g1', 0), r.get('exact_agree_g2', 0)):.0f}%",
            "Difficulty": f"{diff_color} {difficulty}",
        })

    if section_rows:
        st.dataframe(pd.DataFrame(section_rows), use_container_width=True, hide_index=True)

    st.divider()

    # --- Section difficulty chart ---
    st.subheader("Section Difficulty Chart")
    chart_data = []
    for sec in sections:
        if sec not in results or sec == "_overall":
            continue
        r = results[sec]
        w1_avg = (r.get("within1_g1", 0) + r.get("within1_g2", 0)) / 2
        chart_data.append({"Section": sec.upper(), "Within-1 %": w1_avg})

    if chart_data:
        chart_df = pd.DataFrame(chart_data)
        st.bar_chart(chart_df, x="Section", y="Within-1 %", horizontal=False)

    # Store for flagged items tab
    st.session_state["analysis_df"] = df
    st.session_state["analysis_sections"] = sections
    st.session_state["analysis_results"] = results


# =========================================================================
# TAB 3: FLAGGED ITEMS
# =========================================================================
def tab_flagged():
    st.header("Flagged Items")
    st.markdown("Identify scores that may need human review based on 5 criteria.")

    # Data source
    source = st.radio(
        "Data source",
        ["Use analysis data", "Upload results file"],
        index=0 if "analysis_df" in st.session_state else 1,
        horizontal=True,
        key="flagged_source",
    )

    df = None
    sections = None
    if source == "Use analysis data":
        if "analysis_df" in st.session_state:
            df = st.session_state["analysis_df"]
            sections = st.session_state.get("analysis_sections", [])
        else:
            st.warning("No analysis data available. Run the Analysis Dashboard first or upload a file.")
            return
    else:
        uploaded = st.file_uploader("Upload graded results (.xlsx)", type=["xlsx"], key="flagged_upload")
        if uploaded:
            df = pd.read_excel(uploaded)
            sections = []
            for col in df.columns:
                if col.endswith("_gpt_score"):
                    sec = col.replace("_gpt_score", "")
                    if f"{sec}_grader_1" in df.columns and f"{sec}_grader_2" in df.columns:
                        sections.append(sec)
        else:
            st.info("Upload a graded results file or run the Analysis Dashboard first.")
            return

    if df is None or not sections:
        st.warning("No sections with both AI and human grader scores found.")
        return

    # Determine difficulty classification per section (for criterion 4)
    analysis_results = st.session_state.get("analysis_results", {})
    difficult_sections = set()
    for sec in sections:
        if sec in analysis_results:
            r = analysis_results[sec]
            w1_pct = min(r.get("within1_g1", 0), r.get("within1_g2", 0))
            mae = r.get("mae_vs_avg", 0)
            human_range = abs(r.get("g1_mean", 0) - r.get("g2_mean", 0))
            if w1_pct < 80 or mae > 1.0 or human_range > 0.5:
                difficult_sections.add(sec)

    # Compute flags
    flagged_rows = []
    flag_counts = {
        "Gap >= 2 from human avg": 0,
        "AI=1, humans >= 3": 0,
        "AI=4, humans <= 2": 0,
        "Difficult section, gap >= 1": 0,
        "Human graders disagree by >= 2": 0,
    }

    total_scores = 0

    for idx, row in df.iterrows():
        for sec in sections:
            gpt_col = f"{sec}_gpt_score"
            g1_col = f"{sec}_grader_1"
            g2_col = f"{sec}_grader_2"

            gpt = row.get(gpt_col)
            g1 = row.get(g1_col)
            g2 = row.get(g2_col)

            if pd.isna(gpt) or pd.isna(g1) or pd.isna(g2):
                continue

            gpt = float(gpt)
            g1 = float(g1)
            g2 = float(g2)
            human_avg = (g1 + g2) / 2
            gap = abs(gpt - human_avg)

            total_scores += 1
            reasons = []

            # Criterion 1: Score gap >= 2 from human avg
            if gap >= 2:
                reasons.append("Gap >= 2 from human avg")
                flag_counts["Gap >= 2 from human avg"] += 1

            # Criterion 2: AI=1 when humans >= 3
            if gpt == 1 and g1 >= 3 and g2 >= 3:
                reasons.append("AI=1, humans >= 3")
                flag_counts["AI=1, humans >= 3"] += 1

            # Criterion 3: AI=4 when humans <= 2
            if gpt == 4 and g1 <= 2 and g2 <= 2:
                reasons.append("AI=4, humans <= 2")
                flag_counts["AI=4, humans <= 2"] += 1

            # Criterion 4: Difficult/high-var section AND gap >= 1
            if sec in difficult_sections and gap >= 1:
                reasons.append("Difficult section, gap >= 1")
                flag_counts["Difficult section, gap >= 1"] += 1

            # Criterion 5: Human graders disagree by >= 2
            if abs(g1 - g2) >= 2:
                reasons.append("Human graders disagree by >= 2")
                flag_counts["Human graders disagree by >= 2"] += 1

            if reasons:
                # Try to get a student identifier
                student_id = ""
                for id_col in ["student_id", "student", "name", "id"]:
                    if id_col in df.columns:
                        student_id = str(row[id_col])
                        break
                if not student_id:
                    student_id = f"Row {idx + 1}"

                flagged_rows.append({
                    "Student": student_id,
                    "Section": sec.upper(),
                    "AI Score": int(gpt),
                    "Grader 1": int(g1),
                    "Grader 2": int(g2),
                    "Human Avg": round(human_avg, 1),
                    "Gap": round(gap, 1),
                    "Flags": "; ".join(reasons),
                })

    # Summary
    n_flagged = len(flagged_rows)
    if total_scores > 0:
        pct = n_flagged / total_scores * 100
        st.metric("Flagged Scores", f"{n_flagged} of {total_scores} ({pct:.0f}%)")
    else:
        st.warning("No score data found to analyze.")
        return

    if n_flagged == 0:
        st.success("No scores were flagged for review.")
        return

    # Breakdown by flag type
    st.subheader("Breakdown by Flag Type")
    for flag_name, count in flag_counts.items():
        if count > 0:
            st.write(f"- **{flag_name}**: {count}")

    st.divider()

    # Flagged items table
    st.subheader("Flagged Items")
    flagged_df = pd.DataFrame(flagged_rows)
    st.dataframe(flagged_df, use_container_width=True, hide_index=True)

    # Download
    csv_data = flagged_df.to_csv(index=False)
    st.download_button(
        "Download Flagged Items (.csv)",
        data=csv_data,
        file_name="flagged_items.csv",
        mime="text/csv",
    )


# =========================================================================
# TAB 4: CONVERT RUBRIC
# =========================================================================
def tab_convert():
    st.header("Convert Rubric")
    st.markdown(
        "Upload a PDF or DOCX rubric and the AI will parse it into section columns "
        "for use with the grader."
    )

    uploaded = st.file_uploader(
        "Upload rubric file",
        type=["pdf", "docx"],
        key="convert_upload",
    )

    if not uploaded:
        st.info("Upload a PDF or DOCX rubric file to convert it.")
        return

    # Model selection for conversion
    convert_model = st.selectbox(
        "Model for conversion",
        list(MODEL_INFO.keys()),
        index=list(MODEL_INFO.keys()).index("gpt-4o"),
        key="convert_model",
    )
    convert_provider = MODEL_INFO[convert_model]["provider"]
    convert_env_var = API_KEY_ENV_VARS[convert_provider]
    has_convert_key = bool(os.environ.get(convert_env_var, "").strip())

    if not has_convert_key:
        convert_key = st.text_input(
            f"{PROVIDER_LABELS[convert_provider]} API Key (required for conversion)",
            type="password",
            key="convert_api_key",
        )
        if not convert_key:
            st.warning(f"A {PROVIDER_LABELS[convert_provider]} API key is required for rubric conversion.")
            return
        _save_api_key_to_env(convert_env_var, convert_key)

    if st.button("Convert Rubric", type="primary"):
        # Save uploaded file
        path = _save_upload(uploaded)
        ext = os.path.splitext(uploaded.name)[-1].lower()

        with st.spinner("Extracting text from document..."):
            try:
                if ext == ".pdf":
                    raw_text = convert_pdf_to_text(path)
                elif ext == ".docx":
                    raw_text = convert_docx_to_text(path)
                else:
                    st.error("Unsupported file type. Please upload a .pdf or .docx file.")
                    return
            except SystemExit:
                st.error("Failed to extract text from the uploaded document.")
                return

        if not raw_text.strip():
            st.error("No text could be extracted from the document.")
            return

        with st.spinner("Parsing rubric with AI..."):
            try:
                old_model = config.MODEL
                old_provider = config.PROVIDER
                config.MODEL = convert_model
                config.PROVIDER = convert_provider

                caller = create_caller(convert_provider)
                messages = [
                    {"role": "system", "content": config.CONVERT_PROMPT},
                    {"role": "user", "content": raw_text},
                ]
                response = call_llm(caller, messages, temperature=0.0, top_p=1.0)

                # Restore config
                config.MODEL = old_model
                config.PROVIDER = old_provider

                # Parse JSON response
                import json
                cleaned = response.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[-1]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                    cleaned = cleaned.strip()

                parsed = json.loads(cleaned)
                for key in config.SECTIONS:
                    parsed.setdefault(key, "")

            except SystemExit:
                st.error("API error during rubric conversion. Check your API key.")
                return
            except json.JSONDecodeError:
                st.error("The AI response could not be parsed as JSON. Try again.")
                return
            except Exception as e:
                st.error(f"Rubric conversion failed: {e}")
                return

        # Show results
        st.success("Rubric converted successfully!")

        found = [k for k, v in parsed.items() if v.strip()]
        empty = [k for k, v in parsed.items() if not v.strip()]
        st.info(f"Sections found: **{', '.join(s.upper() for s in found)}**")
        if empty:
            st.warning(f"Empty sections: {', '.join(s.upper() for s in empty)}")

        # Preview
        with st.expander("Section content preview", expanded=True):
            for sec, text in parsed.items():
                if text.strip():
                    st.markdown(f"**{sec.upper()}**")
                    st.text(text[:500] + ("..." if len(text) > 500 else ""))

        # Build Excel for download
        rubric_df = pd.DataFrame([parsed])
        output_buf = io.BytesIO()
        rubric_df.to_excel(output_buf, index=False)
        output_buf.seek(0)

        st.download_button(
            "Download Converted Rubric (.xlsx)",
            data=output_buf,
            file_name="converted_rubric.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# =========================================================================
# TAB 5: GOLD STANDARD ANALYSIS
# =========================================================================
def tab_gold_standard():
    st.header("Gold Standard Analysis")
    st.markdown(
        "Upload 2\u201310 faculty score files from different administrations of the same "
        "exam to analyze scoring patterns, detect bias, and establish gold standard benchmarks."
    )

    # --- Assessment type selector ---
    gs_type_options = {
        "kpsom_ipass": "KPSOM \u2014 I-PASS Handoff",
        "kpsom_documentation": "KPSOM \u2014 Clinical Documentation",
        "kpsom_ethics": "KPSOM \u2014 Ethics Open-Ended Questions",
        "uk_osce": "Standard OSCE (HPI / PEX / SUM / DDX / Plan)",
    }
    gs_type_id = st.selectbox(
        "Assessment type",
        list(gs_type_options.keys()),
        format_func=lambda k: gs_type_options[k],
        key="gs_type_select",
    )

    # --- Data source selector ---
    data_source = st.radio(
        "Data source",
        ["Upload files", "Use built-in examples"],
        horizontal=True,
        key="gs_data_source",
        help="Use built-in examples to explore the tool before uploading your own data.",
    )

    if data_source == "Use built-in examples":
        ecol1, ecol2, ecol3 = st.columns(3)
        with ecol1:
            n_sessions = st.slider(
                "Number of sessions",
                min_value=2,
                max_value=6,
                value=3,
                key="gs_example_n",
            )
        with ecol2:
            students_per = st.slider(
                "Students per session",
                min_value=4,
                max_value=20,
                value=8,
                key="gs_example_students",
            )
        with ecol3:
            variability = st.slider(
                "Variability",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="gs_variability",
                help=(
                    "Controls how much variation exists in the generated scores. "
                    "**Low (0.0)**: tight, homogeneous scores with minimal session drift — "
                    "simulates highly calibrated faculty. "
                    "**Medium (0.5)**: realistic variation you'd expect across semesters. "
                    "**High (1.0)**: wide score spread with large session-to-session bias — "
                    "simulates poorly calibrated or inconsistent raters."
                ),
            )

        if st.button("Generate Examples", type="primary", key="gs_load_examples"):
            sessions = generate_example_sessions(
                gs_type_id,
                n_sessions=n_sessions,
                students_per_session=students_per,
                variability=variability,
            )
            st.session_state["gs_sessions"] = sessions
            st.session_state["gs_type_id"] = gs_type_id
            st.session_state["gs_example_generated"] = True

        # --- Preview & download generated examples ---
        if st.session_state.get("gs_example_generated") and "gs_sessions" in st.session_state:
            example_sessions = st.session_state["gs_sessions"]
            st.divider()
            st.subheader("Generated Example Sessions")
            st.caption(
                "Preview the generated faculty score data below. Each tab shows "
                "one session — the same format you would upload as a real file."
            )
            session_tabs = st.tabs(
                [f"{s.label}" for s in example_sessions]
            )
            for tab, session in zip(session_tabs, example_sessions):
                with tab:
                    preview_df = session_to_dataframe(session)
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)

                    # Per-session download
                    xlsx_bytes = session_to_excel(session)
                    safe_label = session.label.replace(" ", "_")
                    st.download_button(
                        f"Download {session.label} (.xlsx)",
                        data=xlsx_bytes,
                        file_name=f"example_{gs_type_id}_{safe_label}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"gs_dl_{safe_label}",
                    )

            # Download all sessions as a zip
            import zipfile
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for session in example_sessions:
                    safe_label = session.label.replace(" ", "_")
                    fname = f"example_{gs_type_id}_{safe_label}.xlsx"
                    zf.writestr(fname, session_to_excel(session))
            st.download_button(
                "Download All Sessions (.zip)",
                data=zip_buf.getvalue(),
                file_name=f"example_{gs_type_id}_all_sessions.zip",
                mime="application/zip",
                key="gs_dl_all_zip",
            )

    else:
        # --- File upload ---
        uploaded = st.file_uploader(
            "Upload faculty score files (2\u201310 Excel files)",
            type=["xlsx"],
            accept_multiple_files=True,
            key="gs_files",
        )

        if not uploaded:
            st.info("Upload at least 2 faculty score files to begin.")
            return

        if len(uploaded) < 2:
            st.warning("At least 2 files are required for cross-session analysis.")
            return
        if len(uploaded) > 10:
            st.error("Maximum of 10 files allowed. Please remove some files.")
            return

        # --- Session labels ---
        st.subheader("Session Labels")
        st.caption("Provide a label for each file (e.g., '2023 Spring', '2024 Fall').")
        labels = []
        for i, f in enumerate(uploaded):
            lbl = st.text_input(
                f"Label for {f.name}",
                value=f"Session {i + 1}",
                key=f"gs_label_{i}",
            )
            labels.append(lbl)

        st.divider()

        # --- Load sessions ---
        if st.button("Analyze", type="primary", key="gs_analyze"):
            sessions = []
            with st.spinner("Loading faculty score files..."):
                for i, f in enumerate(uploaded):
                    path = _save_upload(f)
                    try:
                        session = load_faculty_session(path, gs_type_id, labels[i])
                        sessions.append(session)
                    except Exception as exc:
                        st.error(f"Error loading {f.name}: {exc}")
                        return

            err = validate_sessions(sessions)
            if err:
                st.error(err)
                return

            st.session_state["gs_sessions"] = sessions
            st.session_state["gs_type_id"] = gs_type_id

    # --- Display results if sessions are loaded ---
    if "gs_sessions" not in st.session_state:
        return

    sessions = st.session_state["gs_sessions"]
    gs_type_id = st.session_state["gs_type_id"]

    # Compute statistics
    stats = compute_cross_session_stats(sessions)
    st.session_state["gs_stats"] = stats

    # --- Step 2: Statistical Analysis ---
    st.divider()
    st.subheader("Cross-Session Statistics")

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Sessions", stats.session_count)
    mcol2.metric("Total Students", stats.total_students)
    mcol3.metric("Assessment Type", gs_type_options.get(gs_type_id, gs_type_id))

    # Per-section stats table
    rows = []
    for sec in stats.sections:
        ss = stats.section_stats.get(sec)
        if ss is None:
            continue
        row = {
            "Section": sec,
            "Mean": round(ss.mean, 2),
            "Median": round(ss.median, 2),
            "Std Dev": round(ss.std, 2),
            "Min": round(ss.min_score, 2),
            "Max": round(ss.max_score, 2),
        }
        for lbl in stats.session_labels:
            m = ss.per_session_means.get(lbl)
            row[lbl] = round(m, 2) if m is not None else ""
        rows.append(row)

    if rows:
        stats_df = pd.DataFrame(rows)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Bar chart of per-section means
    chart_data = pd.DataFrame({
        "Section": stats.sections,
        "Mean Score": [
            round(stats.section_stats[s].mean, 2)
            for s in stats.sections
        ],
    }).set_index("Section")
    st.bar_chart(chart_data)

    # Per-session comparison
    with st.expander("Per-Session Means (drift view)"):
        drift_rows = []
        for sec in stats.sections:
            ss = stats.section_stats.get(sec)
            if ss is None:
                continue
            row = {"Section": sec}
            for lbl in stats.session_labels:
                row[lbl] = round(ss.per_session_means.get(lbl, 0), 2)
            drift_rows.append(row)
        if drift_rows:
            st.dataframe(pd.DataFrame(drift_rows), use_container_width=True, hide_index=True)

    # --- Cultural Consensus Analysis ---
    st.divider()
    st.subheader("Cultural Consensus Analysis")
    st.caption(
        "Uses Cultural Consensus Theory (Romney, Weller, Batchelder 1986) "
        "to estimate session competence and a consensus-weighted answer key."
    )

    consensus = None
    if stats.session_count >= 2:
        try:
            consensus = compute_consensus_analysis(sessions, stats)
            st.session_state["gs_consensus"] = consensus
        except Exception as exc:
            st.warning(f"Consensus analysis could not be computed: {exc}")

    if consensus:
        # Model fit metrics
        fcol1, fcol2, fcol3 = st.columns(3)
        ratio_display = (
            f"{consensus.eigenvalue_ratio:.2f}"
            if consensus.eigenvalue_ratio != float("inf")
            else "\u221e (perfect)"
        )
        fcol1.metric("Eigenvalue Ratio (\u03bb\u2081/\u03bb\u2082)", ratio_display)
        fcol2.metric("Model Fit", consensus.fit_label)
        fcol3.metric(
            "Single Culture",
            "Yes" if consensus.single_culture_holds else "No",
        )

        # Interpretation
        if consensus.eigenvalue_ratio >= 5.0 or consensus.eigenvalue_ratio == float("inf"):
            st.success(
                "Strong single-culture fit. Faculty across sessions share a consistent "
                "scoring standard. The consensus answer key is highly reliable."
            )
        elif consensus.eigenvalue_ratio >= 3.0:
            st.info(
                "Adequate single-culture fit. Faculty generally share a scoring standard, "
                "though some variation exists."
            )
        else:
            st.warning(
                "Weak fit (\u03bb\u2081/\u03bb\u2082 < 3). Multiple scoring cultures may exist. "
                "The consensus answer key should be interpreted with caution."
            )

        if consensus.has_negative_loadings:
            st.warning(
                "One or more sessions have negative first-factor loadings, "
                "indicating they may apply a fundamentally different scoring standard."
            )

        # Eigenvalues table
        with st.expander("Eigenvalues"):
            ev_rows = []
            for i, ev in enumerate(consensus.eigenvalues):
                ev_rows.append({
                    "Component": i + 1,
                    "Eigenvalue": round(ev, 4),
                    "% Variance": round(
                        ev / sum(consensus.eigenvalues) * 100, 1
                    ) if sum(consensus.eigenvalues) > 0 else 0,
                })
            st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)

        # Session competence + first-factor loadings
        with st.expander("Session Competence & First-Factor Loadings", expanded=True):
            comp_rows = []
            for label in consensus.session_competence:
                loading = consensus.first_factor_loadings[label]
                comp_rows.append({
                    "Session": label,
                    "First-Factor Loading": round(loading, 4),
                    "Sign": "+" if loading >= 0 else "\u2212 (divergent)",
                    "Competence Weight": round(
                        consensus.session_competence[label], 4
                    ),
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        # Consensus answer key
        with st.expander("Consensus Answer Key", expanded=True):
            key_rows = []
            for sec in consensus.consensus_means:
                div = consensus.divergence[sec]
                key_rows.append({
                    "Section": sec,
                    "Consensus Mean": round(consensus.consensus_means[sec], 3),
                    "Simple Mean": round(consensus.simple_means[sec], 3),
                    "Difference": round(div, 3),
                    "Note": "Divergent" if div > 0.25 else "",
                })
            st.dataframe(pd.DataFrame(key_rows), use_container_width=True, hide_index=True)

    # --- Step 3: LLM Bias Analysis ---
    st.divider()
    st.subheader("LLM Bias Analysis")

    # Model selector (simplified)
    gs_model_options = list(MODEL_INFO.keys())
    gs_default_idx = gs_model_options.index("gemini-2.5-flash") if "gemini-2.5-flash" in gs_model_options else 0
    gs_model = st.radio(
        "Choose a model for bias analysis",
        gs_model_options,
        index=gs_default_idx,
        format_func=lambda m: f"{m} ({PROVIDER_LABELS[MODEL_INFO[m]['provider']]})",
        key="gs_model_select",
    )

    gs_provider = MODEL_INFO[gs_model]["provider"]
    gs_env_var = API_KEY_ENV_VARS[gs_provider]
    gs_has_key = bool(os.environ.get(gs_env_var, "").strip())

    if gs_has_key:
        st.success(f"API key found in environment (`{gs_env_var}`)")
        gs_api_key_input = None
    else:
        gs_api_key_input = st.text_input(
            f"{PROVIDER_LABELS[gs_provider]} API Key",
            type="password",
            key="gs_api_key",
            help=f"Set {gs_env_var} environment variable to avoid entering this each time.",
        )

    if st.button("Run Bias Analysis", key="gs_run_bias"):
        if not gs_has_key and not gs_api_key_input:
            st.error("Please provide an API key.")
            return

        if gs_api_key_input and not gs_has_key:
            _save_api_key_to_env(gs_env_var, gs_api_key_input)

        with st.spinner("Running LLM bias analysis..."):
            try:
                caller = create_caller(gs_provider, gs_model)
                messages = build_bias_prompt(stats, gs_type_id)
                response = call_llm(caller, messages, temperature=0.3, top_p=1.0)
                result = parse_bias_response(response)
                st.session_state["gs_bias_result"] = result
            except Exception as exc:
                st.error(f"LLM analysis failed: {exc}")
                return

    if "gs_bias_result" in st.session_state:
        bias = st.session_state["gs_bias_result"]

        st.markdown(f"**Summary:** {bias.summary}")

        if bias.systematic_biases:
            with st.expander("Systematic Biases", expanded=True):
                for item in bias.systematic_biases:
                    st.markdown(
                        f"- **{item.get('section', 'N/A')}** "
                        f"({item.get('direction', '')}, {item.get('magnitude', '')}): "
                        f"{item.get('description', '')}"
                    )

        if bias.section_patterns:
            with st.expander("Section Patterns"):
                for item in bias.section_patterns:
                    st.markdown(
                        f"- **{item.get('section', 'N/A')}**: {item.get('description', '')}"
                    )

        if bias.drift_over_years:
            with st.expander("Year-over-Year Drift"):
                for item in bias.drift_over_years:
                    st.markdown(
                        f"- **{item.get('section', 'N/A')}** "
                        f"({item.get('direction', '')}): {item.get('description', '')}"
                    )

        if bias.distribution_anomalies:
            with st.expander("Distribution Anomalies"):
                for item in bias.distribution_anomalies:
                    st.markdown(
                        f"- **{item.get('section', 'N/A')}**: {item.get('description', '')}"
                    )

        if bias.outlier_patterns:
            with st.expander("Outlier Patterns"):
                for item in bias.outlier_patterns:
                    st.markdown(f"- {item.get('description', '')}")

        if bias.recommendations:
            with st.expander("Recommendations", expanded=True):
                for rec in bias.recommendations:
                    st.markdown(f"- {rec}")

    # --- Step 4: Faculty Review & Benchmark Input ---
    st.divider()
    st.subheader("Establish Gold Standard Benchmarks")
    st.caption(
        "Review the analysis above and enter benchmark score ranges for each section. "
        "These define the expected faculty scoring corridor for this assessment."
    )

    benchmark_data = {}
    sections = stats.sections
    gs_consensus = st.session_state.get("gs_consensus")

    for sec in sections:
        ss = stats.section_stats.get(sec)
        with st.expander(f"Section: {sec}", expanded=False):
            if ss:
                stats_line = (
                    f"**Stats:** mean={ss.mean:.2f}, median={ss.median:.2f}, "
                    f"std={ss.std:.2f}, range=[{ss.min_score:.1f}\u2013{ss.max_score:.1f}]"
                )
                if gs_consensus and sec in gs_consensus.consensus_means:
                    stats_line += (
                        f" | **Consensus:** {gs_consensus.consensus_means[sec]:.2f}"
                    )
                st.markdown(stats_line)

            # Use consensus mean as center if available, else simple mean
            if gs_consensus and sec in gs_consensus.consensus_means:
                center = gs_consensus.consensus_means[sec]
            elif ss:
                center = ss.mean
            else:
                center = 0.0
            margin = ss.std if ss else 1.0

            bcol1, bcol2 = st.columns(2)
            with bcol1:
                bmin = st.number_input(
                    "Benchmark Min",
                    value=round(center - margin, 1),
                    step=0.5,
                    key=f"gs_bmin_{sec}",
                )
            with bcol2:
                bmax = st.number_input(
                    "Benchmark Max",
                    value=round(center + margin, 1),
                    step=0.5,
                    key=f"gs_bmax_{sec}",
                )

            notes = st.text_area(
                "Notes / rationale",
                key=f"gs_notes_{sec}",
                height=68,
            )
            approved = st.text_input(
                "Approved by (faculty name)",
                key=f"gs_approved_{sec}",
            )

            benchmark_data[sec] = {
                "min": bmin,
                "max": bmax,
                "notes": notes,
                "approved_by": approved,
            }

    # --- Step 5: Generate & Download ---
    st.divider()
    if st.button("Generate Gold Standard", type="primary", key="gs_generate"):
        from datetime import date as _date

        bias_result = st.session_state.get("gs_bias_result", BiasAnalysisResult())
        benchmark = GoldStandardBenchmark(
            assessment_type_id=gs_type_id,
            created_date=_date.today().isoformat(),
            sections=benchmark_data,
            bias_findings=bias_result,
            stats=stats,
            consensus=st.session_state.get("gs_consensus"),
            version=1,
        )

        excel_bytes = generate_benchmark_excel(benchmark)
        json_str = generate_benchmark_json(benchmark)

        st.session_state["gs_excel"] = excel_bytes
        st.session_state["gs_json"] = json_str
        st.session_state["gs_benchmark"] = benchmark

    if "gs_benchmark" in st.session_state:
        bm = st.session_state["gs_benchmark"]
        st.success("Gold standard benchmark generated.")

        # Preview
        preview_rows = []
        for sec, info in bm.sections.items():
            preview_rows.append({
                "Section": sec,
                "Min": info["min"],
                "Max": info["max"],
                "Notes": info.get("notes", ""),
                "Approved By": info.get("approved_by", ""),
            })
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button(
                "Download Excel",
                data=st.session_state["gs_excel"],
                file_name=f"gold_standard_{bm.assessment_type_id}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="gs_download_excel",
            )
        with dcol2:
            st.download_button(
                "Download JSON",
                data=st.session_state["gs_json"],
                file_name=f"gold_standard_{bm.assessment_type_id}.json",
                mime="application/json",
                key="gs_download_json",
            )


# =========================================================================
# Tab 6 — Synthetic Data Generator
# =========================================================================


def tab_synthetic_generator():
    st.header("Synthetic Data Generator")
    st.markdown(
        "Generate realistic, end-to-end OSCE test data using independent LLM agents. "
        "Each **student** has a unique persona (background, academic level, writing style) "
        "and each **faculty rater** has a unique scoring profile. Optionally upload "
        "de-identified examples to ground the generation in your real data."
    )

    # --- Assessment type ---
    synth_type_options = {
        "kpsom_ipass": "KPSOM — I-PASS Handoff",
        "kpsom_documentation": "KPSOM — Clinical Documentation",
        "kpsom_ethics": "KPSOM — Ethics Open-Ended Questions",
        "uk_osce": "Standard OSCE (HPI / PEX / SUM / DDX / Plan)",
    }
    synth_type_id = st.selectbox(
        "Assessment type",
        list(synth_type_options.keys()),
        format_func=lambda k: synth_type_options[k],
        key="synth_type_select",
    )

    # --- Generation parameters ---
    st.subheader("Generation Parameters")
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        synth_n_sessions = st.slider(
            "Number of sessions (years)",
            min_value=1,
            max_value=5,
            value=2,
            key="synth_n_sessions",
            help="Each session gets a unique faculty rater persona.",
        )
    with pcol2:
        synth_n_students = st.slider(
            "Students per session",
            min_value=2,
            max_value=15,
            value=5,
            key="synth_n_students",
            help="Each student gets a unique persona with diverse background.",
        )
    with pcol3:
        synth_variability = st.slider(
            "Variability",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="synth_variability",
            help=(
                "**Low (0.0)**: Mostly moderate raters, tighter score ranges. "
                "**Medium (0.5)**: Realistic mix of lenient/moderate/strict raters. "
                "**High (1.0)**: Extreme rater differences, wide score spreads."
            ),
        )

    # --- Persistent example files per assessment type ---
    st.subheader("Example Files (persistent per assessment type)")
    st.caption(
        "Upload de-identified files from a real administration. These are saved to "
        "disk and persist across restarts and generations. The generator uses them "
        "as templates for format, tone, and detail level — but generates entirely "
        "new clinical content."
    )

    _slot_labels = {
        "rubric": ("Example Rubric", ".docx or .xlsx", ["docx", "xlsx"]),
        "student_notes": ("Example Student Notes", ".xlsx", ["xlsx"]),
        "faculty_scores": ("Example Faculty Scores", ".xlsx", ["xlsx"]),
    }

    excol1, excol2, excol3 = st.columns(3)
    for col, slot in zip([excol1, excol2, excol3], _EXAMPLE_SLOTS):
        label, ext_hint, ext_list = _slot_labels[slot]
        with col:
            saved_name = _get_example_display_name(synth_type_id, slot)
            if saved_name:
                st.success(f"**{label}:** {saved_name}", icon="\u2705")
                if st.button(
                    f"Remove {label.lower()}",
                    key=f"synth_rm_{slot}_{synth_type_id}",
                ):
                    _remove_example(synth_type_id, slot)
                    st.rerun()
            else:
                st.info(f"**{label}:** None saved", icon="\u2139\ufe0f")

            new_file = st.file_uploader(
                f"Upload {label.lower()} ({ext_hint})",
                type=ext_list,
                key=f"synth_upload_{slot}_{synth_type_id}",
            )
            if new_file is not None and new_file.name != saved_name:
                _save_example(synth_type_id, slot, new_file)
                st.rerun()

    # --- Model selection ---
    st.subheader("LLM Configuration")
    synth_model_options = {
        "gemini-2.5-flash": ("Google", "google"),
        "gemini-2.5-pro": ("Google", "google"),
        "gpt-4o": ("OpenAI", "openai"),
        "gpt-4o-mini": ("OpenAI", "openai"),
        "claude-sonnet-4-6": ("Anthropic", "anthropic"),
        "claude-haiku-4-5": ("Anthropic", "anthropic"),
    }
    synth_model = st.radio(
        "Model",
        list(synth_model_options.keys()),
        format_func=lambda k: f"{k} ({synth_model_options[k][0]})",
        horizontal=True,
        key="synth_model",
    )
    synth_provider = synth_model_options[synth_model][1]

    # API key
    synth_env_var = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }[synth_provider]
    synth_has_key = bool(os.environ.get(synth_env_var))

    if not synth_has_key:
        synth_api_key = st.text_input(
            f"{synth_provider.title()} API Key",
            type="password",
            key="synth_api_key",
        )
    else:
        synth_api_key = None
        st.success(f"{synth_env_var} detected in environment.")

    synth_temperature = st.slider(
        "Base temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        key="synth_temperature",
        help="Base creativity level. Variability further scales this.",
    )

    # --- Generate ---
    st.divider()
    total_calls = synth_n_sessions * (synth_n_students + 2)
    st.caption(
        f"This will make approximately **{total_calls} LLM calls** "
        f"({synth_n_sessions} sessions × ({synth_n_students} students + rubric + scoring))."
    )

    if st.button("Generate Synthetic Data", type="primary", key="synth_generate"):
        if not synth_has_key and not synth_api_key:
            st.error("Please provide an API key.")
            return

        if synth_api_key and not synth_has_key:
            _save_api_key_to_env(synth_env_var, synth_api_key)

        # Load example files from persistent storage
        example_rubric_text = None
        example_notes = None
        example_scores = None

        rubric_path = _get_example_path(synth_type_id, "rubric")
        if rubric_path:
            if rubric_path.endswith(".docx"):
                example_rubric_text = convert_docx_to_text(rubric_path)
            else:
                df = pd.read_excel(rubric_path)
                example_rubric_text = df.to_string()

        notes_path = _get_example_path(synth_type_id, "student_notes")
        if notes_path:
            df = pd.read_excel(notes_path)
            if len(df) > 0:
                example_notes = {
                    col: str(df[col].iloc[0])
                    for col in df.columns
                    if pd.notna(df[col].iloc[0])
                }

        scores_path = _get_example_path(synth_type_id, "faculty_scores")
        if scores_path:
            try:
                from gold_standard import load_faculty_session
                loaded = load_faculty_session(scores_path, synth_type_id, "example")
                example_scores = loaded.scores
            except Exception:
                st.warning("Could not parse example scores file. Proceeding without.")

        # Create LLM caller — synthetic generation needs higher token
        # limits because rubric JSON responses are large.
        import config as _cfg
        _cfg.MODEL = synth_model
        _cfg.PROVIDER = synth_provider
        _prev_max_tokens = _cfg.MAX_TOKENS
        _cfg.MAX_TOKENS = max(_cfg.MAX_TOKENS, 16384)
        caller = create_caller(synth_provider, synth_model)

        sessions = []
        errors = []
        progress_bar = st.progress(0, text="Starting generation...")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _gen_session(s_idx):
            """Generate one session. No Streamlit calls — runs in threads."""
            return generate_synthetic_session(
                type_id=synth_type_id,
                session_index=s_idx,
                n_students=synth_n_students,
                variability=synth_variability,
                llm_caller=caller,
                temperature=synth_temperature,
                example_rubric_text=example_rubric_text,
                example_notes=example_notes,
                example_scores=example_scores,
                seed=42 + s_idx,
                progress_callback=None,  # no Streamlit from threads
            )

        if synth_n_sessions == 1:
            # Single session — run on main thread with progress
            def progress_cb(step_name, current, total):
                frac = min(current / max(total, 1), 0.99)
                progress_bar.progress(frac, text=f"Session 1: {step_name}")

            try:
                session = generate_synthetic_session(
                    type_id=synth_type_id,
                    session_index=0,
                    n_students=synth_n_students,
                    variability=synth_variability,
                    llm_caller=caller,
                    temperature=synth_temperature,
                    example_rubric_text=example_rubric_text,
                    example_notes=example_notes,
                    example_scores=example_scores,
                    seed=42,
                    progress_callback=progress_cb,
                )
                sessions.append(session)
            except Exception as exc:
                errors.append((1, exc))
                logger.exception("Synthetic generation failed")
        else:
            # Multiple sessions — run in parallel
            progress_bar.progress(0.05, text=f"Generating {synth_n_sessions} sessions in parallel...")
            with ThreadPoolExecutor(max_workers=min(synth_n_sessions, 4)) as pool:
                future_to_idx = {
                    pool.submit(_gen_session, s_idx): s_idx
                    for s_idx in range(synth_n_sessions)
                }
                done_count = 0
                for future in as_completed(future_to_idx):
                    s_idx = future_to_idx[future]
                    try:
                        sessions.append(future.result())
                    except Exception as exc:
                        errors.append((s_idx + 1, exc))
                        logger.exception("Synthetic generation failed for session %d", s_idx + 1)
                    done_count += 1
                    progress_bar.progress(
                        min(done_count / synth_n_sessions, 0.99),
                        text=f"Completed {done_count}/{synth_n_sessions} sessions",
                    )

            # Sort by session index so order is deterministic
            sessions.sort(key=lambda s: s.label)

        for s_num, exc in errors:
            st.error(f"Session {s_num} generation failed: {exc}")

        _cfg.MAX_TOKENS = _prev_max_tokens
        progress_bar.progress(1.0, text="Complete!")
        if sessions:
            # Build metadata about which example files grounded this generation
            examples_used = []
            rubric_name = _get_example_display_name(synth_type_id, "rubric")
            notes_name = _get_example_display_name(synth_type_id, "student_notes")
            scores_name = _get_example_display_name(synth_type_id, "faculty_scores")
            if rubric_name:
                examples_used.append(f"Rubric: {rubric_name}")
            if notes_name:
                examples_used.append(f"Student notes: {notes_name}")
            if scores_name:
                examples_used.append(f"Faculty scores: {scores_name}")

            generation_entry = {
                "sessions": sessions,
                "type_id": synth_type_id,
                "examples_used": examples_used,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": synth_model,
                "n_students": synth_n_students,
            }

            # Append to history (most recent first), keep up to 10
            history = st.session_state.get("synth_history", [])
            history.insert(0, generation_entry)
            st.session_state["synth_history"] = history[:10]

    # --- Display recent generations as collapsible expanders ---
    if "synth_history" not in st.session_state:
        return

    history = st.session_state["synth_history"]
    if not history:
        return

    st.divider()
    st.subheader("Recent Generations")
    st.caption(f"Showing {len(history)} most recent generation(s) (max 10).")

    for gen_idx, generation in enumerate(history):
        sessions = generation["sessions"]
        gen_type_id = generation["type_id"]
        gen_meta = SYNTH_TYPE_META[gen_type_id]
        examples_used = generation["examples_used"]
        timestamp = generation["timestamp"]
        model_name = generation["model"]
        n_students = generation["n_students"]

        # Build header for the generation expander
        session_labels = ", ".join(s.label for s in sessions)
        type_label = synth_type_options.get(gen_type_id, gen_type_id)
        gen_header = (
            f"{type_label} — {len(sessions)} session(s) "
            f"({session_labels}) — {timestamp}"
        )

        with st.expander(gen_header, expanded=(gen_idx == 0)):
            # --- Example provenance banner ---
            if examples_used:
                st.info(
                    "**Grounded in your examples:** "
                    + " | ".join(examples_used)
                )
            else:
                st.warning(
                    "**No example files provided.** This generation used only "
                    "the LLM's general medical knowledge — it was not grounded "
                    "in your real rubrics, notes, or scores."
                )

            st.caption(f"Model: {model_name} · Students/session: {n_students}")

            # Use sub-tabs for sessions within this generation
            session_tabs = st.tabs([s.label for s in sessions])

            for tab, session in zip(session_tabs, sessions):
                with tab:
                    # --- Faculty persona ---
                    with st.expander(
                        f"Faculty Rater: {session.faculty.name}",
                        expanded=False,
                    ):
                        st.markdown(
                            f"**Specialty:** {session.faculty.specialty}  \n"
                            f"**Experience:** {session.faculty.years_experience} years  \n"
                            f"**Scoring tendency:** {session.faculty.scoring_tendency} — "
                            f"{session.faculty.background_note}  \n"
                            f"**Focus areas:** {', '.join(session.faculty.focus_areas)}"
                        )

                    # --- Rubric ---
                    safe_lbl = session.label.replace(" ", "_")
                    with st.expander("Rubric", expanded=False):
                        st.markdown(rubric_to_display_text(session.rubric))
                        rcol1, rcol2 = st.columns(2)
                        with rcol1:
                            st.download_button(
                                "Download Rubric (.xlsx)",
                                data=rubric_to_excel(session.rubric, gen_type_id),
                                file_name=f"synth_{safe_lbl}_rubric.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"synth_dl_rubric_{gen_idx}_{safe_lbl}",
                            )
                        if gen_type_id == "uk_osce":
                            with rcol2:
                                st.download_button(
                                    "Download Answer Key (.xlsx)",
                                    data=answer_key_to_excel(session.rubric, gen_type_id),
                                    file_name=f"synth_{safe_lbl}_answer_key.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"synth_dl_ak_{gen_idx}_{safe_lbl}",
                                )

                    # --- Student notes preview ---
                    with st.expander("Student Notes", expanded=True):
                        notes_rows = []
                        for sid in sorted(session.student_notes.keys()):
                            student = next(
                                (s for s in session.students if s.student_id == sid),
                                None,
                            )
                            row = {"Student": sid}
                            if student:
                                row["Level"] = student.academic_level.split("—")[0].strip()
                            for sec in session.sections:
                                display = gen_meta["sections"][sec][0]
                                text = session.student_notes[sid].get(sec, "")
                                row[display] = (
                                    (text[:120] + "...") if len(text) > 120 else text
                                )
                            notes_rows.append(row)
                        st.dataframe(
                            pd.DataFrame(notes_rows),
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.download_button(
                            "Download Student Notes (.xlsx)",
                            data=student_notes_to_excel(session),
                            file_name=f"synth_{safe_lbl}_student_notes.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"synth_dl_notes_{gen_idx}_{safe_lbl}",
                        )

                    # --- Faculty scores preview ---
                    with st.expander("Faculty Scores", expanded=True):
                        score_rows = []
                        for sid in sorted(session.faculty_scores.keys()):
                            row = {"Student": sid}
                            for sec in session.sections:
                                display = gen_meta["sections"][sec][0]
                                row[display] = session.faculty_scores[sid].get(sec, "")
                            score_rows.append(row)
                        st.dataframe(
                            pd.DataFrame(score_rows),
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.download_button(
                            "Download Faculty Scores (.xlsx)",
                            data=faculty_scores_to_excel(session),
                            file_name=f"synth_{safe_lbl}_faculty_scores.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"synth_dl_scores_{gen_idx}_{safe_lbl}",
                        )

                    # --- Student personas ---
                    with st.expander("Student Personas", expanded=False):
                        for student in session.students:
                            st.markdown(
                                f"**Student {student.student_id}**  \n"
                                f"Background: {student.background}  \n"
                                f"Level: {student.academic_level}  \n"
                                f"Style: {student.writing_style}  \n"
                                f"Strengths: {', '.join(student.clinical_strengths)}  \n"
                                f"Weaknesses: {', '.join(student.clinical_weaknesses)}"
                            )
                            st.divider()

                    # --- Full session download ---
                    st.download_button(
                        f"Download Full Session ({session.label}) as ZIP",
                        data=session_to_zip(session),
                        file_name=f"synth_{safe_lbl}_complete.zip",
                        mime="application/zip",
                        key=f"synth_dl_full_{gen_idx}_{safe_lbl}",
                    )

            # --- Download all sessions in this generation ---
            if len(sessions) > 1:
                st.download_button(
                    "Download All Sessions (.zip)",
                    data=all_sessions_to_zip(sessions),
                    file_name=f"synth_{gen_type_id}_all_sessions.zip",
                    mime="application/zip",
                    key=f"synth_dl_all_{gen_idx}",
                )

            # --- Feed into Gold Standard ---
            st.divider()
            st.caption("Load these sessions into Gold Standard for analysis.")
            if st.button(
                "Load into Gold Standard",
                key=f"synth_to_gs_{gen_idx}",
            ):
                from gold_standard import SessionData
                gs_sessions = []
                for session in sessions:
                    gs_sessions.append(SessionData(
                        label=session.label,
                        assessment_type_id=session.assessment_type_id,
                        sections=session.sections,
                        scores=session.faculty_scores,
                        student_count=len(session.faculty_scores),
                    ))
                st.session_state["gs_sessions"] = gs_sessions
                st.session_state["gs_type_id"] = gen_type_id
                st.success(
                    f"Loaded {len(gs_sessions)} synthetic sessions into Gold Standard. "
                    "Switch to the Gold Standard tab to view analysis."
                )


# =========================================================================
# Main app layout
# =========================================================================
st.title("OSCE Grader")
st.caption("AI-powered grading for medical student post-encounter notes")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Grade Notes",
    "Analysis Dashboard",
    "Flagged Items",
    "Convert Rubric",
    "Gold Standard",
    "Synthetic Data",
])

with tab1:
    tab_grade_notes()

with tab2:
    tab_analysis()

with tab3:
    tab_flagged()

with tab4:
    tab_convert()

with tab5:
    tab_gold_standard()

with tab6:
    tab_synthetic_generator()
