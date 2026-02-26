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
    grade_section_with_key,
    read_rubric_and_key,
    validate_input_columns,
)
from providers import create_caller

# Also import text extraction and LLM conversion from web worker
WEB_DIR = os.path.join(REPO_ROOT, "web")
sys.path.insert(0, WEB_DIR)
from grading_worker import CONVERT_PROMPT, compute_summary_stats, run_dry_run

from scripts.convert_rubric import convert_docx_to_text, convert_pdf_to_text

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

# TEST: Auto-deploy verification — remove after confirming
st.toast("Hello! Auto-deploy is working!")

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

    # --- File uploads ---
    col1, col2, col3 = st.columns(3)
    with col1:
        rubric_file = st.file_uploader("Rubric (.xlsx)", type=["xlsx"], key="rubric_upload")
    with col2:
        key_file = st.file_uploader("Answer Key (.xlsx)", type=["xlsx"], key="key_upload")
    with col3:
        notes_file = st.file_uploader("Student Notes (.xlsx)", type=["xlsx"], key="notes_upload")

    # --- Previews ---
    if rubric_file:
        with st.expander("Rubric preview", expanded=False):
            df_r = pd.read_excel(rubric_file)
            st.dataframe(df_r.head(5), use_container_width=True)
            rubric_file.seek(0)

    if key_file:
        with st.expander("Answer Key preview", expanded=False):
            df_k = pd.read_excel(key_file)
            st.dataframe(df_k.head(5), use_container_width=True)
            key_file.seek(0)

    detected_sections = []
    if notes_file:
        with st.expander("Student Notes preview", expanded=False):
            df_n = pd.read_excel(notes_file)
            st.dataframe(df_n.head(5), use_container_width=True)
            detected_sections = _detect_sections(df_n)
            if detected_sections:
                st.info(f"Detected sections: **{', '.join(s.upper() for s in detected_sections)}**")
            else:
                st.warning("No recognized section columns found.")
            notes_file.seek(0)

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
        if not rubric_file or not key_file or not notes_file:
            st.error("Please upload all three files (rubric, answer key, student notes).")
            return False
        if not has_env_key and not api_key_input:
            st.error(f"Please provide a {PROVIDER_LABELS[provider]} API key.")
            return False
        return True

    # --- Set API key if provided ---
    def _set_api_key():
        if api_key_input and not has_env_key:
            os.environ[env_var] = api_key_input

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

        rubric_path = _save_upload(rubric_file)
        key_path = _save_upload(key_file)
        notes_path = _save_upload(notes_file)

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

    # --- Grade ---
    if grade_clicked:
        if not _validate():
            return

        _set_api_key()
        _set_config()

        rubric_path = _save_upload(rubric_file)
        key_path = _save_upload(key_file)
        notes_path = _save_upload(notes_file)

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

        # Progress bar + status
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

        # Store results in session state for other tabs
        st.session_state["results_df"] = df
        st.session_state["results_sections"] = sections

        # Summary stats
        stats = compute_summary_stats(df, sections)
        if stats:
            st.subheader("Summary Statistics")
            st.dataframe(pd.DataFrame(stats), use_container_width=True)

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

    # Check for OpenAI API key (rubric conversion uses OpenAI)
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not env_key:
        convert_key = st.text_input(
            "OpenAI API Key (required for rubric conversion)",
            type="password",
            key="convert_api_key",
        )
        if not convert_key:
            st.warning("An OpenAI API key is required for LLM-based rubric conversion.")
            return
        os.environ["OPENAI_API_KEY"] = convert_key

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
                # Use OpenAI for conversion (matches web worker behavior)
                old_model = config.MODEL
                old_provider = config.PROVIDER
                config.MODEL = "gpt-4o"
                config.PROVIDER = "openai"

                caller = create_caller("openai")
                messages = [
                    {"role": "system", "content": CONVERT_PROMPT},
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
# Main app layout
# =========================================================================
st.title("OSCE Grader")
st.caption("AI-powered grading for medical student post-encounter notes")

tab1, tab2, tab3, tab4 = st.tabs([
    "Grade Notes",
    "Analysis Dashboard",
    "Flagged Items",
    "Convert Rubric",
])

with tab1:
    tab_grade_notes()

with tab2:
    tab_analysis()

with tab3:
    tab_flagged()

with tab4:
    tab_convert()
