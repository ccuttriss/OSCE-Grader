"""Evaluate LLM grading accuracy against human grader scores.

Reads a graded results Excel file that contains both human grader columns
(e.g. hpi_grader_1, hpi_grader_2) and LLM score columns (e.g. hpi_gpt_score),
then computes agreement metrics.

Usage:
    python scripts/evaluate.py results.xlsx
    python scripts/evaluate.py results.xlsx --model gpt-4o
    python scripts/evaluate.py results.xlsx --sections hpi pex sum ddx support plan
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def evaluate(results_path: str, sections: list[str] | None = None, model: str | None = None) -> dict:
    """Compute agreement metrics between GPT and human graders.

    Returns a dict with per-section and overall metrics.
    """
    df = pd.read_excel(results_path)

    # Auto-detect sections from columns if not specified
    if sections is None:
        sections = []
        for col in df.columns:
            if col.endswith("_gpt_score"):
                sec = col.replace("_gpt_score", "")
                # Only include if human grader columns exist too
                if f"{sec}_grader_1" in df.columns:
                    sections.append(sec)

    if not sections:
        print("ERROR: No sections found with both GPT and human grader scores.")
        sys.exit(1)

    results = {}
    all_diffs_g1 = []
    all_diffs_g2 = []
    all_diffs_avg = []

    print("=" * 80)
    print(f"EVALUATION REPORT — {results_path}")
    if model:
        print(f"Model: {model}")
    print(f"Students: {len(df)}  |  Sections: {', '.join(sections)}")
    print("=" * 80)

    for section in sections:
        gpt_col = f"{section}_gpt_score"
        g1_col = f"{section}_grader_1"
        g2_col = f"{section}_grader_2"

        if gpt_col not in df.columns:
            print(f"  SKIP {section}: no GPT score column")
            continue
        if g1_col not in df.columns:
            print(f"  SKIP {section}: no human grader columns")
            continue

        # Drop rows where GPT score is missing
        mask = df[gpt_col].notna()
        gpt = df.loc[mask, gpt_col].astype(float)
        g1 = df.loc[mask, g1_col].astype(float)
        g2 = df.loc[mask, g2_col].astype(float)
        human_avg = (g1 + g2) / 2

        diff_g1 = (gpt - g1).abs()
        diff_g2 = (gpt - g2).abs()
        diff_avg = (gpt - human_avg).abs()

        all_diffs_g1.extend(diff_g1.tolist())
        all_diffs_g2.extend(diff_g2.tolist())
        all_diffs_avg.extend(diff_avg.tolist())

        n = len(gpt)
        exact_g1 = (diff_g1 == 0).sum()
        exact_g2 = (diff_g2 == 0).sum()
        within1_g1 = (diff_g1 <= 1).sum()
        within1_g2 = (diff_g2 <= 1).sum()

        sec_results = {
            "n": n,
            "gpt_mean": gpt.mean(),
            "g1_mean": g1.mean(),
            "g2_mean": g2.mean(),
            "human_avg_mean": human_avg.mean(),
            "mae_vs_g1": diff_g1.mean(),
            "mae_vs_g2": diff_g2.mean(),
            "mae_vs_avg": diff_avg.mean(),
            "exact_agree_g1": exact_g1 / n * 100,
            "exact_agree_g2": exact_g2 / n * 100,
            "within1_g1": within1_g1 / n * 100,
            "within1_g2": within1_g2 / n * 100,
            "bias": (gpt - human_avg).mean(),  # negative = GPT grades harsher
        }
        results[section] = sec_results

        print(f"\n--- {section.upper()} ---")
        print(f"  Means:   GPT={gpt.mean():.2f}  G1={g1.mean():.2f}  G2={g2.mean():.2f}  HumanAvg={human_avg.mean():.2f}")
        print(f"  Bias:    {sec_results['bias']:+.2f} (negative = GPT harsher)")
        print(f"  MAE:     vs G1={diff_g1.mean():.2f}  vs G2={diff_g2.mean():.2f}  vs Avg={diff_avg.mean():.2f}")
        print(f"  Exact:   vs G1={sec_results['exact_agree_g1']:.0f}%  vs G2={sec_results['exact_agree_g2']:.0f}%")
        print(f"  Within1: vs G1={sec_results['within1_g1']:.0f}%  vs G2={sec_results['within1_g2']:.0f}%")

        # Show per-student breakdown if any are off by 2+
        big_misses = diff_avg[diff_avg > 1]
        if len(big_misses) > 0:
            print(f"  ⚠ {len(big_misses)} student(s) off by >1 from human avg:")
            for idx in big_misses.index:
                print(f"    Row {idx}: GPT={gpt[idx]:.0f}  G1={g1[idx]:.0f}  G2={g2[idx]:.0f}")

    # Overall summary
    if all_diffs_avg:
        total = len(all_diffs_avg)
        overall_mae = sum(all_diffs_avg) / total
        overall_within1 = sum(1 for d in all_diffs_avg if d <= 1) / total * 100
        overall_exact_g1 = sum(1 for d in all_diffs_g1 if d == 0) / total * 100
        overall_exact_g2 = sum(1 for d in all_diffs_g2 if d == 0) / total * 100
        overall_within1_g1 = sum(1 for d in all_diffs_g1 if d <= 1) / total * 100
        overall_within1_g2 = sum(1 for d in all_diffs_g2 if d <= 1) / total * 100
        overall_bias = sum(d for d in (pd.Series(all_diffs_avg))) / total  # unsigned
        # For signed bias, recalculate
        all_signed = []
        for section in sections:
            if section not in results:
                continue
            gpt_col = f"{section}_gpt_score"
            g1_col = f"{section}_grader_1"
            g2_col = f"{section}_grader_2"
            mask = df[gpt_col].notna()
            gpt = df.loc[mask, gpt_col].astype(float)
            g1 = df.loc[mask, g1_col].astype(float)
            g2 = df.loc[mask, g2_col].astype(float)
            human_avg = (g1 + g2) / 2
            all_signed.extend((gpt - human_avg).tolist())

        signed_bias = sum(all_signed) / len(all_signed)

        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        print(f"  Total score comparisons: {total}")
        print(f"  Overall MAE vs human avg: {overall_mae:.2f}")
        print(f"  Overall bias: {signed_bias:+.2f} (negative = GPT harsher)")
        print(f"  Exact agreement:  vs G1={overall_exact_g1:.0f}%  vs G2={overall_exact_g2:.0f}%")
        print(f"  Within-1 agreement: vs G1={overall_within1_g1:.0f}%  vs G2={overall_within1_g2:.0f}%")
        print(f"  Within-1 vs human avg: {overall_within1:.0f}%")
        print()

        # Pass/fail verdict
        if overall_within1 >= 90:
            print("  ✅ TARGET MET: ≥90% of scores within 1 point of human avg")
        elif overall_within1 >= 80:
            print(f"  🟡 CLOSE: {overall_within1:.0f}% within 1 point (target: 90%)")
        else:
            print(f"  ❌ NOT MET: {overall_within1:.0f}% within 1 point (target: 90%)")

        results["_overall"] = {
            "total": total,
            "mae": overall_mae,
            "bias": signed_bias,
            "within1_pct": overall_within1,
            "within1_g1_pct": overall_within1_g1,
            "within1_g2_pct": overall_within1_g2,
        }

        # --- Cost estimation ---
        if model:
            try:
                import config
                if model in config.MODEL_COSTS:
                    input_cost, output_cost = config.MODEL_COSTS[model]
                    # Rough estimate: ~800 input tokens + ~250 output tokens per section
                    est_input = total * 800
                    est_output = total * 250
                    est_cost = (est_input * input_cost + est_output * output_cost) / 1_000_000
                    per_student = est_cost / len(df) if len(df) > 0 else 0
                    print(f"\n  Cost estimate ({model}):")
                    print(f"    This run ({len(df)} students): ${est_cost:.4f}")
                    print(f"    Per student: ${per_student:.4f}")
                    print(f"    Per 1,000 students: ${per_student * 1000:.2f}")
                    results["_overall"]["est_cost"] = est_cost
                    results["_overall"]["est_cost_per_student"] = per_student
            except ImportError:
                pass  # config not available outside scripts/ context

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GPT grading accuracy against human graders."
    )
    parser.add_argument("results", help="Path to the graded results Excel file")
    parser.add_argument(
        "--sections",
        nargs="+",
        default=None,
        help="Section names to evaluate (auto-detected if omitted)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for report header and cost estimation",
    )
    args = parser.parse_args()
    evaluate(args.results, args.sections, args.model)


if __name__ == "__main__":
    main()
