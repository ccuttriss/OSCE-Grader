"""Psychometric analysis modules for OSCE Grader.

Provides seven analysis modules that complement the existing Gold Standard
pipeline in gold_standard.py:

1. Inter-Rater Reliability (ICC, kappa, Krippendorff's alpha)
2. Generalizability Theory (variance decomposition, D-study)
3. Classical Test Theory Item Analysis (difficulty, discrimination, point-biserial)
4. Distribution Analysis (normality, skewness, ceiling/floor effects)
5. Agreement Analysis (Bland-Altman, SEM)
6. DIF Analysis (Mantel-Haenszel, logistic regression)
7. Response Pattern Clustering
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model extensions
# ---------------------------------------------------------------------------


@dataclass
class RaterSessionData:
    """Per-rater scores for inter-rater reliability and G-theory."""

    session_label: str
    assessment_type_id: str
    sections: list[str]
    rater_scores: dict[str, dict[int, dict]]
    # rater_id -> {student_id -> {section: score}}


@dataclass
class DistributionResult:
    """Results from distribution analysis."""

    normality_tests: dict[str, dict] = field(default_factory=dict)
    # section -> {shapiro_w, shapiro_p, dagostino_p, is_normal}
    skewness: dict[str, float] = field(default_factory=dict)
    kurtosis: dict[str, float] = field(default_factory=dict)
    ceiling_pct: dict[str, float] = field(default_factory=dict)
    floor_pct: dict[str, float] = field(default_factory=dict)
    flags: list[dict] = field(default_factory=list)


@dataclass
class ItemAnalysisResult:
    """Results from classical test theory item analysis."""

    item_difficulty: dict[str, float] = field(default_factory=dict)
    item_discrimination: dict[str, float] = field(default_factory=dict)
    point_biserial: dict[str, dict] = field(default_factory=dict)
    # section -> {r_pb, p_value}
    alpha_if_deleted: dict[str, float] = field(default_factory=dict)
    overall_alpha: float = 0.0
    flags: list[dict] = field(default_factory=list)


@dataclass
class ClusterResult:
    """Results from response pattern clustering."""

    student_clusters: dict[int, int] = field(default_factory=dict)
    n_clusters: int = 0
    cluster_profiles: dict[int, dict[str, float]] = field(default_factory=dict)
    silhouette_score: float = 0.0
    cluster_sizes: dict[int, int] = field(default_factory=dict)
    interpretation: list[str] = field(default_factory=list)


@dataclass
class InterRaterResult:
    """Results from inter-rater reliability analysis."""

    icc_results: dict[str, dict] = field(default_factory=dict)
    # section -> {icc_type, value, ci_low, ci_high, p_value}
    cohens_kappa: dict[str, float] = field(default_factory=dict)
    fleiss_kappa: dict[str, float] = field(default_factory=dict)
    krippendorff_alpha: dict[str, float] = field(default_factory=dict)
    cronbach_alpha: dict[str, dict] = field(default_factory=dict)
    # section -> {alpha, alpha_if_deleted: dict}
    interpretation: dict[str, str] = field(default_factory=dict)


@dataclass
class AgreementResult:
    """Results from Bland-Altman agreement analysis."""

    mean_diff: dict[str, float] = field(default_factory=dict)
    sd_diff: dict[str, float] = field(default_factory=dict)
    loa_upper: dict[str, float] = field(default_factory=dict)
    loa_lower: dict[str, float] = field(default_factory=dict)
    sem: dict[str, float] = field(default_factory=dict)
    within_loa_pct: dict[str, float] = field(default_factory=dict)
    proportional_bias: dict[str, bool] = field(default_factory=dict)


@dataclass
class GTheoryResult:
    """Results from generalizability theory analysis."""

    variance_components: dict[str, float] = field(default_factory=dict)
    relative_g_coefficient: float = 0.0
    absolute_g_coefficient: float = 0.0
    sem: float = 0.0
    d_study: dict[int, float] = field(default_factory=dict)
    # {n_raters: projected_g_coefficient}


@dataclass
class DIFResult:
    """Results from differential item functioning analysis."""

    mh_results: dict[str, dict] = field(default_factory=dict)
    # section -> {chi2, p_value, odds_ratio, mh_delta, ets_class}
    logistic_results: dict[str, dict] = field(default_factory=dict)
    # section -> {uniform_p, nonuniform_p, r2_change}
    flagged_items: list[dict] = field(default_factory=list)


@dataclass
class PsychometricBundle:
    """Complete psychometric analysis results for export."""

    item_analysis: ItemAnalysisResult | None = None
    distribution: DistributionResult | None = None
    irr: InterRaterResult | None = None
    g_theory: GTheoryResult | None = None
    agreement: AgreementResult | None = None
    dif: DIFResult | None = None
    clusters: ClusterResult | None = None


# ---------------------------------------------------------------------------
# Helper: pool scores across sessions
# ---------------------------------------------------------------------------


def pool_session_scores(
    sessions: list,  # list[SessionData] from gold_standard
) -> dict[int, dict[str, float]]:
    """Pool student scores from multiple sessions into one dict.

    If student IDs overlap across sessions, offset them to keep unique.
    Returns {student_id: {section: score}}.
    """
    pooled: dict[int, dict[str, float]] = {}
    offset = 0
    for session in sessions:
        for sid, sdict in session.scores.items():
            new_id = sid + offset
            # Only include non-None scores
            pooled[new_id] = {
                k: v for k, v in sdict.items() if v is not None
            }
        offset += max(session.scores.keys(), default=0) + 1
    return pooled


def _scores_to_array(
    scores: dict[int, dict[str, float]],
    sections: list[str],
) -> tuple[np.ndarray, list[int]]:
    """Convert scores dict to (n_students x n_sections) array.

    Returns (matrix, student_ids). Missing values are NaN.
    """
    student_ids = sorted(scores.keys())
    matrix = np.full((len(student_ids), len(sections)), np.nan)
    for i, sid in enumerate(student_ids):
        for j, sec in enumerate(sections):
            val = scores[sid].get(sec)
            if val is not None:
                matrix[i, j] = val
    return matrix, student_ids


# ---------------------------------------------------------------------------
# Module 4: Distribution Analysis
# ---------------------------------------------------------------------------


def compute_distribution_analysis(
    scores: dict[int, dict[str, float]],
    max_scores: dict[str, float],
    sections: list[str],
) -> DistributionResult:
    """Analyze score distributions for each section.

    Tests normality, computes skewness/kurtosis, and detects ceiling/floor
    effects.
    """
    result = DistributionResult()

    for sec in sections:
        values = [
            scores[sid][sec]
            for sid in scores
            if sec in scores[sid] and scores[sid][sec] is not None
        ]
        if len(values) < 3:
            continue

        arr = np.array(values, dtype=float)
        max_score = max_scores.get(sec, arr.max())

        # Normality tests
        normality = {}
        if len(arr) >= 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    w_stat, w_p = scipy_stats.shapiro(arr)
                    normality["shapiro_w"] = round(float(w_stat), 4)
                    normality["shapiro_p"] = round(float(w_p), 4)
                except Exception:
                    normality["shapiro_w"] = None
                    normality["shapiro_p"] = None

        if len(arr) >= 8:
            try:
                _, dag_p = scipy_stats.normaltest(arr)
                normality["dagostino_p"] = round(float(dag_p), 4)
            except Exception:
                normality["dagostino_p"] = None
        else:
            normality["dagostino_p"] = None

        # Consider normal if Shapiro p > 0.05
        shapiro_p = normality.get("shapiro_p")
        normality["is_normal"] = shapiro_p is not None and shapiro_p > 0.05
        result.normality_tests[sec] = normality

        # Skewness and kurtosis
        skew_val = float(scipy_stats.skew(arr, nan_policy="omit"))
        kurt_val = float(scipy_stats.kurtosis(arr, nan_policy="omit"))
        result.skewness[sec] = round(skew_val, 3)
        result.kurtosis[sec] = round(kurt_val, 3)

        # Ceiling and floor percentages
        ceiling_count = np.sum(arr >= max_score - 0.01)
        floor_count = np.sum(arr <= 0.01)
        n = len(arr)
        result.ceiling_pct[sec] = round(float(ceiling_count / n * 100), 1)
        result.floor_pct[sec] = round(float(floor_count / n * 100), 1)

        # Flagging
        if result.ceiling_pct[sec] > 15:
            result.flags.append({
                "section": sec,
                "issue": "ceiling_effect",
                "detail": (
                    f"{result.ceiling_pct[sec]}% of students scored at "
                    f"maximum ({max_score})"
                ),
            })
        if result.floor_pct[sec] > 15:
            result.flags.append({
                "section": sec,
                "issue": "floor_effect",
                "detail": (
                    f"{result.floor_pct[sec]}% of students scored at "
                    f"minimum (0)"
                ),
            })
        if abs(skew_val) > 1.0:
            direction = "negatively" if skew_val < 0 else "positively"
            result.flags.append({
                "section": sec,
                "issue": "skewed_distribution",
                "detail": (
                    f"Substantially {direction} skewed "
                    f"(skewness = {skew_val:.2f})"
                ),
            })
        if not normality["is_normal"] and shapiro_p is not None:
            result.flags.append({
                "section": sec,
                "issue": "non_normal",
                "detail": (
                    f"Distribution is non-normal "
                    f"(Shapiro-Wilk p = {shapiro_p:.4f})"
                ),
            })

    return result


# ---------------------------------------------------------------------------
# Module 3: Classical Test Theory Item Analysis
# ---------------------------------------------------------------------------


def compute_item_analysis(
    scores: dict[int, dict[str, float]],
    max_scores: dict[str, float],
    sections: list[str],
) -> ItemAnalysisResult:
    """Compute CTT item statistics for each section.

    Calculates item difficulty, discrimination (D-index), corrected
    point-biserial correlation, Cronbach's alpha, and alpha-if-deleted.
    """
    result = ItemAnalysisResult()
    matrix, student_ids = _scores_to_array(scores, sections)

    if matrix.shape[0] < 5:
        logger.warning("Too few students (%d) for meaningful item analysis", matrix.shape[0])
        return result

    # Replace NaN with 0 for total score computation
    matrix_filled = np.nan_to_num(matrix, nan=0.0)
    total_scores = matrix_filled.sum(axis=1)

    for j, sec in enumerate(sections):
        col = matrix_filled[:, j]
        max_score = max_scores.get(sec, col.max() if col.max() > 0 else 1.0)

        # Item difficulty: mean score / max possible
        p = float(np.mean(col) / max_score) if max_score > 0 else 0.0
        result.item_difficulty[sec] = round(p, 3)

        # Item discrimination (D-index): top 27% mean - bottom 27% mean
        n = len(total_scores)
        n_group = max(1, int(n * 0.27))
        sorted_idx = np.argsort(total_scores)
        bottom_idx = sorted_idx[:n_group]
        top_idx = sorted_idx[-n_group:]
        upper_mean = float(np.mean(col[top_idx]))
        lower_mean = float(np.mean(col[bottom_idx]))
        d_index = (upper_mean - lower_mean) / max_score if max_score > 0 else 0.0
        result.item_discrimination[sec] = round(d_index, 3)

        # Corrected point-biserial: correlate section score with total minus section
        corrected_total = total_scores - col
        if np.std(col) > 1e-10 and np.std(corrected_total) > 1e-10:
            r_pb, p_val = scipy_stats.pearsonr(col, corrected_total)
            result.point_biserial[sec] = {
                "r_pb": round(float(r_pb), 3),
                "p_value": round(float(p_val), 4),
            }
        else:
            result.point_biserial[sec] = {"r_pb": 0.0, "p_value": 1.0}

    # Cronbach's alpha (overall and if-deleted)
    try:
        import pingouin as pg
        df = pd.DataFrame(matrix_filled, columns=sections)
        alpha_result = pg.cronbach_alpha(df)
        result.overall_alpha = round(float(alpha_result[0]), 3)

        # Alpha-if-deleted for each section
        for j, sec in enumerate(sections):
            remaining = [s for s in sections if s != sec]
            if len(remaining) >= 2:
                df_reduced = df[remaining]
                a_del = pg.cronbach_alpha(df_reduced)
                result.alpha_if_deleted[sec] = round(float(a_del[0]), 3)
    except Exception as exc:
        logger.warning("Cronbach's alpha computation failed: %s", exc)

    # Flagging
    for sec in sections:
        p = result.item_difficulty.get(sec, 0.5)
        d = result.item_discrimination.get(sec, 0.3)
        r_pb = result.point_biserial.get(sec, {}).get("r_pb", 0.3)
        a_del = result.alpha_if_deleted.get(sec)

        if p > 0.90:
            result.flags.append({
                "section": sec, "issue": "too_easy",
                "detail": f"Item difficulty p = {p:.2f} (> 0.90, ceiling effect)",
            })
        elif p < 0.15:
            result.flags.append({
                "section": sec, "issue": "too_hard",
                "detail": f"Item difficulty p = {p:.2f} (< 0.15, floor effect)",
            })

        if d < 0.20:
            result.flags.append({
                "section": sec, "issue": "low_discrimination",
                "detail": f"D-index = {d:.2f} (< 0.20, poor discrimination)",
            })

        if r_pb < 0.15:
            result.flags.append({
                "section": sec, "issue": "low_point_biserial",
                "detail": f"Point-biserial r = {r_pb:.2f} (< 0.15)",
            })

        if a_del is not None and a_del > result.overall_alpha + 0.01:
            result.flags.append({
                "section": sec, "issue": "decreases_reliability",
                "detail": (
                    f"Removing this section increases alpha from "
                    f"{result.overall_alpha:.3f} to {a_del:.3f}"
                ),
            })

    return result


# ---------------------------------------------------------------------------
# Module 7: Response Pattern Clustering
# ---------------------------------------------------------------------------


def compute_response_clusters(
    scores: dict[int, dict[str, float]],
    sections: list[str],
    max_clusters: int = 5,
    method: str = "agglomerative",
) -> ClusterResult:
    """Identify subgroups of students with similar response patterns.

    Uses agglomerative clustering by default with automatic k selection
    via silhouette analysis.
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    result = ClusterResult()
    matrix, student_ids = _scores_to_array(scores, sections)

    # Need at least 4 students for meaningful clustering
    if matrix.shape[0] < 4:
        logger.warning("Too few students (%d) for clustering", matrix.shape[0])
        return result

    # Replace NaN with column means for clustering
    col_means = np.nanmean(matrix, axis=0)
    for j in range(matrix.shape[1]):
        nan_mask = np.isnan(matrix[:, j])
        matrix[nan_mask, j] = col_means[j]

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(matrix)

    # Find optimal k via silhouette analysis
    max_k = min(max_clusters, matrix.shape[0] - 1)
    if max_k < 2:
        return result

    best_k = 2
    best_score = -1.0

    for k in range(2, max_k + 1):
        try:
            if method == "kmeans":
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
            else:
                model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    # Fit final model with best k
    if method == "kmeans":
        model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    else:
        model = AgglomerativeClustering(n_clusters=best_k)
    labels = model.fit_predict(X)

    result.n_clusters = best_k
    result.silhouette_score = round(float(best_score), 3)

    # Map student IDs to cluster labels
    for i, sid in enumerate(student_ids):
        result.student_clusters[sid] = int(labels[i])

    # Compute cluster profiles (mean score per section)
    for cluster_id in range(best_k):
        mask = labels == cluster_id
        result.cluster_sizes[cluster_id] = int(mask.sum())
        profile = {}
        for j, sec in enumerate(sections):
            profile[sec] = round(float(matrix[mask, j].mean()), 2)
        result.cluster_profiles[cluster_id] = profile

    # Auto-generate interpretations
    overall_means = {sec: float(matrix[:, j].mean()) for j, sec in enumerate(sections)}
    for cluster_id in range(best_k):
        profile = result.cluster_profiles[cluster_id]
        size = result.cluster_sizes[cluster_id]

        # Find which sections are notably above/below overall mean
        above = []
        below = []
        for sec in sections:
            diff = profile[sec] - overall_means[sec]
            sec_std = float(matrix[:, sections.index(sec)].std())
            if sec_std > 1e-10:
                z = diff / sec_std
                if z > 0.5:
                    above.append(sec)
                elif z < -0.5:
                    below.append(sec)

        description = f"Cluster {cluster_id + 1} (n={size}): "
        if above and below:
            description += f"Strong in {', '.join(above)}; weak in {', '.join(below)}"
        elif above:
            description += f"High performers, especially in {', '.join(above)}"
        elif below:
            description += f"Struggles with {', '.join(below)}"
        else:
            description += "Average across all sections"
        result.interpretation.append(description)

    return result


# ---------------------------------------------------------------------------
# Module 1: Inter-Rater Reliability
# ---------------------------------------------------------------------------


def compute_inter_rater_reliability(
    rater_scores: dict[str, dict[int, dict[str, float]]],
    sections: list[str],
) -> InterRaterResult:
    """Compute inter-rater reliability metrics.

    Parameters
    ----------
    rater_scores:
        rater_id -> {student_id -> {section: score}}
    sections:
        List of section keys to analyze.
    """
    result = InterRaterResult()
    rater_ids = sorted(rater_scores.keys())
    n_raters = len(rater_ids)

    if n_raters < 2:
        logger.warning("Need at least 2 raters for IRR analysis")
        return result

    # Find common students (scored by all raters)
    student_sets = [set(rater_scores[r].keys()) for r in rater_ids]
    common_students = sorted(set.intersection(*student_sets))
    if len(common_students) < 3:
        logger.warning("Too few common students (%d) for IRR", len(common_students))
        return result

    for sec in sections:
        # Build rater x student matrix for this section
        rater_matrix = np.full((n_raters, len(common_students)), np.nan)
        for i, rater_id in enumerate(rater_ids):
            for j, sid in enumerate(common_students):
                val = rater_scores[rater_id].get(sid, {}).get(sec)
                if val is not None:
                    rater_matrix[i, j] = val

        # Skip if too many missing values
        valid_cols = ~np.any(np.isnan(rater_matrix), axis=0)
        if valid_cols.sum() < 3:
            continue

        rater_matrix = rater_matrix[:, valid_cols]

        # ICC via pingouin
        try:
            import pingouin as pg
            # Build long-format DataFrame for pingouin
            rows = []
            for i, rater_id in enumerate(rater_ids):
                for j in range(rater_matrix.shape[1]):
                    rows.append({
                        "targets": j,
                        "raters": rater_id,
                        "ratings": rater_matrix[i, j],
                    })
            icc_df = pd.DataFrame(rows)
            icc_table = pg.intraclass_corr(
                data=icc_df, targets="targets",
                raters="raters", ratings="ratings",
            )
            # ICC(2,1) — two-way random, single measures
            icc21 = icc_table[icc_table["Type"] == "ICC2"]
            if not icc21.empty:
                row = icc21.iloc[0]
                # pingouin uses "CI95" (not "CI95%") in recent versions
                ci_key = "CI95%" if "CI95%" in row.index else "CI95"
                ci = row.get(ci_key, [0, 1])
                result.icc_results[sec] = {
                    "icc_type": "ICC(2,1)",
                    "value": round(float(row["ICC"]), 3),
                    "ci_low": round(float(ci[0]), 3),
                    "ci_high": round(float(ci[1]), 3),
                    "p_value": round(float(row["pval"]), 4),
                }
        except Exception as exc:
            logger.warning("ICC computation failed for %s: %s", sec, exc)

        # Cohen's kappa (pairwise, averaged) for 2 raters
        # or averaged pairwise for 3+ raters
        try:
            from sklearn.metrics import cohen_kappa_score
            kappas = []
            for i in range(n_raters):
                for k in range(i + 1, n_raters):
                    r1 = rater_matrix[i]
                    r2 = rater_matrix[k]
                    # Discretize to integers for kappa (required by sklearn)
                    r1_int = np.round(r1).astype(int)
                    r2_int = np.round(r2).astype(int)
                    if len(np.unique(np.concatenate([r1_int, r2_int]))) < 2:
                        continue
                    kap = cohen_kappa_score(
                        r1_int, r2_int, weights="linear",
                    )
                    kappas.append(kap)
            if kappas:
                result.cohens_kappa[sec] = round(float(np.mean(kappas)), 3)
        except Exception as exc:
            logger.warning("Cohen's kappa failed for %s: %s", sec, exc)

        # Krippendorff's alpha
        try:
            import krippendorff as kripp
            alpha = kripp.alpha(
                rater_matrix,
                level_of_measurement="ordinal",
            )
            result.krippendorff_alpha[sec] = round(float(alpha), 3)
        except Exception as exc:
            logger.warning("Krippendorff's alpha failed for %s: %s", sec, exc)

        # Interpretation based on ICC
        icc_val = result.icc_results.get(sec, {}).get("value", 0)
        if icc_val >= 0.75:
            result.interpretation[sec] = "excellent"
        elif icc_val >= 0.60:
            result.interpretation[sec] = "good"
        elif icc_val >= 0.40:
            result.interpretation[sec] = "fair"
        else:
            result.interpretation[sec] = "poor"

    # Cronbach's alpha across raters (treating raters as "items")
    for sec in sections:
        try:
            import pingouin as pg
            rater_cols = {}
            for rater_id in rater_ids:
                col = []
                for sid in common_students:
                    val = rater_scores[rater_id].get(sid, {}).get(sec)
                    col.append(val if val is not None else np.nan)
                rater_cols[rater_id] = col
            df = pd.DataFrame(rater_cols).dropna()
            if len(df) >= 3 and len(df.columns) >= 2:
                alpha_val = pg.cronbach_alpha(df)
                result.cronbach_alpha[sec] = {
                    "alpha": round(float(alpha_val[0]), 3),
                }
        except Exception as exc:
            logger.warning("Cronbach alpha for raters failed for %s: %s", sec, exc)

    return result


# ---------------------------------------------------------------------------
# Module 5: Agreement Analysis (Bland-Altman + SEM)
# ---------------------------------------------------------------------------


def compute_agreement_analysis(
    scores_a: dict[int, dict[str, float]],
    scores_b: dict[int, dict[str, float]],
    sections: list[str],
    reliability: dict[str, float] | None = None,
) -> AgreementResult:
    """Compute Bland-Altman agreement statistics between two scoring methods.

    Parameters
    ----------
    scores_a:
        Method A scores (e.g., faculty). student_id -> {section: score}
    scores_b:
        Method B scores (e.g., AI grader). student_id -> {section: score}
    sections:
        List of section keys to analyze.
    reliability:
        Optional dict of section -> reliability coefficient for SEM.
    """
    result = AgreementResult()

    # Find common students
    common = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    if len(common) < 3:
        logger.warning("Too few paired observations (%d) for agreement analysis", len(common))
        return result

    for sec in sections:
        a_vals = []
        b_vals = []
        for sid in common:
            va = scores_a.get(sid, {}).get(sec)
            vb = scores_b.get(sid, {}).get(sec)
            if va is not None and vb is not None:
                a_vals.append(va)
                b_vals.append(vb)

        if len(a_vals) < 3:
            continue

        a_arr = np.array(a_vals)
        b_arr = np.array(b_vals)

        diffs = a_arr - b_arr
        means = (a_arr + b_arr) / 2.0

        mean_d = float(np.mean(diffs))
        sd_d = float(np.std(diffs, ddof=1))

        result.mean_diff[sec] = round(mean_d, 3)
        result.sd_diff[sec] = round(sd_d, 3)
        result.loa_upper[sec] = round(mean_d + 1.96 * sd_d, 3)
        result.loa_lower[sec] = round(mean_d - 1.96 * sd_d, 3)

        # % within limits of agreement
        within = np.sum(
            (diffs >= mean_d - 1.96 * sd_d) & (diffs <= mean_d + 1.96 * sd_d)
        )
        result.within_loa_pct[sec] = round(float(within / len(diffs) * 100), 1)

        # Proportional bias: correlation between differences and means
        if np.std(means) > 1e-10:
            r, p = scipy_stats.pearsonr(means, diffs)
            result.proportional_bias[sec] = abs(r) > 0.3 and p < 0.05
        else:
            result.proportional_bias[sec] = False

        # SEM
        if reliability and sec in reliability:
            rel = reliability[sec]
            sd_scores = float(np.std(a_arr, ddof=1))
            if rel < 1.0:
                result.sem[sec] = round(sd_scores * math.sqrt(1 - rel), 3)
            else:
                result.sem[sec] = 0.0
        else:
            # Use SD of differences / sqrt(2) as rough SEM estimate
            result.sem[sec] = round(sd_d / math.sqrt(2), 3)

    return result


# ---------------------------------------------------------------------------
# Module 2: Generalizability Theory
# ---------------------------------------------------------------------------


def compute_g_theory(
    scores_3d: np.ndarray,
    facet_labels: list[str] | None = None,
    n_raters_for_dstudy: list[int] | None = None,
) -> GTheoryResult:
    """Compute G-theory variance components and coefficients.

    Parameters
    ----------
    scores_3d:
        3D array of shape (n_persons, n_raters, n_items).
        Can contain NaN for missing data.
    facet_labels:
        Names for the facets, e.g. ["person", "rater", "item"].
    n_raters_for_dstudy:
        List of rater counts to project in D-study. Default [1..10].
    """
    result = GTheoryResult()

    if facet_labels is None:
        facet_labels = ["person", "rater", "item"]

    n_p, n_r, n_i = scores_3d.shape

    if n_p < 2 or n_r < 2 or n_i < 1:
        logger.warning("Insufficient data for G-theory: %s", scores_3d.shape)
        return result

    # Use manual ANOVA-based variance decomposition
    # (generalizit package API is unstable; manual approach is validated)
    result = _manual_g_theory(scores_3d, facet_labels)

    # D-study: project G-coefficient for different numbers of raters
    if n_raters_for_dstudy is None:
        n_raters_for_dstudy = list(range(1, 11))

    var_p = result.variance_components.get("person", 0)
    var_r = result.variance_components.get("rater", 0)
    var_i = result.variance_components.get("item", 0)
    var_pr = result.variance_components.get("person:rater", 0)
    var_pi = result.variance_components.get("person:item", 0)
    var_ri = result.variance_components.get("rater:item", 0)
    var_pri = result.variance_components.get("person:rater:item", 0)

    for nr in n_raters_for_dstudy:
        # Relative error variance
        rel_error = var_pr / nr + var_pi / n_i + var_pri / (nr * n_i)
        if var_p + rel_error > 0:
            g_rel = var_p / (var_p + rel_error)
        else:
            g_rel = 0.0
        result.d_study[nr] = round(float(g_rel), 3)

    # SEM
    rel_error = var_pr / n_r + var_pi / n_i + var_pri / (n_r * n_i)
    result.sem = round(float(math.sqrt(max(0, rel_error))), 3)

    return result


def _manual_g_theory(
    scores_3d: np.ndarray,
    facet_labels: list[str],
) -> GTheoryResult:
    """Manual G-theory via ANOVA-based variance decomposition.

    Uses the random-effects ANOVA approach for a fully crossed
    person x rater x item design.
    """
    result = GTheoryResult()
    n_p, n_r, n_i = scores_3d.shape

    # Replace NaN with grand mean for computation
    grand_mean = np.nanmean(scores_3d)
    filled = np.where(np.isnan(scores_3d), grand_mean, scores_3d)

    # Marginal means
    mean_p = filled.mean(axis=(1, 2))   # person means
    mean_r = filled.mean(axis=(0, 2))   # rater means
    mean_i = filled.mean(axis=(0, 1))   # item means

    # Sum of squares
    ss_p = n_r * n_i * np.sum((mean_p - grand_mean) ** 2)
    ss_r = n_p * n_i * np.sum((mean_r - grand_mean) ** 2)
    ss_i = n_p * n_r * np.sum((mean_i - grand_mean) ** 2)

    mean_pr = filled.mean(axis=2)
    mean_pi = filled.mean(axis=1)
    mean_ri = filled.mean(axis=0)

    ss_pr = n_i * np.sum((mean_pr - mean_p[:, None] - mean_r[None, :] + grand_mean) ** 2)
    ss_pi = n_r * np.sum((mean_pi - mean_p[:, None] - mean_i[None, :] + grand_mean) ** 2)
    ss_ri = n_p * np.sum((mean_ri - mean_r[:, None] - mean_i[None, :] + grand_mean) ** 2)

    ss_total = np.sum((filled - grand_mean) ** 2)
    ss_pri = ss_total - ss_p - ss_r - ss_i - ss_pr - ss_pi - ss_ri

    # Degrees of freedom
    df_p = n_p - 1
    df_r = n_r - 1
    df_i = n_i - 1
    df_pr = df_p * df_r
    df_pi = df_p * df_i
    df_ri = df_r * df_i
    df_pri = df_p * df_r * df_i

    # Mean squares
    ms_p = ss_p / max(df_p, 1)
    ms_r = ss_r / max(df_r, 1)
    ms_i = ss_i / max(df_i, 1)
    ms_pr = ss_pr / max(df_pr, 1)
    ms_pi = ss_pi / max(df_pi, 1)
    ms_ri = ss_ri / max(df_ri, 1)
    ms_pri = ss_pri / max(df_pri, 1)

    # Variance components (can be negative; set floor at 0)
    var_p = max(0, (ms_p - ms_pr - ms_pi + ms_pri) / (n_r * n_i))
    var_r = max(0, (ms_r - ms_pr - ms_ri + ms_pri) / (n_p * n_i))
    var_i = max(0, (ms_i - ms_pi - ms_ri + ms_pri) / (n_p * n_r))
    var_pr = max(0, (ms_pr - ms_pri) / n_i)
    var_pi = max(0, (ms_pi - ms_pri) / n_r)
    var_ri = max(0, (ms_ri - ms_pri) / n_p)
    var_pri = max(0, ms_pri)

    result.variance_components = {
        "person": round(var_p, 4),
        "rater": round(var_r, 4),
        "item": round(var_i, 4),
        "person:rater": round(var_pr, 4),
        "person:item": round(var_pi, 4),
        "rater:item": round(var_ri, 4),
        "person:rater:item": round(var_pri, 4),
    }

    # Relative G-coefficient
    rel_error = var_pr / n_r + var_pi / n_i + var_pri / (n_r * n_i)
    if var_p + rel_error > 0:
        result.relative_g_coefficient = round(var_p / (var_p + rel_error), 3)

    # Absolute G-coefficient
    abs_error = (
        var_r / n_r + var_i / n_i + var_pr / n_r + var_pi / n_i
        + var_ri / (n_r * n_i) + var_pri / (n_r * n_i)
    )
    if var_p + abs_error > 0:
        result.absolute_g_coefficient = round(var_p / (var_p + abs_error), 3)

    return result


# ---------------------------------------------------------------------------
# Module 6: DIF Analysis
# ---------------------------------------------------------------------------


def compute_dif_analysis(
    scores: dict[int, dict[str, float]],
    group_membership: dict[int, str],
    sections: list[str],
    focal_group: str | None = None,
) -> DIFResult:
    """Compute Differential Item Functioning via Mantel-Haenszel and logistic regression.

    Parameters
    ----------
    scores:
        student_id -> {section: score}
    group_membership:
        student_id -> group label (e.g., "male"/"female")
    sections:
        Section keys to test for DIF.
    focal_group:
        Which group to treat as focal. If None, uses the smaller group.
    """
    result = DIFResult()

    # Identify groups
    groups = set(group_membership.values())
    if len(groups) != 2:
        logger.warning("DIF requires exactly 2 groups, found %d", len(groups))
        return result

    groups = sorted(groups)
    if focal_group is None:
        # Use smaller group as focal
        counts = {g: sum(1 for v in group_membership.values() if v == g) for g in groups}
        focal_group = min(counts, key=counts.get)

    reference_group = [g for g in groups if g != focal_group][0]

    # Common students with group labels
    common = sorted(
        sid for sid in scores
        if sid in group_membership
    )

    if len(common) < 10:
        logger.warning("Too few students (%d) for DIF analysis", len(common))
        return result

    # Compute total scores
    all_sections_set = set(sections)
    totals = {}
    for sid in common:
        total = sum(
            scores[sid].get(s, 0) or 0
            for s in sections
        )
        totals[sid] = total

    for sec in sections:
        # Get section scores and total-minus-section for matching
        focal_data = []
        ref_data = []
        for sid in common:
            val = scores[sid].get(sec)
            if val is None:
                continue
            matching_var = totals[sid] - val  # total minus this section
            grp = group_membership[sid]
            entry = {"score": val, "matching": matching_var, "group": grp}
            if grp == focal_group:
                focal_data.append(entry)
            else:
                ref_data.append(entry)

        if len(focal_data) < 5 or len(ref_data) < 5:
            continue

        # Mantel-Haenszel DIF
        try:
            mh = _mantel_haenszel_dif(focal_data, ref_data, sec)
            result.mh_results[sec] = mh
        except Exception as exc:
            logger.warning("MH DIF failed for %s: %s", sec, exc)

        # Logistic regression DIF
        try:
            lr = _logistic_regression_dif(focal_data, ref_data, sec)
            result.logistic_results[sec] = lr
        except Exception as exc:
            logger.warning("Logistic DIF failed for %s: %s", sec, exc)

        # Flag significant DIF items
        mh_result = result.mh_results.get(sec, {})
        lr_result = result.logistic_results.get(sec, {})
        ets_class = mh_result.get("ets_class", "A")
        uniform_p = lr_result.get("uniform_p", 1.0)

        if ets_class in ("B", "C") or uniform_p < 0.05:
            result.flagged_items.append({
                "section": sec,
                "ets_class": ets_class,
                "mh_delta": mh_result.get("mh_delta", 0),
                "uniform_p": uniform_p,
                "detail": (
                    f"DIF detected: ETS class {ets_class}, "
                    f"MH delta = {mh_result.get('mh_delta', 'N/A')}"
                ),
            })

    return result


def _mantel_haenszel_dif(
    focal_data: list[dict],
    ref_data: list[dict],
    section: str,
) -> dict:
    """Compute Mantel-Haenszel DIF statistic for one section.

    Stratifies by total score quintiles, then computes common odds ratio.
    """
    all_data = focal_data + ref_data
    matching_scores = [d["matching"] for d in all_data]

    # Create quintile strata based on matching variable
    try:
        quintiles = np.percentile(matching_scores, [20, 40, 60, 80])
    except Exception:
        return {"chi2": 0, "p_value": 1, "odds_ratio": 1, "mh_delta": 0, "ets_class": "A"}

    def get_stratum(val):
        for i, q in enumerate(quintiles):
            if val <= q:
                return i
        return len(quintiles)

    # For each stratum, compute 2x2 table
    # Using median split on section score for "high"/"low"
    sec_scores = [d["score"] for d in all_data]
    median_score = np.median(sec_scores)

    numerator = 0.0
    denominator = 0.0
    chi2_num = 0.0
    total_n = 0

    strata_tables = []
    for stratum in range(len(quintiles) + 1):
        focal_high = sum(1 for d in focal_data if get_stratum(d["matching"]) == stratum and d["score"] > median_score)
        focal_low = sum(1 for d in focal_data if get_stratum(d["matching"]) == stratum and d["score"] <= median_score)
        ref_high = sum(1 for d in ref_data if get_stratum(d["matching"]) == stratum and d["score"] > median_score)
        ref_low = sum(1 for d in ref_data if get_stratum(d["matching"]) == stratum and d["score"] <= median_score)

        n = focal_high + focal_low + ref_high + ref_low
        if n < 2:
            continue

        strata_tables.append((focal_high, focal_low, ref_high, ref_low, n))
        # MH common odds ratio components
        numerator += focal_high * ref_low / n
        denominator += focal_low * ref_high / n

    if denominator < 0.001:
        odds_ratio = float("inf") if numerator > 0 else 1.0
    else:
        odds_ratio = numerator / denominator

    # MH delta (log odds ratio on ETS scale)
    if odds_ratio > 0 and odds_ratio != float("inf"):
        mh_delta = -2.35 * math.log(odds_ratio)
    else:
        mh_delta = 0.0

    # ETS classification
    abs_delta = abs(mh_delta)
    if abs_delta < 1.0:
        ets_class = "A"
    elif abs_delta < 1.5:
        ets_class = "B"
    else:
        ets_class = "C"

    # Chi-square test using scipy
    # Aggregate into single 2x2 table for simplicity
    total_focal_high = sum(t[0] for t in strata_tables)
    total_focal_low = sum(t[1] for t in strata_tables)
    total_ref_high = sum(t[2] for t in strata_tables)
    total_ref_low = sum(t[3] for t in strata_tables)

    table = np.array([
        [total_focal_high, total_focal_low],
        [total_ref_high, total_ref_low],
    ])
    if table.min() >= 0 and table.sum() > 0:
        try:
            chi2, p_val, _, _ = scipy_stats.chi2_contingency(table, correction=True)
        except Exception:
            chi2, p_val = 0.0, 1.0
    else:
        chi2, p_val = 0.0, 1.0

    return {
        "chi2": round(float(chi2), 3),
        "p_value": round(float(p_val), 4),
        "odds_ratio": round(float(odds_ratio), 3) if odds_ratio != float("inf") else None,
        "mh_delta": round(float(mh_delta), 3),
        "ets_class": ets_class,
    }


def _logistic_regression_dif(
    focal_data: list[dict],
    ref_data: list[dict],
    section: str,
) -> dict:
    """Compute logistic regression DIF for one section.

    Compares three nested models to detect uniform and non-uniform DIF.
    """
    import statsmodels.api as sm

    # Build dataset
    all_data = focal_data + ref_data
    matching = np.array([d["matching"] for d in all_data])
    group = np.array([1 if d["group"] == focal_data[0]["group"] else 0 for d in all_data])
    # Binarize section score at median for logistic regression
    sec_scores = np.array([d["score"] for d in all_data])
    median_score = np.median(sec_scores)
    outcome = (sec_scores > median_score).astype(float)

    # Standardize matching variable
    if np.std(matching) > 1e-10:
        matching_z = (matching - np.mean(matching)) / np.std(matching)
    else:
        matching_z = matching - np.mean(matching)

    result = {"uniform_p": 1.0, "nonuniform_p": 1.0, "r2_change": 0.0}

    try:
        # Model 1: outcome ~ matching (no DIF)
        X1 = sm.add_constant(matching_z)
        m1 = sm.Logit(outcome, X1).fit(disp=0, method="bfgs", maxiter=100)

        # Model 2: outcome ~ matching + group (uniform DIF)
        X2 = sm.add_constant(np.column_stack([matching_z, group]))
        m2 = sm.Logit(outcome, X2).fit(disp=0, method="bfgs", maxiter=100)

        # Model 3: outcome ~ matching + group + matching*group (non-uniform DIF)
        interaction = matching_z * group
        X3 = sm.add_constant(np.column_stack([matching_z, group, interaction]))
        m3 = sm.Logit(outcome, X3).fit(disp=0, method="bfgs", maxiter=100)

        # Likelihood ratio tests
        lr_uniform = -2 * (m1.llf - m2.llf)
        lr_nonuniform = -2 * (m2.llf - m3.llf)

        result["uniform_p"] = round(float(scipy_stats.chi2.sf(lr_uniform, 1)), 4)
        result["nonuniform_p"] = round(float(scipy_stats.chi2.sf(lr_nonuniform, 1)), 4)

        # R-squared change (McFadden's pseudo R-squared)
        r2_m1 = 1 - m1.llf / m1.llnull if m1.llnull != 0 else 0
        r2_m2 = 1 - m2.llf / m2.llnull if m2.llnull != 0 else 0
        result["r2_change"] = round(float(r2_m2 - r2_m1), 4)

    except Exception as exc:
        logger.warning("Logistic regression DIF failed for %s: %s", section, exc)

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_psychometric_analysis(
    sessions: list,  # list[SessionData]
    sections: list[str],
    max_scores: dict[str, float],
    rater_data: RaterSessionData | None = None,
    ai_scores: dict[int, dict[str, float]] | None = None,
    demographics: dict[int, str] | None = None,
) -> PsychometricBundle:
    """Run all applicable psychometric analyses.

    Modules that require optional data (rater-level, AI scores, demographics)
    are skipped if that data is not provided.
    """
    bundle = PsychometricBundle()

    # Pool scores across sessions
    pooled = pool_session_scores(sessions)

    # Always available
    try:
        bundle.distribution = compute_distribution_analysis(pooled, max_scores, sections)
    except Exception as exc:
        logger.error("Distribution analysis failed: %s", exc)

    try:
        bundle.item_analysis = compute_item_analysis(pooled, max_scores, sections)
    except Exception as exc:
        logger.error("Item analysis failed: %s", exc)

    try:
        bundle.clusters = compute_response_clusters(pooled, sections)
    except Exception as exc:
        logger.error("Clustering failed: %s", exc)

    # Requires per-rater data
    if rater_data is not None:
        try:
            bundle.irr = compute_inter_rater_reliability(
                rater_data.rater_scores, sections,
            )
        except Exception as exc:
            logger.error("IRR analysis failed: %s", exc)

        # G-theory: build 3D array from rater data
        try:
            rater_ids = sorted(rater_data.rater_scores.keys())
            student_ids = sorted(set.union(
                *(set(rater_data.rater_scores[r].keys()) for r in rater_ids)
            ))
            scores_3d = np.full(
                (len(student_ids), len(rater_ids), len(sections)), np.nan,
            )
            for ri, rater_id in enumerate(rater_ids):
                for si, sid in enumerate(student_ids):
                    for ii, sec in enumerate(sections):
                        val = rater_data.rater_scores[rater_id].get(sid, {}).get(sec)
                        if val is not None:
                            scores_3d[si, ri, ii] = val
            bundle.g_theory = compute_g_theory(scores_3d)
        except Exception as exc:
            logger.error("G-theory analysis failed: %s", exc)

    # Requires paired AI vs faculty scores
    if ai_scores is not None:
        try:
            bundle.agreement = compute_agreement_analysis(
                pooled, ai_scores, sections,
            )
        except Exception as exc:
            logger.error("Agreement analysis failed: %s", exc)

    # Requires demographic groups
    if demographics is not None:
        try:
            bundle.dif = compute_dif_analysis(pooled, demographics, sections)
        except Exception as exc:
            logger.error("DIF analysis failed: %s", exc)

    return bundle
