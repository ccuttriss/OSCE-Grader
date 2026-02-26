# OSCE Grader Calibration Report

## Executive Overview

This report summarizes the prompt engineering and model selection process used to calibrate GPT-based OSCE grading against expert human graders. Through four iterations of tuning, we achieved **99% within-1-point agreement** with human graders, up from an initial baseline of 83%.

The calibration was performed using 14 de-identified student post-encounter notes graded across 6 sections (HPI, Physical Exam, Summary, Differential Diagnosis, Supporting Evidence, Plan), totaling 84 individual score comparisons. Each student had scores from two independent human graders serving as ground truth.

---

## Key Metrics Explained

### Within-1 Agreement (Primary Target)

This metric measures the percentage of GPT scores that fall within 1 point of the human grader average on a 1-4 scale. It is the most important metric for practical use because:

- **A 1-point tolerance is standard in inter-rater reliability studies.** Even expert human graders frequently disagree by 1 point on subjective assessments. Expecting exact agreement from an AI system would be an unrealistically high bar that human graders themselves do not meet.
- **Scores outside the 1-point window are clinically meaningful errors.** On a 4-point rubric, a 2-point discrepancy (e.g., GPT scores 2, humans score 4) could change a student's pass/fail outcome or misrepresent their competency.
- **90% within-1 is the minimum acceptable threshold.** This means at most 1 in 10 section scores may need manual review, making AI-assisted grading practical while maintaining quality.

### Bias (Systematic Error Direction)

Bias measures whether GPT grades consistently higher or lower than human graders. A negative bias means GPT is systematically harsher; a positive bias means it is more lenient.

- **Bias = -0.42** (final configuration) means GPT scores are, on average, 0.42 points lower than the human average. This is a mild systematic tendency toward strictness.
- **Why it matters:** If bias is large and consistent, students graded by GPT are systematically disadvantaged compared to students graded by humans. A bias near zero ensures fairness regardless of grading method.
- **Baseline bias was -0.88**, meaning GPT was giving scores nearly a full point below human graders on average. The tuned configuration cut this roughly in half.

### MAE (Mean Absolute Error)

MAE is the average magnitude of the difference between GPT and human scores, regardless of direction. Unlike bias, it captures both over-scoring and under-scoring.

- **MAE = 0.50** (final) means the average GPT score differs from the human average by half a point. On a 4-point scale, this represents strong agreement.
- **Baseline MAE was 0.90** (nearly a full point of average error). The tuned configuration reduced error by 44%.

---

## Iteration Results

| Metric | Baseline | Iter 1 | Iter 2 | Iter 3 | Iter 4 (Best) |
|---|---|---|---|---|---|
| **Model** | gpt-4o-mini | gpt-4o-mini | gpt-4o-mini | gpt-4o-mini | **gpt-4o** |
| **Temperature** | 0.5 | 0.3 | 0.3 | 0.1 | **0.3** |
| **Within-1** | 83% | 95% | 95% | 94% | **99%** |
| **MAE** | 0.90 | 0.79 | 0.71 | 0.77 | **0.50** |
| **Bias** | -0.88 | -0.77 | -0.63 | -0.71 | **-0.42** |
| **Exact Agreement** | 25% | 25% | 32% | 28% | **51%** |

---

## What Changed Between Iterations

### Baseline (Original Prompt)
The original prompt was a brief, generic instruction: *"I am a medical educator... score each section separately... provide a detailed explanation."* It gave GPT no guidance on how to interpret the answer key or rubric levels, resulting in overly literal, checklist-style grading.

**Root causes of poor performance:**
- GPT treated the answer key as a strict checklist, penalizing for any missing bullet point
- Paraphrasing was treated as incorrect (e.g., "abdominal pain" vs. "flank pain")
- Students were penalized for including extra relevant clinical information
- GPT would jump from a score of 4 directly to 2 for minor omissions

### Iteration 1: Holistic Grading Instructions
Added explicit guidelines telling GPT to: grade holistically rather than by checklist, accept paraphrasing and synonyms, not penalize for extra information, and reserve low scores for genuinely deficient responses.

**Impact:** Within-1 jumped from 83% to 95% (+12 points). This was the single largest improvement, demonstrating that prompt design is the most impactful lever.

### Iteration 2: Clinical Synonyms and Alternatives
Added targeted instructions for accepting clinical synonyms (e.g., "intense" = "severe", "throwing up" = "emesis") and reasonable alternative clinical approaches (e.g., CT scan vs. ultrasound for kidney stones).

**Impact:** MAE improved from 0.79 to 0.71, bias improved from -0.77 to -0.63, and exact agreement rose from 25% to 32%. The within-1 rate held at 95%.

### Iteration 3: Lower Temperature (0.1)
Tested whether reducing randomness (temperature 0.5 to 0.1) would improve consistency.

**Impact:** Performance slightly worsened (within-1 dropped to 94%). Very low temperature made the model more rigid and deterministic, but also more prone to anchoring on literal answer-key matching. This approach was abandoned.

### Iteration 4: Model Upgrade to gpt-4o
Switched from gpt-4o-mini to gpt-4o while keeping the iteration 2 prompt and temperature 0.3.

**Impact:** Every metric improved dramatically. Within-1 reached 99%, MAE dropped to 0.50, bias fell to -0.42, and exact agreement reached 51%. The full-size model demonstrated meaningfully better clinical reasoning and nuanced judgment. Only 1 of 84 scores was off by more than 1 point, and that single outlier was a case where GPT scored *higher* than both human graders (not harsher).

---

## Cost Analysis

Pricing is based on current OpenAI API rates (February 2026):

| | gpt-4o-mini | gpt-4o | Multiplier |
|---|---|---|---|
| Input (per 1M tokens) | $0.15 | $2.50 | 16.7x |
| Output (per 1M tokens) | $0.60 | $10.00 | 16.7x |

### Estimated Costs Per Run

Based on measured token usage from the 14-student calibration run (6 sections per student, 84 total API calls):

| Scale | gpt-4o-mini | gpt-4o | Additional Cost |
|---|---|---|---|
| **Per student** | $0.001 | $0.023 | +$0.022 |
| **14 students (1 run)** | $0.02 | $0.32 | +$0.30 |
| **100 students** | $0.14 | $2.31 | +$2.17 |
| **1,000 students** | $1.39 | $23.11 | +$21.72 |
| **1,000 runs (14,000 students)** | $19.41 | $323.57 | +$304.16 |

### Cost-Benefit Assessment

The gpt-4o model costs approximately **16.7x more** than gpt-4o-mini. However, in the context of educational assessment:

- **At $0.023 per student**, gpt-4o costs roughly 2 cents per student. Even at scale (1,000 students), the total is $23 — a fraction of the cost of a single human grader's time.
- **The accuracy gain is substantial:** 99% vs. 95% within-1 agreement means roughly 4 fewer scoring errors per 100 section scores. For a cohort of 100 students across 6 sections, that is approximately 24 fewer scores requiring manual review.
- **For most educational institutions**, the $21.72 difference per 1,000 students is negligible compared to the value of more accurate, human-aligned grading.

**Recommendation:** Use gpt-4o for production grading. The cost difference is minimal at typical class sizes, and the accuracy improvement is significant. Reserve gpt-4o-mini for rapid prototyping, prompt testing, or scenarios where cost is the primary constraint and 95% within-1 agreement is acceptable.

---

## Final Configuration

```python
MODEL = "gpt-4o"
TEMPERATURE = 0.3
TOP_P = 1.0
```

The full optimized grading prompt is stored in `scripts/config.py` under `GRADING_PROMPT`.

---

## Limitations and Recommendations

1. **Sample size:** Calibration was performed on 14 students from a single OSCE case (flank pain). Results should be validated on additional cases and larger cohorts before deploying at scale.

2. **Rubric dependency:** The prompt is designed for 4-point rubrics. Different scoring scales may require re-calibration.

3. **Ongoing monitoring:** We recommend running the `evaluate.py` script periodically against a held-out set of human-graded responses to detect any drift in model behavior as OpenAI updates their models.

4. **Human review:** Even at 99% within-1 agreement, edge cases will occur. We recommend flagging any GPT score of 1 or 2 for optional human review, as these are the scores most likely to diverge from human judgment.
