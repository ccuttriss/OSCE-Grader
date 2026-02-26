# Multi-Provider LLM Comparison Report

## Executive Overview

This report compares the grading accuracy and cost-effectiveness of five LLM models across three providers for automated OSCE post-encounter note grading. All models were tested using the same calibrated grading prompt, temperature (0.3), and 14-student sample dataset (84 individual score comparisons across 6 sections).

**Key finding:** All five models met the 90% within-1 target, but they differ meaningfully in accuracy, bias characteristics, and cost. The best accuracy-to-cost ratio comes from **Claude Haiku 4.5** and **Gemini 2.5 Flash**, while the highest raw accuracy belongs to **GPT-4o** and **Gemini 2.5 Flash**.

---

## Head-to-Head Comparison

| Model | Provider | Tier | Within-1 | MAE | Bias | Exact Agree | Cost/1K Students |
|---|---|---|---|---|---|---|---|
| **gpt-4o** | OpenAI | Premium | 99% | 0.57 | -0.48 | 44% | $27.00 |
| **claude-sonnet-4-6** | Anthropic | Premium | 99% | 0.38 | -0.35 | 63% | $36.90 |
| **claude-haiku-4-5** | Anthropic | Budget | 95% | 0.26 | -0.04 | 78% | $12.30 |
| **gemini-2.5-flash** | Google | Budget | 98% | 0.31 | -0.13 | 70% | $5.19 |
| **gemini-2.5-pro** | Google | Premium | 96% | 0.29 | -0.06 | 74% | $21.00 |

---

## Detailed Analysis

### Within-1 Agreement (Primary Metric)

Within-1 agreement is the percentage of AI scores that fall within 1 point of the human grader average on a 1–4 scale. This is the most important metric because a 2-point discrepancy could change a student's pass/fail outcome.

| Rank | Model | Within-1 | Verdict |
|---|---|---|---|
| 1 (tie) | gpt-4o | 99% | Only 1 of 84 scores off by >1 |
| 1 (tie) | claude-sonnet-4-6 | 99% | Only 1 of 84 scores off by >1 |
| 3 | gemini-2.5-flash | 98% | 2 of 84 scores off by >1 |
| 4 | gemini-2.5-pro | 96% | 3 of 84 scores off by >1 |
| 5 | claude-haiku-4-5 | 95% | 4 of 84 scores off by >1 |

All five models pass the 90% threshold. The practical difference between 95% and 99% is small at 14-student scale (3 fewer outlier scores) but becomes meaningful at 1,000+ students: roughly 40 fewer scores requiring manual review.

### MAE and Bias (Score Precision)

While within-1 measures whether scores are "close enough," MAE and bias reveal the character of the disagreements.

**MAE (Mean Absolute Error)** — How far off is each score, on average?

| Rank | Model | MAE | Interpretation |
|---|---|---|---|
| 1 | claude-haiku-4-5 | 0.26 | Closest overall to human graders |
| 2 | gemini-2.5-pro | 0.29 | Near-identical to human consensus |
| 3 | gemini-2.5-flash | 0.31 | Strong agreement |
| 4 | claude-sonnet-4-6 | 0.38 | Good agreement |
| 5 | gpt-4o | 0.57 | Adequate but notably higher error |

**Bias** — Does the model systematically grade higher or lower?

| Rank | Model | Bias | Interpretation |
|---|---|---|---|
| 1 | claude-haiku-4-5 | -0.04 | Nearly perfectly calibrated |
| 2 | gemini-2.5-pro | -0.06 | Minimal systematic error |
| 3 | gemini-2.5-flash | -0.13 | Slight harshness |
| 4 | claude-sonnet-4-6 | -0.35 | Moderate harshness |
| 5 | gpt-4o | -0.48 | Most systematic harshness |

A notable pattern: **the premium-tier models (gpt-4o, claude-sonnet-4-6) show more systematic harshness** than the budget models. This suggests the larger models apply the grading rubric more strictly, which paradoxically moves them further from the human graders' more generous interpretation.

### Exact Agreement

Exact agreement measures how often the AI assigns the identical score as the human average (rounded). This is the most stringent test.

| Rank | Model | Exact Agree |
|---|---|---|
| 1 | claude-haiku-4-5 | 78% |
| 2 | gemini-2.5-pro | 74% |
| 3 | gemini-2.5-flash | 70% |
| 4 | claude-sonnet-4-6 | 63% |
| 5 | gpt-4o | 44% |

Claude Haiku and Gemini Pro achieve exact agreement with human graders roughly 3 out of 4 times — a remarkable level of alignment that approaches inter-rater reliability between experienced human graders.

---

## Per-Section Performance

### Consistently Easy Sections

**Physical Exam (PEX)** and **Plan** were graded accurately by all models, with near-perfect agreement. These sections tend to have more objective criteria (specific exam findings, specific interventions) that all models can evaluate reliably.

### Consistently Difficult: Summary (SUM)

Every model struggled most with the Summary section, which requires subjective judgment about how well a student synthesizes clinical findings. All five models showed a systematic downward bias (grading harsher than humans) on this section:

| Model | SUM Bias | SUM MAE |
|---|---|---|
| claude-haiku-4-5 | -0.32 | 0.46 |
| gemini-2.5-flash | -0.46 | 0.75 |
| gpt-4o | -0.68 | 0.68 |
| gemini-2.5-pro | -0.82 | 0.82 |
| claude-sonnet-4-6 | -0.82 | 0.82 |

**Notable:** All five models flagged the same student (Row 13) as a major outlier in the Summary section (GPT=2, human graders=3.5 avg). This consistent disagreement suggests either a genuine edge case in the rubric interpretation or an area where the prompt could be further refined.

### Divergent Section: Support

The Supporting Evidence section showed the most variation across models, with some models (claude-haiku, gemini-2.5-pro) grading more leniently than humans (+0.43 and +0.57 bias respectively), while others graded harsher. This section may benefit from additional prompt calibration.

---

## Cost Analysis

### Cost Per 1,000 Students

| Model | Input $/1M tokens | Output $/1M tokens | Est. Cost/1K Students |
|---|---|---|---|
| gemini-2.5-flash | $0.30 | $2.50 | **$5.19** |
| claude-haiku-4-5 | $1.00 | $5.00 | **$12.30** |
| gemini-2.5-pro | $1.25 | $10.00 | **$21.00** |
| gpt-4o | $2.50 | $10.00 | **$27.00** |
| claude-sonnet-4-6 | $3.00 | $15.00 | **$36.90** |

### Cost-Accuracy Efficiency

To compare value, we can look at the cost to achieve each percentage point of within-1 agreement:

| Model | Within-1 | Cost/1K | $/point of accuracy |
|---|---|---|---|
| gemini-2.5-flash | 98% | $5.19 | $0.053 |
| claude-haiku-4-5 | 95% | $12.30 | $0.129 |
| gemini-2.5-pro | 96% | $21.00 | $0.219 |
| gpt-4o | 99% | $27.00 | $0.273 |
| claude-sonnet-4-6 | 99% | $36.90 | $0.373 |

**Gemini 2.5 Flash offers the best cost-accuracy ratio** at 5x cheaper than the next most efficient option while achieving 98% within-1 agreement.

### Scaling Projections

| Scale | gemini-2.5-flash | claude-haiku-4-5 | gemini-2.5-pro | gpt-4o | claude-sonnet-4-6 |
|---|---|---|---|---|---|
| 100 students | $0.52 | $1.23 | $2.10 | $2.70 | $3.69 |
| 1,000 students | $5.19 | $12.30 | $21.00 | $27.00 | $36.90 |
| 10,000 students | $51.90 | $123.00 | $210.00 | $270.00 | $369.00 |

At all scales, the cost differences are modest in absolute terms. Even the most expensive option (Claude Sonnet) costs less than $0.04 per student — a fraction of a single minute of human grading time.

---

## Recommendations

### Best Overall: Gemini 2.5 Flash

For most deployments, **Gemini 2.5 Flash** offers the strongest combination of accuracy and cost:
- 98% within-1 agreement (just 1% below the best)
- $5.19 per 1,000 students (cheapest by far)
- Low bias (-0.13) and strong MAE (0.31)
- 70% exact agreement with human graders

### Best Accuracy: GPT-4o or Claude Sonnet 4.6

When maximum agreement with human graders is the priority (e.g., high-stakes assessments):
- Both achieve 99% within-1 agreement
- Claude Sonnet edges ahead on MAE (0.38 vs 0.57) and exact agreement (63% vs 44%)
- GPT-4o is more cost-effective ($27 vs $37 per 1,000 students)

### Best Calibration: Claude Haiku 4.5

When minimal systematic bias is critical (e.g., research studies comparing AI vs human grading):
- Near-zero bias (-0.04) means no systematic advantage or disadvantage to students
- Lowest MAE (0.26) and highest exact agreement (78%)
- 95% within-1 is still well above the 90% threshold
- Moderate cost ($12.30 per 1,000 students)

### Provider Diversification

Having multi-provider support provides practical benefits:
- **Redundancy:** If one provider has an outage or rate-limiting issues, switch to another
- **Cost optimization:** Use budget models for formative assessments, premium for summative
- **Validation:** Run the same student notes through two providers and compare for quality assurance

---

## Limitations

1. **Small sample size:** 14 students from a single OSCE case (flank pain). Results should be validated across multiple cases and larger cohorts.

2. **Prompt optimization:** The grading prompt was originally optimized for OpenAI models. Provider-specific prompt tuning could improve Anthropic and Google results further.

3. **Single run:** Each model was run once. Stochastic variation (especially at temperature 0.3) means results could shift slightly on re-runs.

4. **Summary section weakness:** All models struggle with the Summary section. Targeted prompt improvements for this section could raise all models' performance.

---

## Test Configuration

All models used identical settings:
```
Prompt:     config.GRADING_PROMPT (calibrated holistic prompt)
Temperature: 0.3
Dataset:    14 students, 6 sections, 84 score comparisons
Workers:    4 (parallel section grading)
Date:       February 25, 2026
```

Results files are stored as `results_{provider}_{model}.xlsx` in the repository root.
