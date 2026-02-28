# PR Worthiness Analysis: ccuttriss/OSCE-Grader vs christopherjnash/OSCE-Grader

*Analysis performed 2026-02-28*

## Executive Summary

The fork adds **+4,026 / -262 lines** across **37 files** in **11 commits**, transforming a ~150-line single-provider CLI tool into a production-grade, multi-provider system with web UIs, evaluation tooling, and testing. **The changes are PR-worthy in substance but should be split into 3-4 focused PRs for reviewability.**

---

## Original Repository Baseline

| Attribute | Value |
|-----------|-------|
| Repository | `christopherjnash/OSCE-Grader` |
| Stars / Forks | 4 / 3 |
| Last commit | June 27, 2025 |
| Open issues/PRs | 0 / 0 |
| Core code | ~150 lines Python (grader + config) |
| Provider | OpenAI only (gpt-4o-mini) |
| Score extraction | Second LLM API call (doubles cost) |
| Interface | CLI only |
| Tests | None |
| Evaluation tooling | None |

## Fork Changes Summary

### 1. Core Architecture: Multi-Provider LLM Support
- New `scripts/providers.py` (243 lines) -- provider abstraction for OpenAI, Anthropic, Google
- Factory pattern with lazy imports
- Per-provider API key resolution (env var -> file -> legacy fallback)

### 2. Grader Script Overhaul (90 -> 518 lines)
- Eliminated wasteful second API call; regex-based score extraction
- Retry with exponential backoff
- Parallel section grading (ThreadPoolExecutor)
- Thread-safe logging, input validation, intermediate saves
- New CLI flags: `--provider`, `--model`, `--workers`, `--temperature`, `--top_p`

### 3. Prompt Calibration
- 4 iterations: 83% -> 99% within-1-point agreement with human graders
- Documented in `docs/calibration_report.md`

### 4. Evaluation Framework (218 lines)
- AI vs. human grader comparison (MAE, bias, exact/within-1 agreement)
- Per-section analysis, cost estimation, outlier detection

### 5. Two Web UIs
- Streamlit (932 lines): 4-tab app with grading, analysis dashboard, flagging, rubric converter
- Flask (~700 lines): server-based alternative with HTMX, background workers

### 6. Test Suite (169 lines)
- 13 score extraction tests, column validation tests, resume support tests

### 7. Documentation
- New: calibration report, provider comparison report
- Updated: all existing docs (README, setup, troubleshooting, modifying prompt)

### 8. Provider Benchmarks

| Model | Provider | Within-1 | Cost/1K | Bias |
|-------|----------|----------|---------|------|
| gpt-4o | OpenAI | 99% | $27.00 | -0.48 |
| claude-sonnet-4-6 | Anthropic | 99% | $36.90 | -0.35 |
| claude-haiku-4-5 | Anthropic | 95% | $12.30 | -0.04 |
| gemini-2.5-flash | Google | 98% | $5.19 | -0.13 |
| gemini-2.5-pro | Google | 96% | $21.00 | -0.06 |

---

## Assessment

### Strengths
1. **10x value increase** -- transforms a basic script into a production system
2. **Eliminates double API cost** -- regex score extraction vs. second LLM call
3. **Empirically validated** -- 99% within-1 agreement, documented methodology
4. **Clean architecture** -- factory pattern, provider abstraction, separation of concerns
5. **Security-conscious** -- XSS protection, input sanitization, upload limits

### Concerns
1. **Too large for one PR** -- +4,026 lines is unreviewable as a single submission
2. **Two web UIs is confusing** -- Streamlit AND Flask creates ambiguity
3. **AI co-authored** -- all commits co-authored with Claude Opus 4.6
4. **Noise commits** -- 3 auto-deploy test commits should be squashed
5. **Default model change** -- gpt-4o-mini -> gpt-4o (16x cost increase)

---

## Recommended PR Strategy

| PR | Contents | ~Lines | Priority |
|----|----------|--------|----------|
| PR 1 | Core fixes: regex score extraction, retry logic, validation, logging, CLI flags | ~800 | High |
| PR 2 | Multi-provider support: providers.py, config, comparison report | ~700 | High |
| PR 3 | Evaluation framework: evaluate.py, test suite | ~400 | Medium |
| PR 4 | Web UI (Streamlit only), examples, remaining docs | ~1,200 | Medium |

### Pre-submission Checklist
- [ ] Open an issue on upstream to gauge maintainer interest
- [ ] Squash auto-deploy test commits
- [ ] Decide on default model (keep gpt-4o-mini or make provider-aware default)
- [ ] Pick one web UI (recommend Streamlit)
- [ ] Ensure tests pass with clean `pip install -r requirements.txt && pytest`

---

## Verdict

**The changes are absolutely worthwhile.** The fork addresses real deficiencies (cost, single-provider lock-in, no evaluation tooling, no web UI, no tests) with high-quality, well-documented code. The recommendation is to submit as multiple focused PRs rather than one overwhelming contribution, and to open an issue first to establish communication with the upstream maintainer.
