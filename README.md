# OSCE Grader
An AI-powered grading system for medical student OSCE post-encounter notes, using LLMs to automate grading and provide structured feedback.

## Features
- **Multi-provider LLM support:** OpenAI (GPT), Anthropic (Claude), and Google (Gemini)
- Supports **Excel-based** student notes
- Works with **structured rubrics** (Excel)
- Uses LLMs to generate **detailed, section-by-section grading**
- Customizable **grading prompts** via `config.py`
- **Automatic logging** of all LLM interactions for debugging and analysis
- **Parallel section grading** for faster processing (configurable via `--workers`)
- **Automatic retry** with exponential back-off on API failures
- **Evaluation script** to measure grading accuracy against human graders
- Includes **convert_rubric.py** to assist in extracting text from rubric files (PDF/DOCX)

---

## Getting Started
### **1. Install Dependencies**
You'll need **Python 3.8+** and an API key for your chosen provider.

```sh
git clone https://github.com/christopherjnash/OSCE-Grader.git
cd OSCE-Grader
pip install -r requirements.txt
```

---

### **2. Configure the Grader**
The **grading prompt, provider/model selection, API key location, and default file paths** are managed in `scripts/config.py`.
Modify `config.py` as needed to customize the grading behavior for your institution.

Example of `config.py` settings:
```python
PROVIDER = "openai"              # "openai", "anthropic", or "google"
MODEL = "gpt-4o"                 # Must match your selected provider
TEMPERATURE = 0.3
SECTIONS = ['hpi', 'pex', 'sum', 'ddx', 'support', 'plan']
```

---

### **3. Set Up Your API Key**

Each provider uses its own API key. You can provide the key via environment variable (recommended) or a key file in the `scripts/` directory.

| Provider | Environment Variable | Key File |
|----------|---------------------|----------|
| OpenAI | `OPENAI_API_KEY` | `scripts/openai_api_key.txt` |
| Anthropic | `ANTHROPIC_API_KEY` | `scripts/anthropic_api_key.txt` |
| Google | `GOOGLE_API_KEY` | `scripts/google_api_key.txt` |

Environment variables take precedence over key files. For OpenAI, there is also a legacy fallback to `scripts/api_key.txt`.

**Example:**
```sh
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
# or
export GOOGLE_API_KEY="your-key-here"
```

---

### **4. Run the OSCE Grader**
#### **Basic Usage (default provider: OpenAI)**
```sh
python scripts/grader.py \
  --rubric examples/sample_standard_rubric.xlsx \
  --answer_key examples/sample_flankpain_key.xlsx \
  --notes examples/sample_student_notes.xlsx \
  --output results.xlsx
```

#### **Using a Different Provider**
```sh
# Anthropic Claude
python scripts/grader.py --provider anthropic --model claude-sonnet-4-6 \
  --output results.xlsx

# Google Gemini
python scripts/grader.py --provider google --model gemini-2.5-flash \
  --output results.xlsx
```

If you specify `--provider` without `--model`, the default model for that provider is used automatically:

| Provider | Default Model |
|----------|--------------|
| openai | gpt-4o |
| anthropic | claude-sonnet-4-6 |
| google | gemini-2.5-flash |

#### **Script Parameters**
| Flag | Description |
|------|------------|
| `--provider` | LLM provider: `openai`, `anthropic`, or `google` (default: `openai`) |
| `--model` | Model name (auto-detected from provider if omitted) |
| `--rubric` | Path to the grading rubric (Excel) |
| `--answer_key` | Path to the answer key (Excel) |
| `--notes` | Path to student notes (Excel) |
| `--output` | Name of the output file (Excel) |
| `--temperature` | Temperature setting, 0.0-2.0 (default: 0.3) |
| `--top_p` | Top-p (nucleus sampling) setting, 0.0-1.0 (default: 1.0) |
| `--workers` | Number of sections to grade in parallel per student (default: 4) |

**Provider Notes:**
- **Anthropic:** Temperature is clamped to 0.0-1.0 (Claude's supported range). The `top_p` parameter is not used, as Anthropic does not allow both temperature and top_p simultaneously.
- **Google:** Both temperature and top_p are passed through. The `max_output_tokens` is set via `config.MAX_TOKENS`.
- **OpenAI:** All parameters are passed through as-is.

---

## Evaluating Results

The `evaluate.py` script compares AI scores against human grader scores to measure accuracy:

```sh
python scripts/evaluate.py results.xlsx --model gpt-4o
```

This produces a report showing:
- **Within-1 agreement** — percentage of scores within 1 point of human average (target: ≥90%)
- **MAE** — mean absolute error vs. human graders
- **Bias** — whether the model grades systematically higher or lower
- **Exact agreement** — percentage of identical scores
- **Cost estimate** — projected API costs based on model pricing

---

## Supported Models

| Provider | Model | Tier | Notes |
|----------|-------|------|-------|
| OpenAI | gpt-4o | Premium | Best within-1 accuracy (99%) |
| OpenAI | gpt-4o-mini | Budget | Lower cost, slightly less accurate |
| Anthropic | claude-sonnet-4-6 | Premium | Excellent MAE and exact agreement |
| Anthropic | claude-haiku-4-5 | Budget | Best calibration (near-zero bias) |
| Google | gemini-2.5-flash | Budget | Best cost-accuracy ratio |
| Google | gemini-2.5-pro | Premium | Strong accuracy, moderate cost |

See `docs/provider_comparison_report.md` for detailed accuracy and cost analysis across all models.

---

## Batch Processing
A Windows batch script template is included at `scripts/batch.bat` for grading multiple OSCE case types in sequence. See the comments inside the file for how to configure the file-name patterns for your own data.

---

## Converting a Rubric File
If your rubric is in **PDF or DOCX**, you can use `convert_rubric.py` to extract its text into an Excel or CSV file.

**Example Usage:**
```sh
python scripts/convert_rubric.py examples/FlankPainRubric.pdf examples/converted_rubric.xlsx
```

**Important:** `convert_rubric.py` extracts raw text into a single column. The output will require **manual restructuring** into the section-column format expected by `grader.py` (i.e. columns named `hpi`, `pex`, `sum`, `ddx`, `support`, `plan`). See the included `examples/sample_standard_rubric.xlsx` for the expected format.

---

## Customizing the Grading Prompt
- The script **grades section-by-section** for higher accuracy.
- You can modify the **grading prompt** in `config.py` without editing the script directly.

Example:
```python
GRADING_PROMPT = "I am a medical educator, and I need your help grading an assignment... (your modified prompt)"
```

See `docs/modifying_prompt.md` for detailed guidance.

---

## Documentation
- `docs/calibration_report.md` — Prompt engineering and model calibration methodology
- `docs/provider_comparison_report.md` — Multi-provider accuracy and cost comparison
- `docs/modifying_prompt.md` — Guide to customizing the grading prompt
- `docs/troubleshooting.md` — Common issues and fixes

---

## Troubleshooting
See `docs/troubleshooting.md` for common issues and fixes, including:
- API key not found
- Missing column errors
- Unexpected scores
- API rate limits

---

## Resources
- **GitHub Repository:** [OSCE-Grader](https://github.com/christopherjnash/OSCE-Grader)
- **OpenAI API Docs:** [OpenAI](https://platform.openai.com/docs/)
- **Anthropic API Docs:** [Anthropic](https://docs.anthropic.com/)
- **Google Gemini API Docs:** [Google AI](https://ai.google.dev/docs)
- **Troubleshooting Guide:** See `docs/troubleshooting.md`

---

## License
**MIT License** - Free to use and modify.
See [`LICENSE`](LICENSE) for details.

---

## Contributions
Contributions are welcome! If you improve the script or add new features, submit a **pull request** or open an **issue**.
