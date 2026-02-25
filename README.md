# OSCE Grader
An AI-powered grading system for medical student OSCE post-encounter notes, using GPT models to automate grading and provide structured feedback.

## Features
- Supports **Excel-based** student notes
- Works with **structured rubrics** (Excel)
- Uses **ChatGPT** to generate **detailed, section-by-section grading**
- Customizable **grading prompts** via `config.py`
- **Automatic logging** of all LLM interactions for debugging and analysis
- **Easy setup** with Python & OpenAI API
- **Automatic retry** with exponential back-off on API failures
- Includes **convert_rubric.py** to assist in extracting text from rubric files (PDF/DOCX)

---

## Getting Started
### **1. Install Dependencies**
You'll need **Python 3.8+** and an OpenAI API key.

```sh
git clone https://github.com/christopherjnash/OSCE-Grader.git
cd OSCE-Grader
pip install -r requirements.txt
```

---

### **2. Configure the Grader**
The **grading prompt, model selection, API key location, and default file paths** are managed in `scripts/config.py`.
Modify `config.py` as needed to customize the grading behavior for your institution.
Available models and pricing are available in the [OpenAI API Documentation](https://platform.openai.com/docs/pricing)

Example of `config.py` settings:
```python
MODEL = "gpt-4o-mini"
SECTIONS = ['hpi', 'pex', 'sum', 'ddx', 'support', 'plan']
DEFAULT_RUBRIC_PATH = "examples/sample_standard_rubric.xlsx"
DEFAULT_ANSWER_KEY_PATH = "examples/sample_flankpain_key.xlsx"
DEFAULT_NOTES_PATH = "examples/sample_student_notes.xlsx"
DEFAULT_OUTPUT_PATH = "results.xlsx"
```

---

### **3. Set Up Your OpenAI API Key**
You can provide the key in either of two ways (environment variable takes precedence):

1. **Environment variable (recommended):**
   ```sh
   export OPENAI_API_KEY="your-key-here"
   ```
2. **Key file:** Save your key in a file named `api_key.txt` in the `scripts/` folder.

---

### **4. Run the OSCE Grader**
#### **Basic Usage**
```sh
python scripts/grader.py --rubric examples/sample_standard_rubric.xlsx --answer_key examples/sample_flankpain_key.xlsx --notes examples/sample_student_notes.xlsx --output results.xlsx
```

#### **Script Parameters**
| Flag | Description |
|------|------------|
| `--rubric` | Path to the grading rubric (Excel) |
| `--answer_key` | Path to the answer key (Excel) |
| `--notes` | Path to student notes (Excel) |
| `--output` | Name of the output file (Excel) |
| `--temperature` | Temperature setting for the model, 0.0-2.0 (default: 0.5) |
| `--top_p` | Top-p (nucleus sampling) setting, 0.0-1.0 (default: 1.0) |

**Note:** Changing from the 4o models to the newer o1 or o3 models is possible but requires changing the prompt structure in your script. These models also require `temperature` to be set to `1.0` and do not support `top_p`. We recommend reading through the OpenAI documentation and the scripts to understand the changes before adjusting the code yourself.

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
- **Troubleshooting Guide:** See `docs/troubleshooting.md`

---

## License
**MIT License** - Free to use and modify.
See [`LICENSE`](LICENSE) for details.

---

## Contributions
Contributions are welcome! If you improve the script or add new features, submit a **pull request** or open an **issue**.
