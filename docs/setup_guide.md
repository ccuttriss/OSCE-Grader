# OSCE Grader Setup Guide

Welcome to the **OSCE Grader**! This guide will help you set up the grading system for medical student OSCE post-encounter notes using **ChatGPT**.

## Prerequisites
### Install Python (If Not Installed)
- **Windows**: Download and install from [python.org](https://www.python.org/downloads/)
- **Mac/Linux**: Python is usually pre-installed. Check by running:
  ```sh
  python3 --version
  ```

### Install Git (Optional but Recommended)
- Download from [git-scm.com](https://git-scm.com/)
- Check installation:
  ```sh
  git --version
  ```

### Obtain an OpenAI API Key
1. Sign up at [OpenAI](https://platform.openai.com/signup/)
2. Navigate to **API Keys** and generate a new key
3. Copy and **save it securely**
4. Provide the key using one of these methods:
   - **Environment variable (recommended):** `export OPENAI_API_KEY="your-key-here"`
   - **Key file:** Create a file named `api_key.txt` in the `scripts/` folder and paste your key inside

## Installation Steps
### 1. Clone the Repository
```sh
git clone https://github.com/christopherjnash/OSCE-Grader.git
cd OSCE-Grader
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Run the Grader
```sh
python scripts/grader.py --rubric examples/sample_standard_rubric.xlsx --answer_key examples/sample_flankpain_key.xlsx --notes examples/sample_student_notes.xlsx --output results.xlsx
```

### Optional Parameters
| Flag | Description |
|------|------------|
| `--temperature` | Temperature setting for the model, 0.0-2.0 (default: 0.5) |
| `--top_p` | Top-p (nucleus sampling) setting, 0.0-1.0 (default: 1.0) |

## Next Steps
- Test the setup with sample files in `examples/`
- Customize the **grading prompt** using [Modifying Prompt Guide](modifying_prompt.md)
- Troubleshoot any issues using [Troubleshooting Guide](troubleshooting.md)
