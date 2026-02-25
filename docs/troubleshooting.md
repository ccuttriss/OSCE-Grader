# OSCE Grader Troubleshooting Guide

## Common Issues & Fixes

### API Key Not Found
**Issue:**
- The script fails to run with an API key error.

**Fix:**
1. Ensure you have an **OpenAI API Key**.
2. Provide the key using one of these methods:
   - **Environment variable (recommended):**
     ```sh
     export OPENAI_API_KEY="your-key-here"
     ```
   - **Key file:** Save your API key inside a file named `api_key.txt` in the `scripts/` folder. The file must not be empty or contain only whitespace.

### Dependencies Not Installed
**Issue:**
- The script throws an error about missing Python packages (e.g. `ModuleNotFoundError` for `openai` or `openpyxl`).

**Fix:**
- Install all required dependencies by running:
  ```sh
  pip install -r requirements.txt
  ```

### Input File Not Found
**Issue:**
- The script reports that an input file does not exist.

**Fix:**
1. Double-check the paths passed to `--rubric`, `--answer_key`, and `--notes`.
2. Paths are relative to your current working directory, not the `scripts/` folder.
3. Use the sample files in `examples/` to verify your setup works before switching to your own data.

### File Format Issues
**Issue:**
- The script crashes when loading student notes or rubric files.

**Fix:**
1. Ensure files are in **Excel (`.xlsx`)** format.
2. Convert Word/PDF rubrics using `convert_rubric.py` (note: the output will need manual restructuring into the expected column format).
3. Check that column headers in Excel match the expected section fields:
   `hpi`, `pex`, `sum`, `ddx`, `support`, `plan`.

### Missing Columns Error
**Issue:**
- The script reports missing expected columns.

**Fix:**
1. Open your student notes Excel file and verify it contains columns for each
   grading section: `hpi`, `pex`, `sum`, `ddx`, `support`, `plan`.
2. Column names are **case-sensitive** and must be lowercase.
3. The list of expected columns can be customized in `config.py` via the
   `SECTIONS` setting.

### Invalid Temperature or Top-P
**Issue:**
- The script reports an invalid `--temperature` or `--top_p` value.

**Fix:**
1. `--temperature` must be between **0.0 and 2.0** (default: 0.5).
2. `--top_p` must be between **0.0 and 1.0** (default: 1.0).
3. If using **o1 or o3 models**, temperature must be set to `1.0` and `top_p` is not supported by those models.

### Unexpected Scores or Formatting Issues
**Issue:**
- The grading output seems incorrect or inconsistent.

**Fix:**
1. Ensure that your **grading rubric is formatted correctly**.
2. If necessary, **modify the grading prompt** to match institutional needs (see [Modifying Prompt Guide](modifying_prompt.md)).
3. Review the log file (same name as the output file but with a `.log` extension) to inspect the full LLM interaction for each section.
4. The grader extracts the numeric score using regex. Ensure your prompt instructs the model to place the final score as a standalone integer on its own line.

### API Rate Limits Exceeded
**Issue:**
- The script fails due to OpenAI API rate limits.

**Fix:**
1. The grader includes automatic retry logic with exponential back-off. In most
   cases it will recover on its own.
2. If rate limit errors persist, reduce the batch size (fewer students per run).
3. Upgrade to a **higher-tier OpenAI plan** if applicable.
4. You can adjust the retry behaviour in `config.py` via `MAX_RETRIES` and
   `RETRY_DELAY`.

### Large Log Files
**Issue:**
- The `.log` file grows very large after multiple runs.

**Fix:**
1. The log file uses append mode, so repeated runs against the same output file
   will accumulate entries. Delete or rename the log file between runs if size
   is a concern.
2. Each grading run produces one log entry per section per student. For a class
   of 100 students across 6 sections, expect ~600 log entries per run.

### Other Issues
- Check the [GitHub Issues](https://github.com/christopherjnash/OSCE-Grader/issues) for ongoing bug reports.
- If you encounter a unique issue, **submit a GitHub issue** with error details.
