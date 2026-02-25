# Modifying the Grading Prompt

## Why Modify the Prompt?
The grading prompt determines how the LLM evaluates and scores student responses. Customizing it ensures that the model aligns with **your institution's rubric and grading style.**

---
## Default Grading Prompt
The default prompt is defined in `scripts/config.py` as `GRADING_PROMPT` and follows this structure:

```plaintext
SYSTEM MESSAGE:
"I am a medical educator, and I need your help grading an assignment.
My students recently completed an OSCE post-encounter note based on a standardized patient interview.
I have provided a structured scoring rubric with expected responses.
The rubric is broken into individual sections. To ensure accuracy, please score each section separately.
For each section, provide a detailed explanation of your reasoning before giving a final score.
At the end of your evaluation, place the final score as an integer on a new line with no markup."
```

The final line is important for automated score extraction. The grader uses regex to find the score, so the prompt should always ask the model to place the score as a standalone integer.

---
## How to Modify the Prompt
### **Option 1: Modify `config.py` (Recommended)**
You can edit `scripts/config.py` to update the grading prompt without changing the script logic.

Example:
```python
GRADING_PROMPT = "I am a medical educator... (your modified prompt)"
```

### **Option 2: Edit `grader.py` Directly**
Locate the section in `grader.py` that defines the **system message** and update it:

```python
messages = [
    {"role": "system", "content": "Your modified prompt text here."}
]
```

---
## Customizing Sections
The sections that are graded are defined in `scripts/config.py` as the `SECTIONS` list:

```python
SECTIONS = ['hpi', 'pex', 'sum', 'ddx', 'support', 'plan']
```

These must match the column names in your student notes Excel file. If your workflow includes additional sections (e.g. `org` for organization), add them to this list.

---
## Best Practices for Prompt Customization
- **Be Clear & Specific** - Define exactly what you want the model to evaluate.
- **Use Examples** - Provide sample student responses and expected answers in the rubric.
- **Avoid Overloading** - Keep prompts concise to prevent AI confusion.
- **Keep the score instruction** - Always ask the model to place the final score as an integer on its own line so the grader can extract it reliably.
- **Test & Iterate** - Run a few test cases to refine performance.

---
## Testing Your Changes
After modifying the prompt, test its effectiveness:
```sh
python scripts/grader.py --rubric examples/sample_standard_rubric.xlsx --answer_key examples/sample_flankpain_key.xlsx --notes examples/sample_student_notes.xlsx --output results.xlsx
```

Check the log file (`results.log`) to review the full LLM interaction for each section and verify that scores are being extracted correctly.

If the results aren't what you expect, tweak the prompt further and **re-run the script.**

A well-tuned prompt ensures accurate and fair grading!
