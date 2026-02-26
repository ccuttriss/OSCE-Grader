# OSCE Grader Configuration File

# OpenAI API Key
# Loaded from the API_KEY_FILE path below, or from the OPENAI_API_KEY environment variable.
# The environment variable takes precedence if set.
API_KEY_FILE = "api_key.txt"

# Model Selection
MODEL = "gpt-4o-mini"

# Default Grading Prompt
GRADING_PROMPT = """I am a medical educator, and I need your help grading an assignment.
My students recently completed an OSCE post-encounter note based on a standardized patient interview.
I have provided a structured scoring rubric with expected responses.
The rubric is broken into individual sections. To ensure accuracy, please score each section separately.
For each section, provide a detailed explanation of your reasoning before giving a final score.

You MUST respond with valid JSON in this exact format:
{"explanation": "Your detailed reasoning here...", "score": 8}

The "score" field must be an integer. The "explanation" field should contain your full evaluation."""

# Default Temperature & Top-P (configurable via CLI flags --temperature and --top_p)
TEMPERATURE = 0.1
TOP_P = 1.0

# Sections to grade (column names expected in the student notes Excel file).
# These must match the column headers in your student-notes Excel file.
#
# NOTE: Some rubrics include an "org" (organization) section that is scored
# by human graders only and does not appear as a content column in the student
# notes.  If your workflow does include an "org" column in the student notes
# that should be AI-graded, add 'org' to this list.
SECTIONS = ['hpi', 'pex', 'sum', 'ddx', 'support', 'plan']

# API retry settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds (doubles on each retry)

# File Paths (Users can set defaults here)
DEFAULT_RUBRIC_PATH = "examples/sample_standard_rubric.xlsx"
DEFAULT_ANSWER_KEY_PATH = "examples/sample_flankpain_key.xlsx"
DEFAULT_NOTES_PATH = "examples/sample_student_notes.xlsx"
DEFAULT_OUTPUT_PATH = "results.xlsx"
