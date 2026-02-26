# OSCE Grader Configuration File

# ---------------------------------------------------------------------------
# LLM Provider & Model
# ---------------------------------------------------------------------------

# Provider: "openai", "anthropic", or "google"
PROVIDER = "openai"

# Model to use (must match the selected provider).
# If switching providers via --provider without specifying --model,
# the default model for that provider is used (see DEFAULT_MODELS below).
MODEL = "gpt-4o"

# Default model per provider (used when --provider is set without --model)
DEFAULT_MODELS = {
    "openai":    "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
    "google":    "gemini-2.5-flash",
}

# Maximum output tokens (required by Anthropic, useful for Google)
MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
# Each provider checks its environment variable first, then a key file in
# the scripts/ directory.  The OpenAI provider also falls back to the
# legacy API_KEY_FILE below.
#
#   OpenAI:    OPENAI_API_KEY    or  openai_api_key.txt
#   Anthropic: ANTHROPIC_API_KEY or  anthropic_api_key.txt
#   Google:    GOOGLE_API_KEY    or  google_api_key.txt
#
API_KEY_FILE = "api_key.txt"  # legacy fallback for OpenAI only

# Default Grading Prompt
GRADING_PROMPT = """I am a medical educator, and I need your help grading a student's OSCE post-encounter note.

IMPORTANT GRADING GUIDELINES:
- The answer key is a GUIDE showing key concepts, NOT a strict checklist. Students do NOT need to use the exact same wording. Paraphrasing, synonyms, and different phrasing that conveys the same clinical concept should receive full credit.
- Accept CLINICAL SYNONYMS generously: e.g., "abdominal pain" and "flank pain" for the same location, "intense" and "severe" for pain severity, "throwing up" and "emesis" for vomiting. If the student's language clearly refers to the same clinical finding, give credit.
- Grade HOLISTICALLY. Consider whether the student demonstrates understanding of the key clinical concepts overall, rather than counting individual missing bullet points.
- Do NOT penalize students for including additional relevant clinical information beyond the answer key. Extra relevant details show thoroughness, not disorganization.
- Accept REASONABLE ALTERNATIVE clinical approaches. For example, if the answer key lists one imaging modality but the student suggests a clinically appropriate alternative, give credit.
- When the rubric describes score levels (e.g., 4=almost all key info, 3=most key findings, 2=many omitted), apply them as a human clinical educator would — generously for students who capture the essential clinical picture.
- Reserve a score of 2 for responses that are GENUINELY incomplete — where the student is missing MULTIPLE major required concepts, not just minor details or descriptors. A score of 2 means the response is substantially deficient.
- If a student covers the major required concepts but misses some secondary details, that is typically a 3, not a 2.
- BEFORE assigning a score below 3, ask yourself: "Would a reasonable, experienced clinical educator consider this response adequate?" If the answer is yes or borderline, assign a 3.
- When in doubt between two scores, choose the HIGHER score.

Score each section separately. Provide a brief explanation of your reasoning, then place the final score as an integer on a new line with no other text or markup."""

# Default Temperature & Top-P (configurable via CLI flags --temperature and --top_p)
TEMPERATURE = 0.3
TOP_P = 1.0

# Sections to grade (column names expected in the student notes Excel file).
# These must match the column headers in your student-notes Excel file.
#
# NOTE: Some rubrics include an "org" (organization) section that is scored
# by human graders only and does not appear as a content column in the student
# notes.  If your workflow does include an "org" column in the student notes
# that should be AI-graded, add 'org' to this list.
SECTIONS = ['hpi', 'pex', 'sum', 'ddx', 'support', 'plan']

# Concurrency: max number of sections to grade in parallel per student.
# Set to 1 for sequential processing. Higher values speed up grading but
# increase API request rate. A good default is the number of sections.
MAX_WORKERS = 4

# API retry settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds (doubles on each retry)

# File Paths (Users can set defaults here)
DEFAULT_RUBRIC_PATH = "examples/sample_standard_rubric.xlsx"
DEFAULT_ANSWER_KEY_PATH = "examples/sample_flankpain_key.xlsx"
DEFAULT_NOTES_PATH = "examples/sample_student_notes.xlsx"
DEFAULT_OUTPUT_PATH = "results.xlsx"

# ---------------------------------------------------------------------------
# Cost per million tokens (input, output) for supported models.
# Used by evaluate.py for cost estimation.
# ---------------------------------------------------------------------------
MODEL_COSTS = {
    # OpenAI
    "gpt-4o":              (2.50, 10.00),
    "gpt-4o-mini":         (0.15,  0.60),
    # Anthropic
    "claude-sonnet-4-6":   (3.00, 15.00),
    "claude-haiku-4-5":    (1.00,  5.00),
    # Google
    "gemini-2.5-flash":    (0.30,  2.50),
    "gemini-2.5-pro":      (1.25, 10.00),
}
