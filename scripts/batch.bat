@echo off
REM =========================================================================
REM Batch grading script (Windows)
REM
REM Runs grader.py multiple times across different OSCE case types.  Each case
REM is graded twice (%%x loop) to allow consistency comparison between runs.
REM
REM HOW TO USE:
REM   1. Place your data files in the same directory as this script, OR
REM      update the paths below to point to the correct locations.
REM   2. Run this script FROM the scripts/ directory:
REM        cd scripts
REM        batch.bat
REM
REM BEFORE USING: Update the file-name patterns below to match your own
REM rubric, answer-key, and student-notes files.  The naming convention
REM assumed here is:
REM   - Rubric:     <case>_rubric.xlsx     (or a shared rubric for all cases)
REM   - Answer key: <case>_key.xlsx
REM   - Notes:      <case>_notes.xlsx
REM   - Output:     output_<case>_<run>.xlsx
REM
REM Example with the included sample files (single case, single run):
REM   python grader.py --rubric ..\examples\sample_standard_rubric.xlsx ^
REM       --answer_key ..\examples\sample_flankpain_key.xlsx ^
REM       --notes ..\examples\sample_student_notes.xlsx ^
REM       --output ..\results.xlsx
REM =========================================================================

REM Ensure we are running from the directory that contains this script
cd /d "%~dp0"

setlocal enabledelayedexpansion
set titles=flankpain hematemesis insomnia kneepain nasalcongestion syncope troubleconcentrating uterinebleeding

for %%t in (%titles%) do (
    for /l %%x in (1, 1, 2) do (
        python grader.py --rubric "%%t_rubric.xlsx" --answer_key "%%t_key.xlsx" --notes "%%t_notes.xlsx" --output "output_%%t_%%x.xlsx" --temperature 0.1 --top_p 1.0
    )
)
