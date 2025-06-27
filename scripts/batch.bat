@echo off
setlocal enabledelayedexpansion
set titles=flankpain hematemesis insomnia kneepain nasalcongestion syncope troubleconcentrating uterinebleeding

for %%t in (%titles%) do (
    for /l %%x in (1, 1, 2) do (
        python grader.py standardrubric.xlsx "%%tkey.xlsx" "%%tnotes.xlsx" "output_%%t_%%x.xlsx" --temperature 0.1 --top_p 1.0
    )
)
