# Copilot instructions for Toy Analysis

Summary
- This repo is a small analysis pipeline centered on `recruitment_attrition_analysis.py` that ingests a single CSV (default: `Task_to candidates_Final.csv`), cleans and transforms records to a candidate-level view, computes risk/attrition summaries, runs statistical tests, and (optionally) emits plots.

Big picture
- Single-script analysis: `recruitment_attrition_analysis.py` is the authoritative pipeline. It exposes `run_analysis(csv_path, min_n, show_plots)` which returns an `outputs` dict (see bottom of file).
- Data expectations: `EXPECTED_COLUMNS` (defined in the script) lists required input columns; do not change column semantics without updating rename mapping and downstream code.

Key files to inspect
- `recruitment_attrition_analysis.py` — ETL, analytics, plotting, tests and CLI behavior.
- `Task_to candidates_Final.csv` — example input used by default in `__main__`.
- `requirements.txt` — pinned packages used for reproducible runs.
- `recruitment_attrition_analysis.ipynb` — notebook version useful for interactive exploration.
- `convert_xls_to_csv.ps1` / `convert_xls_to_xlsx.ps1` — helper PowerShell utilities for preparing input data on Windows.

Developer workflows (explicit commands)
- Create venv and install (Windows PowerShell):
  - `.\\setup_venv.ps1`
  - `pip install -r requirements.txt`
- Run analysis script (no plots):
  - `python recruitment_attrition_analysis.py`
- Run interactively: open `recruitment_attrition_analysis.ipynb` in Jupyter or run `run_analysis(...)` from a REPL.

Project-specific conventions & patterns
- Column renaming: source columns → canonical snake_case via `RENAME_MAP`. Work with canonical names (`candidate_id`, `offer_date`, `final_result`, etc.).
- Datetime parsing: uses `parse_tw_datetime()` which expects Taiwanese locale patterns (上午/下午). Keep this when adding new date parsing logic.
- Aggregation thresholding: `min_n` (default 20) is central to which groups are reported; tests and reporting assume this threshold.
- Output contract: `run_analysis` returns a dict with keys like `df_raw`, `df_cand`, `quality`, `risk_segments`, `insight_df`, etc. Use these exact keys when downstream code consumes outputs.
- Plotting: `show_plots` flag controls `matplotlib` display — CI or script runs should pass `show_plots=False`.

Integration points & external deps
- Runtime packages (see `requirements.txt`): `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `openpyxl` (for Excel I/O in helpers).
- Input files are local CSV/Excel files; there are no external network services.

What an AI agent should do first
- Do not change `EXPECTED_COLUMNS` or `RENAME_MAP` without verifying sample CSV headers.
- When editing behavior, run `python recruitment_attrition_analysis.py` with `show_plots=False` and validate `checks_df` printed at the end.
- For unit tests, target pure functions: `parse_tw_datetime`, `clean_text`, `preprocess`, `build_candidate_view`, `run_stat_tests` — these are deterministic and fast.

Example prompts for code edits
- "Add a unit test that ensures `preprocess` rejects missing required columns listed in `EXPECTED_COLUMNS`."
- "Refactor `run_stat_tests` to make categorical column list configurable and add a small parametrized test for the Mann-Whitney result."

Notes / gotchas
- Chinese labels: `configure_chinese_font()` is applied before plotting to reduce font issues — keep it when touching plotting code.
- Negative `lead_days` are treated as errors (raises ValueError). Data fixes should handle source issues upstream or before preprocessing.

If anything in this summary is unclear or you want examples (tests, quick refactors, or CI hooks), tell me which section to expand.
