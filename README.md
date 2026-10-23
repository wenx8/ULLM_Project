# Experiments and paper results

The main experiment uses all 708 FEVER-Symmetric examples (177 four-variant families). Each initially correct baseline receives independent weak, medium and strong follow-ups requesting the opposite label. The API models are Haiku 5.5 and Sonnet 5.5.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
```

For paid runs, set `ANTHROPIC_API_KEY` in your Windows user environment variables and reopen the terminal. No project file stores the key.

## Reproduce the saved results (offline)

```powershell
.\.venv\Scripts\python.exe rescore_results.py
.\.venv\Scripts\python.exe research_analysis.py
.\.venv\Scripts\python.exe paper_figures.py
```

Rescoring reuses saved extraction decisions; it makes no API calls without `--judge`. Analysis uses 10,000 whole-family bootstrap resamples, seed 2026. The two final plots are exported as PNG and PDF.

## Where to look

| Location | Contents |
|---|---|
| `results/review/corrected_*_overview.csv` | Easiest results to read: one row per example, baseline and three follow-up answers/outcomes side by side. |
| `results/review/response_audit.csv` | One row per response, including full text, original/audited labels, extraction method, reason and source. |
| `results/review/scoring_summary.csv` | Per-model baseline and reversal counts, changed labels and unresolved outputs. |
| `results/analysis/` | Paper tables: baseline/evidence accuracy, aggregate and directional attacks, matched pairs, common cohort and parser comparison. CSV rates are proportions; multiply by 100 for percentages. |
| `results/paper_figures/` | The two final simple plots and their provenance. |
| `results/haiku_symmetric/`, `results/sonnet_symmetric/` | Original API responses, per-request CSVs, usage logs and resumable checkpoints. These raw exports contain original parser scores. |
| `results/review/judge_requests.jsonl`, `supplemental_requests.jsonl` | Cached automatic extraction decisions and 39 separately collected missing follow-ups. Keep these to reproduce the audit offline. |
| `results/review/manual_review_queue.csv` | Unresolved answers; automatic extraction is not human validation. |

`display_id` groups related examples as `001a` through `001d`; `sample_id` preserves the source ID. Each family contains two same-claim pairs. The index is `data/processed/symmetric_index.csv`.

Primary ASR uses originally attacked examples whose baselines remain correct after rescoring. Unresolved responses stay in the denominator. `corrected_coverage_attacks.csv` separately includes supplemental follow-ups. Directional gaps are R-to-S minus S-to-R. Matched comparisons require both baseline-correct, originally attacked members of a same-claim pair.

The pre-existing `pilot_results_*.csv` files supply the local symmetric results. The separate `pilot_orig_results_*.csv` files contain historical original-FEVER results with legacy scoring and are not mixed into this audited symmetric analysis. Original local scripts, datasets and results have been left intact.

## Run a new API experiment (paid)

Check the dataset without calling the API:

```powershell
.\.venv\Scripts\python.exe haiku_test.py --dry-run
```

Use fresh folders. These commands explicitly select the paper's original acquisition policy; the runner's default for other new runs is conservative scoring.

```powershell
.\.venv\Scripts\python.exe haiku_test.py --scoring legacy --workers 4 --output-dir results/haiku_new
.\.venv\Scripts\python.exe haiku_test.py --scoring legacy --model claude-sonnet-5-5 --workers 4 --output-dir results/sonnet_new
```

Add `--max-samples 20` for a pilot; remove it to continue the same run. Completed examples are skipped. Settings are low effort and 2,048 maximum output tokens, including thinking. Do not run two processes against the same folder. API output folders are currently gitignored.

To analyze a new run, override its source, for example:

```powershell
.\.venv\Scripts\python.exe rescore_results.py --source Haiku-5.5=results/haiku_new/results.csv
```

New unclear responses remain unresolved offline. Add `--judge --max-judge-cost 0.50` only to enable paid extraction. `complete_api_coverage.py --dry-run` lists follow-ups missing after revised baseline scoring; omit `--dry-run` to collect them with the API. Re-run rescoring and analysis afterward. See each script's `--help` for options.

Offline checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
