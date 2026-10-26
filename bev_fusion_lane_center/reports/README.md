# Reports

Cleaned up on 2026-10-24 (two passes). Read the two weekly reports first.

## weekly/ (the documents to read)

| file | content |
|---|---|
| `Week1_Loops1-9_Report_EN.pdf` | Week 1 (11 - 17 Oct), loops 1 - 9: baseline, grid head, key points, route transfer, robustness, fine-tuning, simulated SD map, training against map faults |
| `Week2_Loops10-17_Report_EN.pdf` | Week 2 (18 - 24 Oct), loops 10 - 17: trace network, planner consumer, differentiable builders and champion, route head along the edges, robustness with gate and map builder |
| `Loop18_Fork_Pruning_Localisation_Report_EN.pdf` | Loop 18 (24 - 25 Oct, 12 pages, updated 10:10 with the controls, failure series, third joint negative and range recall): E209 closed, rough localisation + map trust, fork pruning (J gap closed), failure-trained route head, recall variants incl. the fast j1rc and its equal-budget controls, numba assembly (chain 17.5 ms), TEST once, calibration, building blocks 2025 - 26; builder `build/reports/build_loop18_report.py` |
| `Weeks1-2_LaneGraph_Presentation_EN.pptx` | 24-slide deck of both weeks |
| `Automated_Research_Workflow_short_EN.pptx` | 12-slide pitch for the team and managers: idea, what it gave us, what others do, how to do it, research pipeline (small vs large data, modular vs joint training), our setup, four trust rules, noise, tree / ledger / state files, what we ask for. (The older 20-slide version is staged in _to_delete/cleanup_2026-10-24b/.) |
| `Loop17_DRAFT_Robust_Differentiable_Lane_Graph_Report_EN.pdf` | draft of the loop 17 report, built before the E209 crash; cannot be rebuilt as is (see Removed) |
| `Background_Approach_Sparse_Lane_Graph_EN.pdf` | the idea of the sparse lane graph |

Both weekly reports end with the list of corrected claims and an experiment index. Known gap: E209 (unseen failure types) has no result.

## Other folders

- `build/` - builders, in two subfolders (`reports/`, `decks/`); see `build/README.md`. Temporary files (html, image cache, slide exports) are not kept.
- `source/figures/` - result images used by the builders; `source/weekly/` - digests, number reviews, deck charts.

## Removed

Everything older is in `../_to_delete/` (`reports_cleanup_2026-10-24/` and `cleanup_2026-10-24b/`), same relative paths: condensed reports of loops 1 - 14, loop 4 / 5 / 15 / 16 reports, decision and E2E reports, html copies of the PDFs, per-experiment mini reports, loop 1 figures, old builders and text sources. Per-experiment detail is in `../experiments/`.
To finish the loop 17 report (open item in HANDOVER.md), move the builder files (`build_loop17_report.py`, `loop17_text*.py`, `fig_p17_*.py`, `build_e2e_condensed.py`, `report_lib.py`, `finish_report6.ps1`) and `source/text|tables|cards|review|build` back from `cleanup_2026-10-24b/reports/`.

## Rebuild
```
wsl bash tools/env/_py14.sh reports/build/reports/build_week2_report.py
powershell -File reports/build/reports/print_weekly.ps1 -Html <html> -Pdf <pdf>
powershell -File reports/build/decks/build_research_workflow_pptx.ps1
```
The html is an intermediate file: print the PDF, then do not keep the html next to it.
Rules: English; architecture drawing + numbers table + result images per experiment; one fresh numbers reviewer before a report is final; state the setting and the noise band.
