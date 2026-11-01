# Reports

Cleaned up on 2026-10-24 (two passes). Read the two weekly reports first.

## weekly/ (the documents to read)

| file | content |
|---|---|
| `Week1_Loops1-9_Report_EN.pdf` | Week 1 (11 - 17 Oct), loops 1 - 9: baseline, grid head, key points, route transfer, robustness, fine-tuning, simulated SD map, training against map faults |
| `Week2_Loops10-17_Report_EN.pdf` | Week 2 (18 - 24 Oct), loops 10 - 17: trace network, planner consumer, differentiable builders and champion, route head along the edges, robustness with gate and map builder |
| `Loop18_Fork_Pruning_Localisation_Report_EN.pdf` | Loop 18 (24 - 25 Oct, 12 pages, updated 10:10 with the controls, failure series, third joint negative and range recall): E209 closed, rough localisation + map trust, fork pruning (J gap closed), failure-trained route head, recall variants incl. the fast j1rc and its equal-budget controls, numba assembly (chain 17.5 ms), TEST once, calibration, building blocks 2025 - 26; builder `build/reports/build_loop18_report.py` |
| `Week3_Loops18-20_Report_EN.pdf` | Week 3 (24 - 27 Oct), loops 18 - 20: seed floor (four paired seeds, Q SD 0.0025), matched vertex budgets, the damaged distance map, map benefit on new error draws, ground-truth offset by city, Poisson set loss CUT at a matched budget (-0.073 Q), D4 candidates by city, the framework test (what worked, what hurt), corrected claims, experiment index (the timing repeat E253 was blocked by another chat's training) |
| `Lane_graph_recall_val.mp4` (+ a poster png) | results video of the `recall` system (VAL scenes 23 Boston and 7 Singapore One-North, the two whose scene Q is nearest the median; 80 frames, 3 fps): front camera, prediction red over ground truth green, per-frame F1@1.0; made by `tools/prototype/make_results_video.py` (moved from `temp/loop21/` on 2026-10-31; `tools/prototype/lane_prototype.py --video` makes the same video for any scenes) (mp4v: VLC / Windows Media Player) |
| `Weeks1-2_LaneGraph_Presentation_EN.pptx` | 24-slide deck of both weeks |
| `Automated_Research_Workflow_short_EN.pptx` | 12-slide pitch for the team and managers: idea, what it gave us, what others do, how to do it, research pipeline (small vs large data, modular vs joint training), our setup, four trust rules, noise, tree / ledger / state files, what we ask for. (The older 20-slide version is staged in _to_delete/cleanup_2026-10-24b/.) |
| `Background_Approach_Sparse_Lane_Graph_EN.pdf` | the idea of the sparse lane graph |

All three weekly reports end with the list of corrected claims and an experiment index. Known gap: E209 (unseen failure types) has no result.

## Other folders

- `build/` - builders, in two subfolders (`reports/`, `decks/`); see `build/README.md`. Temporary files (html, image cache, slide exports) are not kept.
- `results/weekN/` - the result record of a week: `Week<N>_Results.md` in git, the extensive PDF report, its figures and the example videos next to it (not in git); see `results/README.md`.
- `source/figures/` - result images used by the builders; `source/weekly/` - digests, number reviews, deck charts.

## Removed

Everything older is in `../_to_delete/` (`reports_cleanup_2026-10-24/` and `cleanup_2026-10-24b/`), same relative paths: condensed reports of loops 1 - 14, loop 4 / 5 / 15 / 16 reports, decision and E2E reports, html copies of the PDFs, per-experiment mini reports, loop 1 figures, old builders and text sources. Per-experiment detail is in `../experiments/`.
2026-10-31: the files of `weekly/` that git does not hold (binaries) and the `temp/` scratch were found missing from disk, also not in `../_to_delete/` (cause unknown). The scratch came back from git; the status report after loop 21, the loop-18 report, the results video and the two-week deck were rebuilt from their builders the same day (the video run reproduced the stored `recall` card). The loop-17 draft report (Loop17_DRAFT_Robust_Differentiable_Lane_Graph_Report_EN.pdf, built before the E209 crash) could not be rebuilt and is gone.
To finish the loop 17 report (open item in HANDOVER.md), move the builder files (`build_loop17_report.py`, `loop17_text*.py`, `fig_p17_*.py`, `build_e2e_condensed.py`, `report_lib.py`, `finish_report6.ps1`) and `source/text|tables|cards|review|build` back from `cleanup_2026-10-24b/reports/`.

## Rebuild
```
wsl bash tools/env/_py14.sh reports/build/reports/build_week2_report.py
powershell -File reports/build/reports/print_weekly.ps1 -Html <html> -Pdf <pdf>
powershell -File reports/build/decks/build_research_workflow_pptx.ps1
```
The html is an intermediate file: print the PDF, then do not keep the html next to it.
Rules: English; architecture drawing + numbers table + result images per experiment; one fresh numbers reviewer before a report is final; state the setting and the noise band.
