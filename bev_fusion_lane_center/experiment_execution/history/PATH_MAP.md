# Path map: old layout (until 2026-10-24) -> new layout

Reports, memory notes and old state files name the old paths. The prefix `C:\ML-Experiment\` became `C:\ML-Experiment\bev_fusion_lane_center\`; WSL: `/mnt/c/ML-Experiment/` -> `/mnt/c/ML-Experiment/bev_fusion_lane_center/`. Every individual file move is in `MIGRATION_LOG.csv` (columns time, step, src, dst); `tools/restructure/` holds the scripts that did it.

| old | new |
|---|---|
| `scripts/` | `scripts/` (live code, launchers); historic scripts in `scripts/_archive/<group>/` or `experiments/E<id>_<slug>/code/`; data builders in `scripts/data_prep/`; `graph_eval.py` -> `evaluation/sparse_lanes/`; `viz_graph.py` -> `tools/viz/`; setup scripts -> `tools/env/`; one-off patches and job control -> `_to_delete/scripts_junk/` |
| `out/loop2/` | `runs/` (kept entries); unneeded entries in `_to_delete/runs/`; `EXPERIMENT_TABLE.*` -> `experiments/`; `LOOP*_INDEX.md`, `TRIAGE_STATE.md` -> `experiment_execution/indexes/` |
| `out/crn/` | `runs/_loop1_crn/` |
| `out/*.log`, `out/flags` | `_to_delete/out_logs/`, `_to_delete/out_flags/` |
| `loop2/STATE.md` | `experiment_execution/STATE.md` |
| `loop2/splits.json` | `data/splits.json` |
| `loop2/*.py` (report text builders) | `scripts/_archive/reports_build/` |
| `report_final/` | `reports/source/` |
| `report_final/text/lit*`, `report_final/lit7`, `lit8`, `lit_condensed.md` | `literature/sources/` (copies; originals in `_to_delete/duplicates_literature/`) |
| `reports/*Plan*.md`, `Hypothesis_Tree*`, `Concept_*`, `Growth_*`, `Literature_Compatibility*`, `Experiment_Table_EN.md` | `experiment_execution/plans/` |
| `reports/Loop16_Prompt_EN.md`, `reports/Loop_Prompt_Next_EN.md` | `experiment_execution/prompts/` (current: `experiment_execution/PROGRAM.md`) |
| `reports/loop2/` | `reports/per_experiment/` |
| `reports/Approach_*`, `reports/Proposal_*` | `reports/background/` |
| `reports/README.md` (index of 2026-10-20) | `reports/archive/README_index_2026-10-20.md`; new `reports/README.md` |
| `README.md` (2026-10-13) | `experiment_execution/history/README_2026-10-13_loop1.md`; new `README.md` in the project root |
| `data/`, `downloads/`, nuScenes archives in the root | `data/`, `downloads/` |
| `bundles/`, `mail13/`, `recon/` | `_to_delete/` |
| `recovered_loop17/` | `experiment_execution/loop17_recovery/` |
| `restructure/` | `tools/restructure/` |
| `wsl/` (Ubuntu disk image) | unchanged, stays at `C:\ML-Experiment\wsl` (registered with WSL, must not be moved by hand) |

Names that exist only inside texts of old reports: `out/loop2/e123_ladder/...` = `runs/e123_ladder/...`; `out/loop2/e17/...` = `runs/e17/...`.

## Reports moved 2026-10-24 (second cleanup)
`reports/<name>` -> `reports/detail/<name>` for: Decision_Report_EN.{html,pdf}, E2E_Differentiable_Condensed_Report_EN.pdf, E2E_Differentiable_Lane_Graph_Presentation_EN.pptx, E2E_Summary_3pages_EN.pdf, LaneGraph_Condensed_Report_EN{,_short}.{html,pdf}, Loop15_Differentiable_Positions_Report_EN.pdf, Loop16_Differentiable_Lane_Graph_Report_EN.pdf. Loop 17 draft copied to `reports/detail/`. New: `reports/weekly/` (two weekly reports + one deck). Older references in indexes and journals keep the old paths.
