# Reports

Final documents at this level; everything needed to rebuild them below. The old index (2026-10-20, with the content of every report) is `archive/README_index_2026-10-20.md`.

| file | content |
|---|---|
| `LaneGraph_Condensed_Report_EN.pdf` (+ `.html`, `_short`) | condensed report of loops 1-14 (229 pages; short version 136 pages) |
| `Decision_Report_EN.pdf` (+ `.html`) | what works best, what was tried, what to hand to the planner |
| `E2E_Differentiable_Condensed_Report_EN.pdf`, `E2E_Summary_3pages_EN.pdf`, `E2E_Differentiable_Lane_Graph_Presentation_EN.pptx` | all experiments by use in a differentiable E2E system; 3-page summary; deck |
| `Loop15_Differentiable_Positions_Report_EN.pdf` | loop 15: differentiable vertex positions (E120 / E121) |
| `Loop16_Differentiable_Lane_Graph_Report_EN.pdf` | loop 16 + 16b: eight differentiable builders, champion D7, route head along the edges, 31 ms chain |
| loop 17 | draft `source/build/Loop17_Robust_Differentiable_Lane_Graph_Report_EN.pdf` (2.1 MB, built before the E209 crash); final report pending (HANDOVER) |

## Folders
- `build/` - **report builders and figure scripts** (`build_loop15_report.py`, `build_loop16_report.py` = template for later loops, `loop15_text*.py`, `loop16_text*.py`, `fig_p14.py` ... `fig_p16_*.py`, `arch_figs*.py`, `report_lib.py`, `finish_report6.ps1` (prints the PDF with headless Chrome), `build_e2e_*`). New report = new `build_loop<NN>_report.py` + `loop<NN>_text*.py` here, output into `source/build/`, final PDF copied to this level.
- `source/` - build workspace (former `report_final/`): `text/` text parts, `figures/`, `tables/`, `cards/`, `build/` (html + pdf output), `review*/` (numbers-review rounds).
- `per_experiment/` - mini reports (html / md / json / pdf) of the early experiments E27 - E72.
- `background/` - approach and proposal of the sparse-graph idea.
- `assets/` - figures used by plan documents; `archive/` - older report versions.

Rules: English; architecture drawing + numbers table + result images per experiment; one fresh numbers reviewer before a report is final; every report states the setting (exact poses, simulated SD map, scene-disjoint splits) and the noise band. Plans and prompts are not here but in `../experiment_execution/`.
