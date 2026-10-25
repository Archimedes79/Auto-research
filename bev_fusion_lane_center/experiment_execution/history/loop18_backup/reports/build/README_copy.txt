# build/

Builders of the documents in `../weekly/`. Run from the project root; all outputs go to `../weekly/`, temporary files to the temp folder (image cache `/tmp/weekly_img_cache`, slide exports in `%TEMP%`).

| folder | file | what |
|---|---|---|
| `reports/` | `build_week1_report.py`, `build_week2_report.py` | write the report as html (run with `tools/env/_py14.sh`); layout in `weekly_lib.py` |
| `reports/` | `print_weekly.ps1 <html> <pdf>` | html to PDF via headless Chrome; then delete the html |
| `decks/` | `make_deck_charts.py` + `build_weekly_pptx.ps1` | charts and the 24-slide two-week deck (PowerPoint COM) |
| `decks/` | `make_systematics_charts.py` + `build_research_workflow_pptx.ps1` | chart and the 12-slide pitch; `ppt_lib.ps1` = shared slide helpers |

Numbers come from `../source/weekly/digest_week*.md` and the figures in `../source/figures/`. The loop 17 report builder is in `../../_to_delete/cleanup_2026-10-24b/reports/build/`.
