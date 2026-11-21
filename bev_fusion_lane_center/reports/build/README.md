# build/

Builders of the documents in `../weekly/`. Run from the project root; all outputs go to `../weekly/`, temporary files to the temp folder (image cache `/tmp/weekly_img_cache`, slide exports in `%TEMP%`).

| folder | file | what |
|---|---|---|
| `reports/` | `build_week1_report.py`, `build_week2_report.py` | write the report as html (run with `tools/env/_py14.sh`); layout in `weekly_lib.py` |
| `reports/` | `fig_w3_results.py`, `build_week3_report.py` | the week 3 report (loops 18 - 20): the figures first (they also write `../source/weekly/week3_numbers.json`), then the html; cards from `runs/eval`, E240 / E252 / E255 numbers from result files; PDF with an absolute path in `print_weekly.ps1` |
| `reports/` | `build_loop18_report.py`, `fig_p18_arch.py`, `fig_p18_frames.py`, `fig_p18_results.py`, `fig_p18_final.py` | the loop 18 report (numbers read from `runs/e17/tables`, `runs/e18/`, `runs/e123_ladder/diag`) and its figures |
| `reports/` | `fig_w3_status.py`, `build_w3_status_report.py` | the week-3 status report (27 October - 2 November): figures from the metric cards, `bev_fusion/records/BOARD.md` and the gradient reports, the two example videos copied; writes into `../results/week3/` (the md there is in git, the pdf, figures and videos are not) |
| `reports/` | `build_status21_report.py` | the status report after loop 21 (2026-10-31), html, printed with `print_weekly.ps1` to `../weekly/Status_after_Loop21_EN.pdf`; numbers through `reports/source/weekly/s21_numbers.json`, the review section from `s21_review.json` |
| `reports/` | `build_w3_framework_report.py` | the week-3 framework report (the BEV system, the road to an end-to-end lane chain, tools and disk work: everything that is not the lane results); figures in `../results/week3_framework/`, every number read from a stored file or quoted with the summary named next to it |
| `reports/` | `print_weekly.ps1 <html> <pdf>` | html to PDF via headless Chrome; then delete the html |
| `decks/` | `make_deck_charts.py` + `build_weekly_pptx.ps1` | charts and the 24-slide two-week deck (PowerPoint COM) |
| `decks/` | `make_systematics_charts.py` + `build_research_workflow_pptx.ps1` | chart and the 12-slide pitch; `ppt_lib.ps1` = shared slide helpers |

Numbers come from `../source/weekly/digest_week*.md` and the figures in `../source/figures/`. The loop 17 report builder was staged in `_to_delete/cleanup_2026-10-24b/reports/build/`, which was emptied on 2026-11-02 (D42); it is only in the git history (if git ever held it).
