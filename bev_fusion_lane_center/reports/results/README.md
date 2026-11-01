# results - the status results of a week, one folder per report

Each `weekN/` folder holds the short result record in git (`WeekN_Results.md`) and, **not in git** (binaries are ignored by `.gitignore`), the extensive report as PDF (`WeekN_Status_Report_EN.pdf`), its figures (`figures/`) and example videos (`video/`). The builders are `../build/reports/fig_w3_status.py` and `build_w3_status_report.py`, the PDF is printed by `../build/reports/print_weekly.ps1`.

| folder | content |
|---|---|
| `week3/` | 27 October - 2 November 2026, **Spurmitte on the frozen GaussianCaR features only**: the prototype of record, the week's experiments, the limits of the design (`Week3_Results.md`; PDF, one example video) |
| `week3_framework/` | the same week, part 2, **not about the Spurmitte results**: the BEV system, the end-to-end set-up (gradient measurements, card E269), the tools and the disk work (`Week3_Framework_Results.md`; PDF, figures and one dashboard video, not in git; builder `build_w3_framework_report.py`; nothing was run for it) |
