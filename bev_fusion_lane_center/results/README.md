# results/ - the result folders of the lane project

One folder per experiment (`E273/`, `E274/`, ...) or per run (`E273.p_s0/`); a report page of one experiment would be `E<id>.report/`. A run folder is the result package that `autoresearch/tools/package.py` writes (`package.json`: code tag, GT version, checkpoint, numbers; `provenance.json`: what the number rests on), plus the figures, frames and videos of that run; its own `README.md` says which card it belongs to and the verdict.

An experiment folder holds the `README.md` index of the experiment and its small result files (sweeps, tables). The next card's `results.md` (written by `result.py`, which `decide.py --write` calls; linked from the schedule's `result` column) belongs next to its card in `../experiments/E<id>_<slug>/`; this folder keeps what a run produced.

In git: the `README.md` files, `package.json`, `provenance.json` and small result json files. Not in git: frames, videos and other heavy files (about 1.2 GB on disk); the checkpoints stay in `../runs/`.

Reproduce a run: `python autoresearch/tools/provenance.py repro <run id>`.

**End-to-end reports** (built by `reporter/reporter.py`). Three are kept here (the owner's choice, D151):

| folder, open this file | what | in git |
|---|---|---|
| `E2E_Report_2026-11-25/E2E_Bericht_2026-11-25.html` | the latest: night report of 2026-11-24 / 25 (D148) | sources and result files |
| `E2E_Report_2026-11-23/E2E_Bericht_2026-11-23.html` | final report of the SD-map line and E392 (D138) | sources and result files |
| `E2E_Champion_Report/E2E_Champion_Bericht.html` | the token champion report of 2026-11-17 with the simple top-to-bottom block drawing of the system | nothing (on disk only); its sources are in the git history: `git show b54a2616^:bev_fusion_lane_center/results/E2E_Champion_Report/report.json` and the `parts/` next to it |

The sources of a report are in git (`report.json`, `parts/*.md`, `parts/*.svg`); the built html, the code zip and the pictures (`parts/figures/`) are not: rebuild with `reporter.py build` / `check` / `pack`. Every report holds the short article "how the ground truth of a lane graph must look" and an architecture drawing of the current system (`../reports/README.md`).

The older overall reports and the report pages of E273 and E276 were moved on 2026-11-25 to `../_to_delete/results_reports_2026-11-25/` (not in git; the list there says what is in it). `l6_view_B/` and `l6_view_E312/` are pictures and videos of the l6 ground-truth view, not reports.
