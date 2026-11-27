# reporter

A standalone tool: a folder of markdown parts becomes one paper-style **html report**; numbers are **filled from result files** and **checked against their sources**; report, code and helper scripts are **packed** into a size-limited zip. It imports nothing of any project. The chapters and what each must hold: `CHAPTERS` in `reporter.py`; how to write a good one: `SKILL.md` (an agent skill: install it as `.claude/skills/reporter/SKILL.md`). Needs python 3.8+ with `markdown-it-py` (Pillow optional); here `wsl ~/miniconda3/envs/gcar/bin/python reporter.py ...`.
```
python reporter.py new   my_report --title "My system"   # scaffold: report.json and parts/, one per chapter with its prompt (refuses a folder that has a report.json)
python reporter.py build my_report                        # -> my_report/report.html (+ media/); prints figures, tables, PENDING marks
python reporter.py check my_report                        # every decimal number of the text must be in the sources
python reporter.py pack  my_report                        # -> package.zip: report.html, media/, README.txt, code/, utilities/
python reporter.py selftest
```

## Minimal example
`my_report/report.json` (paths relative to the report folder; a section's `part` and `svg` relative to `parts/`):
```json
{"title": "My system", "subtitle": "Project, date, status",
 "abstract": "Four sentences with the headline result {{topo_s0}} against its baseline.",
 "teaser": {"image": "parts/teaser.jpg", "caption": "A real example: input left, output right."},
 "sections": [{"id": "quant", "title": "5. Quantitative results", "part": "05_quant.md", "svg": "02_system.svg"}],
 "values": {"topo_s0": {"file": "../runs/a/metrics.json", "key": "scores.topo_f1.0", "fmt": "{:.4f}"}},
 "sources": ["../experiments/E1*/experiment.md", "../runs/**/*.json"],
 "utilities": [{"path": "../scripts/plot_curves.py", "usage": "python plot_curves.py run_dir"}],
 "pack": {"limit": 2000000, "code": ["../training/**/*.py"], "strip": "../training", "readme": "What is in the package ..."}}
```
`my_report/parts/05_quant.md`:
```
Table: Topo F1 on VAL, two seeds.

| system | s0 |
|---|---|
| ours | {{topo_s0}} |

![A failure case: the far lane is missed.](parts/fail.jpg)
![A short drive.](media/clip.mp4)
```
A `Table: ...` line becomes the numbered caption of the table after it, the alt text of a picture (embedded) or video (next to the html) its caption.

## Rules the tool enforces
- `{{name}}` is filled from `values` (a JSON key path with a format, or a literal). A missing file, key, name or picture becomes a highlighted **PENDING** mark. `check` looks for every decimal number of the text (code blocks aside) in the `sources`, exactly or as a rounded source number.
- `--strict` (build, check) exits 1 while a PENDING mark or an unfound number is left; `build --strict` also without an architecture drawing: a section's `svg` part that is not the scaffold's placeholder (a problem's: its `training/architecture.svg`).
- `pack` ships the `utilities` (chapter B lists them, by default with the first line of their docstring) and refuses a zip at or above `pack.limit` (`include_report: false`: code and utilities only, the limit counts html + zip) and, with `require_clean_git`, uncommitted code. Nothing is sent.
- A **copy** of a report folder names the original in `"origin"`: paths that leave the folder are read from there. `research report <problem> <folder>` builds such a copy in the run root and draws the pictures its `"draw"` names.

`archdraw.py` (same folder) draws a system from a JSON spec: `render` writes the svg and the file map of the spec's md; `check` exits 1 when a box points at nothing, a file is in no box, or svg / md are stale; `--selftest`.
