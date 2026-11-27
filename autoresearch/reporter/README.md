# reporter

A small, standalone documentation system. It turns a folder of markdown parts into a **paper-style, engineer-readable report**: one html file (pictures embedded, videos next to it), numbers **filled from result files**, every number **checked against its sources**, the **helper scripts listed and kept**, and report + code + utilities **packed** into a size-limited zip. It imports nothing from any project; copy this folder anywhere.

## The shape of a report (an extended paper)
Abstract and a teaser figure first, then:

| | chapter | what it holds |
|---|---|---|
| 1 | Introduction: purpose and status | the problem in plain words, what was built, the headline result against its baseline, a map of the report |
| 2 | The system: architecture and description | one drawing, the chain, what is learned and what is a named rule, the loss, parameters, time per frame |
| 3 | Required ground truth | what the labels must contain and why, how they are made, **with pictures**, how to know they are right, limits |
| 4 | Experiments done, very short | one row per step: idea, result, verdict (negatives stay in) |
| 5 | Quantitative results | numbered tables with split and denominator, controls, seeds; plots and videos |
| 6 | Qualitative results: good cases and issues | real examples chosen by a written rule, failure modes with counts, baseline and winner side by side |
| 7 | The winning systems | each winner with its architecture, numbers, cost and where the weights are |
| 8 | Conclusion and next steps | three lines of what it shows, what is still hard, next steps very brief |
| A | How to repeat the result | data, commands, seeds, numbers to hit per stage, checks, pitfalls |
| B | Utilities | the helper scripts that are kept (listed automatically, shipped in the package) |
| C | Sources of the numbers | generated |

Layout: serif text, sans headings and tables, numbered figures ("Figure 3.") and tables with captions, an abstract box, a print stylesheet (browser "Save as PDF" gives a clean paper), light and dark mode.

## Use (one page)
Needs python 3.8+ with `markdown-it-py` (Pillow optional: it recompresses images). In this workspace: `wsl ~/miniconda3/envs/gcar/bin/python reporter.py ...` (it has both).

```
python reporter.py new   my_report --title "My system"   # scaffold: report.json + the chapters above, each with a prompt
python reporter.py build my_report                        # -> my_report/report.html (+ media/); prints figures, tables, PENDING marks
python reporter.py check my_report                        # every decimal number of the text must be in the sources
python reporter.py pack  my_report                        # -> package.zip: report.html, media/, README.txt, code/, utilities/ ; refuses over the limit
python reporter.py selftest                               # builds, checks and packs a throw-away example
```
`--strict` on `build` / `check` exits 1 when a PENDING mark or an unfound number is left: the last gate before sending. `build --strict` also exits 1 when the report has no architecture drawing: every report holds one (the owner, 2026-11-25), i.e. at least one section has an `svg` part (a file in `parts/`; the `svg` path is relative to `parts/`) that exists and is not the scaffold's placeholder; a teaser svg or an svg linked as a markdown image does not count. A problem's drawing is its `training/architecture.svg` (made by `archdraw.py`): copy it or a cut of it into `parts/`.

## Minimal example
`my_report/report.json` (paths relative to the report folder):
```json
{"title": "My system", "subtitle": "Project, date, status",
 "abstract": "Four sentences with the headline result {{topo_s0}} against its baseline.",
 "teaser": {"image": "parts/teaser.jpg", "caption": "A real example: input left, output right."},
 "sections": [{"id": "quant", "title": "5. Quantitative results", "part": "05_quant.md"}],
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
| baseline | 0.6259 |
| ours | {{topo_s0}} |

![A failure case: the far lane is missed; look at the right edge.](parts/fail.jpg)   <- a numbered figure, embedded
![A short drive.](media/clip.mp4)                                                   <- a video next to the html, with controls
```
The line `Table: ...` before a table becomes its numbered caption; the alt text of an image becomes its caption. The scaffold gives the other chapters as `parts/01_intro.md ... parts/A1_repeat.md` and the drawing as `parts/02_system.svg`.

## Rules the tool enforces
- **Numbers come from files.** `{{name}}` is filled from `values` (a JSON key path with a format, or a literal), also in the abstract and captions. A missing file or key, an unknown name or a missing picture becomes a visible, highlighted **PENDING** mark, never an empty cell.
- **`check`** takes every decimal number in the text (abstract, captions, chapters; after filling; code blocks excluded) and looks for it in the `sources`: found exactly, or as a rounded form of a source number; everything else is listed with its line.
- **Utilities.** `utilities` lists the helper scripts that stay with the work; chapter B shows name, what it does (the first line of its docstring unless `what` is given) and how to run it; `pack` ships them in `utilities/`.
- **Size.** With `pack.include_report: false` the zip holds code + utilities only (the html is sent next to it) and the limit counts html + zip together. `pack` refuses a zip at or above `pack.limit` and, with `require_clean_git`, code or utilities with uncommitted changes. Videos are not embedded in the html; they travel in `media/` (and in the zip).
- **Nothing is sent.** The tool writes files only.

## Files
| file | what |
|---|---|
| `reporter.py` | the whole report tool, one file, with `selftest` |
| `archdraw.py` | the system drawings: a JSON spec -> svg + the file map of its md (`render`), and `check` (exit 1 when a box points at nothing, a file is in no box, or svg / md are stale); `python archdraw.py render <spec.json>`, `--selftest` |
| `SKILL.md` | the procedure and the quality bar for writing such a report (an agent skill: copy it to `.claude/skills/reporter/SKILL.md` to install) |
| `README.md` | this page |

## Not in the tool, on purpose
Choosing the examples, judging the results and writing the plain-language reading stay with the author; so does a fresh reviewer pass before sending. The tool makes the mechanical part repeatable and the numbers traceable.
