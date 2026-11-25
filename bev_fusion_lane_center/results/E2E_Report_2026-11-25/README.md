# E2E_Report_2026-11-25: the night report of 2026-11-24 / 25 (D148)

What ran since the report of 2026-11-23: E395 (Catmull-Rom read-out, position head, addendum A1), E391b parts A and B (association with the SD map; the caveat on E391's NO FUSION), E393 (the pyramid in the token loss), E394 (trust in a wrong map, closed by hand) and the BEV card t25 (one camera frame and one radar scan, the past carried recursively); the state of the system (champion unchanged, D120), what was cut and why, what is open for the owner, and how a run is repeated from a clone. German text. The older report stays next to it.

## What is in git (the sources) and what is not
| file | in git | what it is |
|---|---|---|
| `report.json` | yes | title, abstract, chapter list, the source list the number check reads |
| `parts/*.md` | yes | the chapters; `02_gt.md` is the ground-truth article, copied unchanged from the report of 2026-11-23 |
| `data/runs/<name>/<file>` | yes | small result files (json / md / txt) copied unchanged from the run folders of both projects: the E395 step-2 verdicts, the E393 token cards, the E391b cards, the warm score tables of the t25 runs; one exception: `e391b_partB/verdict_d.md` is the verdict line of the run log of `E391b.verdict_d`, unchanged, under a heading |
| `parts/figures/*` | **no** (generated pictures are never in git) | the system drawing cut, four charts and two example frames, made by `bev_fusion_lane_center/tools/night_report_figs.py` |
| `E2E_Bericht_2026-11-25.html` | **no** (generated) | the report, built by `reporter/reporter.py` |

## Rebuilding it
First the pictures (from `bev_fusion_lane_center/`, WSL, environment `gcar`; CPU only):

```
bash tools/env/_py14.sh night_report_figs.py --out <absolute path of this folder>/parts/figures            # drawing cut and charts: from files that are in git
bash tools/env/_py14.sh night_report_figs.py --out <absolute path of this folder>/parts/figures --frames   # also the two frames: needs runs/ (E391, E391b), the E365 / E373 / E390 / E391 caches and nuScenes
```

A checkout without the run folders and caches cannot draw the two frames; `--strict` then refuses the build (two missing pictures), the build without `--strict` marks them PENDING. `data/runs/e391b_frames/e391b_frames.json` (in git) holds the numbers of those two frames.

Then, from the repository top:

```
python reporter/reporter.py build bev_fusion_lane_center/results/E2E_Report_2026-11-25 --strict     # refuses a PENDING number or an unknown placeholder
python reporter/reporter.py check bev_fusion_lane_center/results/E2E_Report_2026-11-25              # every decimal number of the text is in a source
```

The check reads the sources named in `report.json`: the experiment cards of E2xx / E3xx, the t25 card and its compare files under `bev_fusion/experiments/`, `DECISIONS.md`, `HANDOVER.md`, the report of 2026-11-23 and `data/runs/`.

## Where the numbers come from
This report has no named placeholders: the numbers are typed from the experiment cards and the result files above, and the check confirms that each decimal number occurs in one of them (it does not confirm that it is quoted in the right place; a fresh numbers review read the draft against the sources). The cards say where each run's code is kept; `autoresearch/tools/recipe.py show <run id>` tells how to repeat a run from a clone.
