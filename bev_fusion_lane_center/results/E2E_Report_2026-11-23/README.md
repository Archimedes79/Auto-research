# E2E_Report_2026-11-23: the final report of the SD-map line and E392 (D138)

The latest report of the lane project: the state of the system on 2026-11-23, E388 / E381 / E389 / E389b in short, the SD-map line (E390 step 1 SOUND, E391 step 2 NO FUSION), E392 (r2 replicates in quality, its cost misses the bar: not adopted), the drafts E393 / E394 and what the owner can decide next. German text, English code comments. Older reports are not kept in git (D140); they are in the git history.

## What is in git (the sources) and what is not
| file | in git | what it is |
|---|---|---|
| `report.json` | yes | title, abstract, chapter list, the `values` pointers (number name -> result file and key), the source list, the code-package list |
| `parts/*.md` | yes | the chapters (numbers are `{{names}}` that the builder fills from the result files); `parts/src_e389_*.md` = the three text parts of the 2026-11-20 report that this one reads numbers from (copied when that report left the tree, D140) |
| `data/runs/<run>/<file>` | yes | the 106 small result files (json / md cards, 1.5 MB) the numbers are read from: copies of the files under `runs/` (D139 / D140, 2026-11-23: all data the latest report needs stays available); `report.json` points here |
| `parts/figures/*.png`, `parts/figures/*.json` | **no** (images and generated tables are never in git) | the five figures and the numbers of the example frames; made by `tools/map_assoc_fig.py` and `tools/report_charts.py` |
| `E2E_Bericht_2026-11-23.html`, `E2E_Code_Paket_2026-11-23.zip` | **no** (generated) | the report and the code package, built by `reporter/reporter.py` |

## Rebuilding it
The numbers come from the result files in `data/runs/` (in git since D140; copies of `runs/e390_*`, `runs/e391_*`, `runs/e392_*`, `runs/e394_step0`): a fresh checkout rebuilds the text with every number and checks it, no run folder needed:

```
python reporter/reporter.py build bev_fusion_lane_center/results/E2E_Report_2026-11-23 --strict      # refuses a PENDING number or an unknown placeholder
python reporter/reporter.py check bev_fusion_lane_center/results/E2E_Report_2026-11-23              # every number of the text is in a source
python autoresearch/tools/check_numbers.py <the html> <the cards and result files named in report.json "sources">
python reporter/reporter.py pack  bev_fusion_lane_center/results/E2E_Report_2026-11-23              # the code package (zip)
```
What a checkout does NOT rebuild: the figures, the html and the zip stay generated (not in git); the figures need the caches and run folders that are not in git (`runs/e365_cache`, the E389 / E390 / E391 data under `runs/`), so a clone rebuilds the numbers and the text, not the figures. The figures: `bash tools/env/_py14.sh map_assoc_fig.py --seed 0 --out <absolute dir>` and `report_charts.py --out <absolute dir>` (from `bev_fusion_lane_center/`, WSL, the `gcar` environment, the caches of E365 / E389 / E390 / E391). Each has a `--selftest` that needs no data.

## Where the numbers come from
Every number with a name in the text is read from a result file by the pointers in `report.json` (`values`); the thresholds, the nulls, the cost and the champion numbers are typed from the experiment cards (`bev_fusion_lane_center/experiments/E388_*` ... `E394_*`) and `HANDOVER.md`, and checked against them by `check_numbers.py`. The cards say where each run's code and scratch scripts are kept (git tags `lcr/<run id>`).
