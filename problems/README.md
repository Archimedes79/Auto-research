# problems - one folder per research problem (being built)

Created on 2026-11-26 by step S0 of the restructure plan (`../autoresearch/INTEGRATION_2026-11-26.md`). **Until a problem's step is done, its live folder is the old one: work there, not here.**

| folder | problem | state |
|---|---|---|
| `toy/` | the reference problem: a disc in noise, small enough for the CPU; the chain test of the system and the template of a new problem | moved here 2026-11-26; still on today's tools, rebuilt on the new core in step S1 |
| `lane_center/` | Spurmitte: a lane-centre-line graph from camera + radar, attached to an SD map | empty; live folder `../bev_fusion_lane_center/`; filled in step S5 |
| `bev_fusion/` | BEV Fusion: camera + radar BEV perception on nuScenes | empty; live folder `../bev_fusion/`; filled in step S4 |

## What a problem folder holds
Code and text only; data and run outputs live on D: (`D:\Auto-research\data\`, `runs\`), the facts of the best networks in `../champions/<name>/`. A problem writes only what is its own: `problem.json` (score and guards, splits and holdout, steps, environment, protected files), `training/train.py` (the run-folder contract), `evaluation/adapter.py` (source, subject, metrics as additive counts) and `evaluation/views.py` (one frame drawn from raw data, ground truth and a prediction). Launching, records, verdicts, reports and provenance are the core's (`../autoresearch/`).

```
<name>/
  PROBLEM.md  DESIGN.md  DECISIONS.md  HANDOVER.md  CLAUDE.md  TEST_LOG.md
  problem.json  ledger.csv (generated)
  dataset/  gt/  net/{prep,model,loss}/  training/  evaluation/
  experiments/E###_slug/{experiment.md, judge.py?, verdict.json, runs/<run>/{config,run,result}.json, report/}
  reports/  tests/  literature/  temp/ (ignored)
```
