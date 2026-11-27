# problems - one folder per research problem

Each is run by the core (`python3 -m research <verb> <problem>`, [`../autoresearch/research/README.md`](../autoresearch/research/README.md));
its best networks are in `../champions/<name>/`. A new problem: [`../autoresearch/EXTENDING.md`](../autoresearch/EXTENDING.md).

| folder | problem |
|---|---|
| [`toy/`](toy/README.md) | the reference problem: a disc in noise, small enough for the CPU; the chain test of the system and the template of a new problem |
| [`lane_center/`](lane_center/README.md) | Spurmitte: a lane-centre-line graph from camera + radar, end to end differentiable |
| [`bev_fusion/`](bev_fusion/README.md) | BEV Fusion: camera + radar bird's-eye-view perception on nuScenes (maps, objects, motion) |

## What a problem folder holds
Code and text only; data and run outputs are in the two roots of the machine file (`research machine`). What a problem writes for the core is
listed in the core's README ("Files a problem writes"); launching, records, verdicts, reports and provenance are the core's.

```
<name>/
  README.md  PROBLEM.md  DESIGN.md  DECISIONS.md  HANDOVER.md  TEST_LOG.md    (the toy: README.md and TEST_LOG.md only)
  problem.json   ledger.csv (generated)
  dataset/ gt/ net/{prep,model,loss}/ training/ evaluation/ tests/            (as far as the problem needs them)
  experiments/E###_slug/{experiment.md, judge.py?, report/?, verdict.json, runs/<run>/{config,run,result}.json}
  reports/<report>/ (a report's report.json and parts)   temp/ (scratch, not in git)
```
