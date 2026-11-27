# problems - one folder per research problem

Each problem is run by the core (`../autoresearch/research/`, `python -m research <verb> <problem>`); its best networks are in
`../champions/<name>/`. The work before the restructure of 2026-11-27 (D154) is in the git tag `v1`.

| folder | problem |
|---|---|
| `toy/` | the reference problem: a disc in noise, small enough for the CPU; the chain test of the system (`python -m research.selftest`) and the template of a new problem |
| `lane_center/` | Spurmitte: a lane-centre-line graph from camera + radar, attached to an SD map (formerly `bev_fusion_lane_center/`) |
| `bev_fusion/` | BEV Fusion: camera + radar BEV perception on nuScenes (formerly `bev_fusion/` at the root) |

## What a problem folder holds
Code and text only; data and run outputs are in the two roots of the machine file (`~/.research/machine.json`; on this PC
`D:\Auto-research\data` and `D:\Auto-research\runs`). A problem writes only what is its own (`../autoresearch/research/README.md`, "Files a
problem writes"): `problem.json` (score and guards, splits and holdout, steps, environment, protected files), `training/train.py` (the
run-folder contract), `evaluation/evaluation.json` with the adapter it names (source, subject, metrics as additive counts) and the cards of
its experiments. Launching, records, verdicts, reports and provenance are the core's.

```
<name>/
  README.md  PROBLEM.md  DESIGN.md  DECISIONS.md  HANDOVER.md  TEST_LOG.md
  problem.json  ledger.csv (generated)
  dataset/  gt/  net/{prep,model,loss}/  training/  evaluation/
  experiments/E###_slug/{experiment.md, judge.py?, verdict.json, runs/<run>/{config,run,result}.json}
  tests/  temp/ (ignored)
```
