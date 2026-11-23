# BEV Fusion winner (by value, a tie inside the band): t21_ctl = the t20 network continued 1500 steps from t20_base_s0 (the equal-budget control arm of t21); code of record at tag bf/t21.ctl_s0 (commit 388a36cdf0); no weights

A winner folder: the source of the system, small enough to hand over; **no weights**. Written by `winner.py` on 2026-11-23 from the code at tag `bf/t21.ctl_s0` (commit `388a36cdf0`, extracted with `git archive`; D140). The python files, the README of `training/` and the two configs of the arm (`configs/t21_ctl.json`, its base `configs/t20_base.json`); the other configs, the drawing and the presets of that commit are in the tag.

## What is here

`code/` holds 47 file(s), 862015 bytes. `MANIFEST.json` lists every file with its sha256; `python winner.py verify <this folder>` re-checks them.

## Numbers to reproduce

VAL 1000 frames of 82 val scenes, GT v8 (signature 0.5_4_6a7aa7864a), EMA weights, cold memory, dt 0: board7 (mean of drivable IoU@0.5, walkway IoU@0, carpark IoU@0, ped crossing IoU@0, lane divider IoU@0.5, road divider IoU@1, det mAP) 0.45489 / 0.45482 (s0 / s1), mean 0.45486 (records/keep3.json); t23_C 0.44969 and t20_base_s0 0.44654 lie inside the band 0.02: a TIE by the metric, t21_ctl chosen by value by hand (2026-11-10). 102.5 ms per frame, idle GPU. Not SOA: behind GaussianCaR on the six map rows and behind CRN on objects (records/t26_g0/). The numbers are on GT v8; the live GT is v13 / v14 (t21_ctl not re-scored).

## How to run

```
DOES NOT RUN AS IS: the code predates D76 and imports autogt (renamed GtGenerator, alias removed D78; D81: old runs are not run again); only the weights load under the live code (init_from). As it ran (ledger): bash temp/train/run_ar.sh temp/compose/t21_ctl_s0.json under run.py, config training/configs/t21_ctl.json (init_from runs/t20_base_s0/ckpt_00006000.pt, 1500 steps, lr 1e-4, warm-up 100, EMA 0.99, batch 3 x accum 2); evaluation then: bash temp/t21/eval_arm.sh runs/t21_ctl_s0; today's front door: python evaluation/evaluate.py runs/<run> --quant. Seed 1 ran at tag bf/t21.ctl_s1 (commit e544bdf063; the working tree changed while it ran). Alternative winner by role (the owner's choice): t20_base_s0 (tag bf/t20.base_s0, commit bd7eb0181b, board7 0.44654, one seed, the accepted reference baseline of D48).
```

## Weights

bev_fusion/runs/t21_ctl_s0/ckpt_00001500.pt (455,747,229 B) and runs/t21_ctl_s1/ckpt_00001500.pt (455,747,357 B), sha256 in records/keep3.json; warm-start parent runs/t20_base_s0/ckpt_00006000.pt; not in git

## The composed job configs

`compose/` holds the three composed configs that the runs started from (`t20_base_s0.json`, the warm-start parent; `t21_ctl_s0.json`; `t21.ctl_s1.json`), copied from the scratch folder `temp/compose/` before D140 removed it; the full config of each run is also in its `runs/<run>/config.json`.
