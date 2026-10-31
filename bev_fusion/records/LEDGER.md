# Run ledger

Generated from the csv by `autoresearch/tools/ledger.py render` - edit the csv (or use `ledger.py note`), not this file.

## Loops

From `autoresearch/tools/metrics.py` (columns explained there; `-v` shows how each row was counted).

| loop | runs | failed | GPU h | exps | KEEP | CUT | TIE | INC | other | KEEP x2 | pred hit | mean err | TEST |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Runs

| id | status | hypothesis | prediction | rule | basis | seed | result | decision | started | min | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smoke.d23 | ok |  |  |  |  |  | launcher smoke test (D23): snapshot and tag bf/smoke.d23, ledger row, done flag | setup check | 2026-10-31T15:20 | 0 | dcb594c1b9 |
| r1.codec_cv | ok | Residual (delta) and physics-derived (phys) coding of range / direction keep more radar detections than the absolute coding, also across scenes | delta - abs kept/oracle at tolerances x0.25: mean over 5 scene folds >= +5 points, wins >= 4 of 5; phys - delta within 2 points; overfitting excess < 3 points for every codec | A better than B if the mean paired fold difference of kept/oracle exceeds max(2 points, seed SD) and t > 2.776 (df 4); else TIE (take the simpler codec); overfitting flagged above 3 points excess over the oracle gap |  |  | tol x0.25, 5 scene folds, 1 seed, 1000 steps: kept/oracle abs 74.9 +- 5.5, delta 95.4 +- 1.4, phys 96.4 +- 1.3 %; paired | delta (and phys) KEEP over abs; delta over phys (phys drops dr / dpsi from the features, TIE inside the band); abs deficit partly slower convergence (2000 steps: 81 / 76 %), 1 seed per fold | 2026-10-31T20:12 | 12 | 47b55c4137 |
| t10.dual_s0 | ok | the frozen DualBEV BEV and the frozen BEVFusion BEV together in one trunk keep the maps of t8_bevfeat AND the objects of t8_ctl | cold VAL 1000: lane divider@0.5 0.34, drivable 0.80, walkway 0.33, front mAP2 0.33 (0.25-0.38), ped AP@2 0.15 | E1 both (lane div, drivable >= bevfeat-0.02, walkway >= bevfeat-0.054, mAP2 >= ctl-0.054) -> seed twin; E2 maps kept, mAP2 0.22-0.31 -> branch-specific object head; E3 objects back, maps <= ctl+0.05 -> trunk ignores branch 2; E4 both mid -> 6000 steps (experiments/t10_dual_features/experiment.md) |  |  | VAL1000 front mAP2 0.361 (ctl 0.367, bevfeat 0.183/0.180); lane div 0.335, drivable 0.806, walkway 0.340, ped AP@2 0.199 | E1: maps of bevfeat + objects of ctl kept; seed twin t10.dual_s1 next | 2026-10-31T21:06 | 115 | 40648bd153 |
