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
