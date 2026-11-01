# Lane graph - morning summary, loop 22 (night of 2026-10-31 to 2026-11-01, D28)

Goal of the night (D28): a ready-to-use prototype with good performance, then gap 1 (coverage at range). Numbers from metric cards (`runs/eval/<name>/<split>.json`) and run records (`experiment_execution/runs/<id>/run.log`); VAL = 873 frames of 22 scenes, TEST = 833 frames of 21 scenes.

## The prototype: `recall_sp0_dp`
rc1 checkpoint (recall weight 3) + fork pruning 0.4 + **spurs kept** (new tonight, E265) + output stage Douglas-Peucker 0.2 m. One command, CPU, cached network outputs:

```
wsl bash tools/env/_py14.sh lane_prototype.py recall_sp0_dp                          # whole VAL split, reproduces runs/eval/recall_sp0_dp/val.json
wsl bash tools/env/_py14.sh lane_prototype.py recall_sp0_dp --scenes 23 7 --video     # two scenes + results video
```

Output: `runs/prototype/recall_sp0_dp/scenes/<scene>.json` (per frame the lane centre lines in the ego frame, x forward / y left, in travel direction, with vertex ids, confidence and successor edges; format in `tools/prototype/README.md`), `summary.md`; video `runs/prototype/recall_sp0_dp_video/recall_sp0_dp_val.mp4` (poster `recall_sp0_dp_val_poster.png`).

| system | split | Q | lanes found | R@1 far | topo F1 | vertices / 100 m | ms per frame |
|---|---|---|---|---|---|---|---|
| `recall_dp` (packaged yesterday evening) | VAL | 0.740 | 0.767 | 0.771 | 0.742 | 10.4 | lane graph ~21, full chain ~27 (stored timing of rc1) |
| **`recall_sp0_dp` (prototype)** | VAL | **0.742** | **0.781** | **0.780** | 0.741 | 10.9 | same timing (one assembly step less; not re-measured) |
| **`recall_sp0_dp` (prototype)** | **TEST** | **0.727** | **0.740** | **0.749** | 0.705 | 12.6 | |
| control E94 | TEST | 0.712 | 0.762 | 0.776 | 0.722 | 22.0 | |

TEST (the one TEST use, logged first, report only): against the control Q +0.015 [+0.000, +0.029], lanes found -0.021 [-0.037, -0.005], R@1 far -0.028 [-0.042, -0.012], duplicates -0.041, 43 % fewer vertices. Against the same checkpoint's loop-18 TEST number (E230.rc, lanes found 0.724, Q 0.727) lanes found +0.016 at the same Q: the VAL gain transfers (prediction 0.735 - 0.74 hit). Gap 1 on TEST: -0.038 -> -0.021 lanes found.

Rule steps still in the delivered graph (E2E gap): path union, clean-up (bubble half), junction anchoring / relocation, fork pruning, output DP. Not checked: route success on spur-kept graphs; ms per frame re-measured.

## The experiments of the night (each decided by its written rule)
| id | question | verdict | key numbers (VAL 873) |
|---|---|---|---|
| E265 | does the clean-up's spur rule delete true far lanes? | **KEEP** | lanes found +0.0115 (4 replication checkpoints, every interval > 0), R@1 far +0.0065, Q +0.0039; per-piece DP 12.8 - 13.8 (budget 14.3); above the equal-budget tau frontier |
| E261 | recall weight 6 (rc2) at its own operating point | **CUT** | in-sample +0.0133 lanes found (< +0.015); held out: over the vertex budget in 10 / 13 halves, Q -0.0075 |
| E262 | rc2 seed twin | cancelled | gated by E261 |
| E264 | union of two seeds' hypothesis trees | **not adopted** | lanes found +0.047, R@1 far +0.040 (replicated on both pairs; +0.038 over the single model at equal false length), but false length 0.0635 > 0.062 and no idle timing |
| E266 | seed ring at +-47.5 m (inference only) | **KEEP** (one gate deviation) | R@1 far +0.0152 mean over 4 checkpoints (4 / 4), Q -0.0016; gate G0 bar missed narrowly (seed share 1.88 x instead of 2 x), flag checked |
| E267 | ring + spurs kept stacked | **CUT** | lanes found +0.022, R@1 far +0.027 (0.771 -> 0.798) on both seeds, but false length +0.006 and P@1 far -0.015 miss their guards |

Prediction hits: E265 +0.012 vs +0.0115; E264 +0.045 vs +0.047; E266 +0.012 vs +0.015; E268 TEST 0.735 - 0.74 vs 0.740; misses: E261 +0.022 vs +0.013, E267 +0.017 vs +0.022.

## What it means, and next
- Gap 1 is mostly lost in the chain, not in the network: the spur rule (E265) and the seed lattice at the box edge (E266) each gave back coverage without training. The strongest systems at range (the stack E267: R@1 far 0.798; the two-seed union E264: lanes found 0.786) are blocked by precision at range (false length, P@1 far), not by coverage.
- Next loop: a precision step at range (range-dependent threshold for edge seeds, or the union with a wider vertex merge: on val250 vtx_r 1.9 keeps lanes found 0.783 at false length 0.060), with idle timing; route success on spur-kept graphs before any champion change.
- Champion / control entries unchanged; promotion stays Mirko's (D4).

Records: `experiment_execution/PLAN.md` (loop 22), `LEDGER.md` (rows E261 - E268), experiment files `experiments/E261_*` - `E268_*` (section 3 each), `evaluation/TEST_USAGE_LOG.md`.
