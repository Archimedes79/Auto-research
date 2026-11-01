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
| `recall_dp` (packaged yesterday evening) | VAL | 0.740 | 0.767 | 0.771 | 0.742 | 10.4 | estimates: lane graph ~21, full chain ~27 (stored rc1 session measured without the time recursion + the recursion cost measured on f1) |
| **`recall_sp0_dp` (prototype)** | VAL | **0.742** | **0.781** | **0.779** | 0.741 | 10.9 | not measured (one clean-up step skipped, but more vertices: DP 12.82 -> 13.70) |
| **`recall_sp0_dp` (prototype)** | **TEST** | **0.727** | **0.740** | **0.749** | 0.705 | 12.6 | |
| control E94 | TEST | 0.712 | 0.762 | 0.776 | 0.722 | 22.0 | |

TEST (the one TEST use, logged first, report only): against the control (paired, same frames) Q +0.015 [+0.000, +0.029] (the lower end only just excludes 0), lanes found -0.021 [-0.037, -0.005], R@1 far -0.028 [-0.042, -0.012], duplicates -0.041, 43 % fewer vertices (the prototype counted after its output-stage DP, the control without one). Against the same checkpoint's loop-18 TEST number (E230.rc, lanes found 0.724, Q 0.727; **another code version**: loop-18 pipeline, no output DP, so an unpaired reference, not a paired control) lanes found +0.016 at the same Q: consistent with the VAL gain (prediction 0.735 - 0.74 hit). With the same caveat, the lanes-found gap to the control on TEST goes from -0.038 (E230.rc) to -0.021.

Rule steps still in the delivered graph (E2E gap): path union, clean-up (bubble half), junction anchoring / relocation, fork pruning, output DP. Not checked: route success on spur-kept graphs; ms per frame re-measured.

## The experiments of the night (each decided by its written rule)
| id | question | verdict | key numbers (VAL 873) |
|---|---|---|---|
| E265 | does the clean-up's spur rule delete true far lanes? | **KEEP** | lanes found +0.0115 (4 replication checkpoints, every interval > 0), R@1 far +0.0065, Q +0.0039; per-piece DP 12.8 - 13.8 (budget 14.3). Equal budget: a lower tau at the same DP finds about the same lanes (+0.001 for spurs kept) at -0.009 Q, so spurs kept = equal coverage at matched budget with better Q; it passes the detail-plan rule only through its Q branch, while the card's kill line (lanes found alone within 0.006 -> CUT) would cut it |
| E261 | recall weight 6 (rc2) at its own operating point | **CUT** | in-sample +0.0133 lanes found (< +0.015); held out: no admissible rc2 point in 7 of 20 halves, over the vertex budget in 10 of 20, Q -0.0075 (mean over the 13 halves with a chosen point) |
| E262 | rc2 seed twin | cancelled | gated by E261 |
| E264 | union of two seeds' hypothesis trees | **not adopted** | lanes found +0.047, R@1 far +0.040 (replicated on both pairs); +0.037 over the single model at tau 0.10, the lowest tau of the sweep (false length 0.0611 vs the union's 0.0635: close, not equal); false length 0.0635 > 0.062 and no idle timing |
| E266 | seed ring at +-47.5 m (inference only) | **KEEP, with two rule deviations** | R@1 far +0.0152 mean over 4 checkpoints (4 / 4), Q -0.0016. Deviations: (1) gate G0 missed (seed share 0.1036 vs bar 0.110); the kill criterion says stop after one debug round; the lead overrode it after checking the flag acts (own cache, share 0.055 -> 0.104, ring R@1 0.702 -> 0.750); (2) the two-sided inner guard (+-0.005 per checkpoint) is exceeded on recall_s1 by a gain: R@1 far inner +0.0068 [+0.0039, +0.0097] (the ring changes the time-recursion prior too) |
| E267 | ring + spurs kept stacked | **CUT** | lanes found +0.022, R@1 far +0.027 (0.771 -> 0.798) on both seeds, but false length +0.006 and P@1 far -0.015 miss their guards |

Prediction hits: E265 +0.012 vs +0.0115; E264 +0.045 vs +0.047; E266 +0.012 vs +0.015; E268 TEST 0.735 - 0.74 vs 0.740; misses: E261 +0.022 vs +0.013, E267 +0.017 vs +0.022.

## What it means, and next
- Gap 1 is mostly lost in the chain, not in the network: the spur rule (E265: coverage equal to a lower threshold's at the same vertex budget, with better Q) and the seed lattice at the box edge (E266: R@1 far +0.015) each gave something back without training. The strongest systems at range (the stack E267: R@1 far 0.798; the two-seed union E264: lanes found 0.786) are blocked by precision at range (false length, P@1 far), not by coverage.
- Next loop: a precision step at range (range-dependent threshold for edge seeds, or the union with a wider vertex merge: on val250 vtx_r 1.9 keeps lanes found 0.783 at false length 0.060, val250 only; on 873 frames at vtx_r 1.5 it is 0.0635), with idle timing; route success on spur-kept graphs before any champion change.
- Champion / control entries unchanged; promotion stays Mirko's (D4).

Records: `experiment_execution/PLAN.md` (loop 22), `LEDGER.md` (rows E261 - E268), experiment files `experiments/E261_*` - `E268_*` (section 3 each), `evaluation/TEST_USAGE_LOG.md`.
