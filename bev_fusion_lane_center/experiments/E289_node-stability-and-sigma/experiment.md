# E289 Unstable points: shift stability of the node read-out, then a position spread per node (D82, thread: "die Punkte sind instabil")

Card of 2026-11-11 (a plan card: the diagnostic runs first, the training arms only if it says so), written by the lane chat on Mirko's remark that the points are unstable and on the literature scan of the day (`scan_2026-11-11_pruning_uncertainty_time.md`, ideas 3 and 4: beta-NLL position spread with an uncertainty-weighted read-out; the shift-stability diagnostic of anti-aliased CNNs / DISK). E285 measured that the lines are stable frame to frame (E283 section 10: 93.6 % re-found within 1 m) but the vertex phase along the line was not measured: a max-pool peak on a 2 m grid aliases, so a node may jump by a cell when the scene moves by half a metre.

## 0. The card (what the tools run)
```json card
{
 "id": "E289",
 "card": "",
 "title": "node jitter of B under a half-metre shift of the input (diagnostic), and, if it is large, a position spread sigma per node (beta-NLL) with a 1/sigma^2-weighted read-out against the plain argmax position",
 "kind": "variant",
 "deterministic": false,
 "metric": "jitter_rate",
 "direction": "lower",
 "band": 0.02,
 "prediction": {"value": 0.15, "vs": "c"},
 "arms": {
  "c": {"recipe": "diagnostic: the delivered nodes of B (close:1.5@0.5) on the cached features and on the same features shifted by 0.5 m in x, in y and by one 2 m cell phase (a roll of the 200 x 200 input by 1, 1, 4 cells, the positions shifted back); a node is stable when a node lies within 1.0 m of it in the other output; jitter_rate = 1 - stable share, mean over the three shifts, VAL 873 frames"},
  "s": {"recipe": "only when jitter_rate of c is above 0.10: the offset head fine-tuned with beta-NLL (sigma per node, beta 0.5) on frozen B, plus a shift-consistency loss (the node map of the shifted input equals the shifted node map); read-out position = the 1/sigma^2-weighted mean of the node and its neighbours' offset votes"}
 },
 "control": "c",
 "variant": "s",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "jitter_rate", "vs": "c", "min_effect": 0.02},
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01}
 ],
 "budget_gpu_h": 1.5,
 "kill": "the diagnostic with no shift reproduces the stored nodes of B exactly (zero jitter): if not, the harness is not deterministic and nothing is read",
 "evaluate": {"split": "val"}
}
```
By hand (a diagnostic first; label "by hand: diagnostic"): step 1 is the number of arm c, no verdict. If jitter_rate of c is below 0.10 the arm s is not run and the finding is "the points are stable under shift; the instability seen is between frames (E290)". If above, arm s is trained (a refinement of B's node head, B untouched: a side stage like E284) and judged by the rule: KEEP when jitter_rate falls by more than 0.02 on both seeds with the guards held, TIE inside: kept and refined (D82), CUT when it rises or a guard breaks. Reported: the median displacement of the matched nodes minus the shift, the ENCE and the Spearman correlation of sigma with the node error (the calibration of the spread), the share of nodes whose vertex phase along the line changes.

## 1. What a shift tells and what it does not
The cached features are a 0.5 m raster of the BEV network's output; rolling the raster by one cell is a shift of the scene by 0.5 m as far as the head can see, but the features were made by a network that is only approximately shift-equivariant, so the test measures the head's own dependence on the grid phase, not the whole system's. A node on the 2 m grid that jumps cells moves by about 2 m; the stored delivered graph then has a vertex on the other side of the line. Real motion between two keyframes (0.5 s) is E290.

## 2. Design
Arm c: GPU forward of B (a few minutes) on 873 frames x 3 shifts, CPU matching; no training. Arm s: only on the diagnostic's word; one new head module (sigma, 0.1 M parameters) trained on frozen B for 2000 steps, two seeds. TEST never.
**Prediction (a number):** jitter_rate of c about 0.15 (one in seven delivered nodes has no partner within 1 m after a half-metre shift); arm s -0.04 at best.

## 3. Command
`temp/e289/shift_probe.py` (scratch; promoted into `evaluation/` when it is used a second time) through `run.py --no-card` for the diagnostic (it feeds no verdict by itself: the verdict, if any, is of arm s). Result `runs/e289/shift_probe_s<S>.json`.
