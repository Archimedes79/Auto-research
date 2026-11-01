# E246 Conformal risk control of range-binned admission thresholds (zero training)

The card (read by `experiment.py` and `decide.py`; schema `autoresearch/schemas/experiment.schema.json`):

```json card
{
 "id": "E246",
 "card": "H19.F2-2",
 "title": "conformal risk control of range-binned node admission thresholds on a scene-level false-length loss, leave-one-scene-out over the 22 VAL scenes",
 "kind": "block",
 "deterministic": true,
 "metric": "R@1_far",
 "direction": "higher",
 "band": 0.01,
 "prediction": {
  "value": 0.02,
  "vs": "c"
 },
 "variant": "a",
 "arms": {
  "a": {
   "recipe": "analysis script temp/loop19/e246_crc.py: leave-one-scene-out CRC fit of tau_near / tau_mid / tau_far on fast_recall raw nodes (grid 0.3 ... 0.9), assemble with tau_bins and fork_prune 0.4, evaluate.py override; budget alpha_bin = false length per 100 m GT of the hand-set operating point in the bin",
   "base": "e18_D7_j1rc",
   "planned_names": [
    "fast_recall_crc"
   ]
  },
  "b": {
   "recipe": "the same rule on champion (replication on a second checkpoint)",
   "base": "e123_D7_r3_f1",
   "planned_names": [
    "champion_crc"
   ]
  },
  "c": {
   "recipe": "hand-set tau 0.7, fork_prune 0.4 (registry fast_recall)",
   "names": [
    "fast_recall"
   ]
  }
 },
 "control": "c",
 "rule": [
  {
   "metric": "R@1_far",
   "min_effect": 0.015
  },
  {
   "metric": "false_len_frac",
   "max_loss": 0.003
  },
  {
   "metric": "Q",
   "max_loss": 0.005
  },
  {
   "metric": "topo_F1",
   "max_loss": 0.005
  },
  {
   "metric": "R@1_near",
   "max_loss": 0.01
  }
 ],
 "budget_gpu_h": 0.05,
 "kill": "in-sample ceiling below +0.015 R@1_far; far-bin loss curves non-monotone by more than 0.1 B after repair; tau_far spread over the leave-one-scene-out fits above 0.15; realised risk above alpha_far by more than 2 standard errors",
 "evaluate": {
  "split": "val",
  "compare": "c"
 }
}
```


One file per experiment; it grows from card (P2) to detail plan (P3 start, block experiments only) to summary (P3 end). Machine-readable part: `experiment.json` next to this file.

## 0. What the tools run (the fields of `experiment.json`)
| field | value |
|---|---|
| id / card / title | E246 / H19.F2-2 / Per-range-bin node admission thresholds (near < 20 m, mid 20 - 40 m, far >= 40 m) chosen by conformal risk control on a scene-level false-length loss, cross-fitted over the 22 VAL scenes |
| kind | block (small: an option `tau_bins` in `assemble` of `training/pipeline/p16_infer.py`, default off, graphs identical to today; plus an analysis script in `temp/loop19/`) |
| deterministic | true (arms share the checkpoint; no training seed; the paired scene bootstrap of `evaluate.py --compare` decides) |
| metric, direction, band | R@1_far, higher, 0.010 (also Q, F1@1, false_len_frac, vert100, topo_F1, lanes_found, ms) |
| prediction | R@1_far +0.020 against arm c on `fast_recall` (0.727 -> 0.747) at equal false_len_frac; Q within +-0.005 |
| arms | a: out-of-fold thresholds (leave-one-scene-out fit of the three bin thresholds by CRC), registry `fast_recall`; b: the same rule on `champion` (replication on a second checkpoint); c: hand-set tau 0.7 with fork pruning 0.4 (`fast_recall`), same frames |
| control | c |
| rule | R@1_far min_effect 0.015 vs c; false_len_frac max_loss 0.003 vs c; Q max_loss 0.005 vs c; topo_F1 max_loss 0.005 vs c; R@1_near max_loss 0.01 vs c |
| budget_gpu_h, kill | 0.05 (only if a raw cache must be regenerated; CPU about 1 h), see section 1 |
| evaluate | split val (873, 22 scenes), compare c |

## 1. Card (written in P2, checked by ar-referee)
| | |
|---|---|
| lens | foundations (F2 calibration with a guarantee, F3 risk view of the threshold) |
| group | A8 uncertainty / calibration, A3 (gap 1 coverage at range, by redistribution of the false-length budget over range) |
| sources | deep/2208.02814_conformal-risk-control.md (sections 5.1 - 5.3, 8), E217 (per-level threshold 0.707 lanes found vs 0.695), E221 (validity bimodal, ECE 0.121), E233, closed question "larger recall weights" (this card changes admission, not a loss weight); new K id K194 |
| status | proposed |

- **Statement** (falsifiable): Choosing the admission threshold of the node validity separately for the near, mid and far range bins, each by the conformal risk control rule on the monotone loss "false predicted length in the bin per 100 m of ground-truth lane in the bin" with the budget of each bin set to the false length that the hand-set threshold (tau 0.7, fork pruning 0.4) has in that bin, and fitting the three thresholds leave-one-scene-out on the 22 VAL scenes, raises R@1 beyond 40 m of `fast_recall` by at least 0.015 on the held-out scenes at the same false length (+0.003 at most), with Q, topo F1 and near recall inside their guards.
- **Mechanism** (why it should work): One global tau treats a far node and a near node alike, but their reliability differs: on `fast_recall` P@1 is 0.95 near and 0.75 far, R@1 0.94 near and 0.73 far. If the false length that the system tolerates is spent where it buys most recall, the near bins give a little (their nodes are almost all right, a slightly higher near tau removes the last false ones) and the far bin takes the budget. The conformal rule makes the choice a procedure with a statement instead of a hand-tuned number: for a loss L_i(tau) in [0, B] that is non-increasing in tau, the smallest tau with `(n / (n + 1)) mean_i L_i(tau) + B / (n + 1) <= alpha` has expected held-out loss <= alpha. This is new compared with E217 (a threshold that decays with the tree level, a proxy for range that gave lanes found +0.012) because it is stated per range bin with a budget; and compared with E219 (rc2, rc4, closed) because nothing is trained, only the admission of the nodes the model already produces. Honest limit (deep note section 9): it chooses among existing graphs, it cannot add a lane the tree does not contain (the valid nodes cover 85 % of far points, the admitted 74 % at tau 0.7: E217), so the ceiling is the 11 points between the two, and the 22 scenes make the shift term B / 23 = 0.043 B.
- **Prediction** (VAL 873 frames, 22 scenes, out-of-fold, against arm c on the same frames): R@1_far +0.020 (+0.012 to +0.03 plausible), R@1_near -0.004, Q +0.003, false_len_frac equal within 0.003, vert100 within +-0.3; held-out false-length risk per bin <= alpha_bin on average over 1000 random 11 / 11 scene splits (mean within one standard error), and the fitted tau_far differs between leave-one-scene-out fits by less than 0.15 (stability at n = 21). Replication on `champion` (arm b): R@1_far +0.015.
- **Decision rule** (band 0.010, deterministic, paired scene bootstrap): KEEP if the clauses of section 0 hold and the bootstrap interval of R@1_far excludes 0 on `fast_recall` and the sign agrees on `champion` (a replication, not a second claim). TIE if |dR@1_far| <= 0.010: keep the hand-set tau (cheaper: one number). The conformal fit itself is judged by the realised held-out risk: if the mean exceeds alpha_bin by more than one standard error in any bin the rule is reported as not controlling the risk at n = 22 (a methodological finding) whatever the recall says. INCONCLUSIVE: nothing more can be added (VAL has 22 scenes; TRAIN cannot calibrate, the heads were fitted on it; a new split is a decision for Mirko).
- **Test design**: Step 0 (layer 1, in-sample ceiling): the best (tau_near, tau_mid, tau_far) on the full VAL at the budget of the hand-set operating point; if its R@1_far gain is below +0.015 the out-of-fold version cannot reach the bar: kill. Step 1 (layer 0): numpy CRC rule with the toy of the deep note section 8 (scores 0.1 ... 0.9, alpha 0.2, B = 1 gives 0.8) and the Monte-Carlo coverage check (uniform scores, n = 50). Step 2: per-scene loss tables from the cached raw nodes (`raw_val873_*.pkl`) for a grid of 8 thresholds per bin (0.3 ... 0.9) with the numba assembly (about 13 ms per frame: 3 bins x 8 values x 873 frames, about 5 min CPU); monotonicity check of the scene-level curves (non-increasing in tau; Corollary 1 repair, violations reported). Step 3: leave-one-scene-out fit and out-of-fold graphs, cards through `evaluate.py` with the override `tau_bins`; the paired comparison against `fast_recall` at tau 0.7. Controls: null = hand-set global tau; the strong simple baseline = the best global tau on the same budget (a single threshold from the same rule with the total budget; if it already gives the gain, the range split is not the lever); sanity = scene-shuffled pairing of loss and ground truth (the rule must return the trivial threshold); frame-level instead of scene-level calibration (should under-cover on held-out scenes: the demonstration of the unit argument). alpha_bin and the grid are fixed on this card before the curves are looked at.
- **Cost** (GPU h / coding days / risk): 0 to 0.05 GPU h; coding 1 day (CRC rule 10 lines, the per-bin false-length loss in `evaluation/sparse_lanes/` next to `topo_metrics.py`, `tau_bins` option, registry override, split study); risk low technically, medium methodologically (22 units, the nodes of the three bins interact through the path union, so the per-bin loss is only approximately nested).
- **Kill criterion**: step 0 gain below +0.015; the scene-level loss curves of the far bin violate monotonicity by more than 0.1 B after the repair; tau_far spreads by more than 0.15 across the leave-one-scene-out fits; or the realised risk exceeds alpha_far by more than 2 standard errors.
- **Differentiable path**: nothing on the training path changes; the thresholds are post-hoc constants after training (not a rule step inside training); the graph builder and its losses are untouched; ms per frame unchanged (a vector compare instead of a scalar compare in the assembly).
- **Inputs**: ready: cached raw nodes of VAL for `champion`, `fast_recall` (paths in `runs/eval/*/val.json` info.cache), numba assembly, scene lists in `data/splits.json` (22 VAL scenes), the paired bootstrap. To build: `tau_bins` option, per-bin false-length metric, CRC rule + split study script in `temp/loop19/`, registry overrides. New technique **K194** "conformal risk control of range-binned node admission thresholds on a scene-level monotone false-length loss", source 2208.02814 (re-implement the rule, MIT repository only as a grid reference), to be added to `literature/register.md` before the first run.
- **Referee** (2026-10-25, ar-referee): ACCEPT, stage-gated, CPU only (runs next to any GPU job; tonight only step 0, then build the rest if it survives). Not a repeat: E217 was a level-decaying proxy for range (lanes found +0.012) and nothing was trained here, so the closed "recall weight" question is untouched; zero training cost, deterministic, with an in-sample ceiling test (+0.015 R@1_far) that kills the card cheaply, a scene-shuffled sanity, a frame-level versus scene-level demonstration, and an honest limit (it cannot add lanes the tree lacks: valid 85 % vs admitted 74 % of far points, so the ceiling is 11 points). Watch: (1) alpha per bin is taken from the hand-set operating point measured on all 22 scenes, which leaks the held-out scene into the budget; compute alpha on the training folds or accept and label it; (2) that hand-set tau 0.7 / pruning 0.4 was itself chosen on VAL, so the comparison is against a tuned reference: report against the best global tau on the same total budget (already planned) as the real control; (3) the three bins interact through the path union and pi = min over ancestors: a lower far tau only helps if ancestors pass, so monotonicity curves matter (planned); (4) the 1-day coding is mostly step 2-3; step 0 needs a per-node tau vector in the assembly (`tau_bins`, shared `p16_infer.py`, default off, golden check) or a scratch copy of `assemble`; (5) K194 into the register. Tonight: step 0 in about 2 h including the scratch assembly; if it reaches +0.015 the rest is a next-loop item.

## 2. Detail plan (block experiments only; written by ar-planner when the loop reaches the card)
What would make it fail: the far-bin loss curve is flat or not monotone (a higher tau does not remove false far length because of the path union); 22 scenes make the fitted far threshold unstable; the 11 points between valid and admitted far nodes are mostly false.

| step | what exactly | where (file / function) | done when | budget |
|---|---|---|---|---|
| ceiling | in-sample best (tau_n, tau_m, tau_f) at the hand-set budget | `temp/loop19/` script on the raw caches | gain number written to this file | 0.5 h CPU |
| code | `tau_bins`, per-bin false length, CRC rule, split study | `p16_infer.assemble`, `evaluation/sparse_lanes/`, `temp/loop19/` | toy tests pass; default graphs bit-identical (`evaluate.py --golden check`) | 1 day |
| evaluate | out-of-fold graphs, cards, `--compare fast_recall` | `evaluate.py` | metric cards | 0.5 h CPU |
| decide | the rule above | `decide.py --experiment E246 --write` | verdict | - |

## 3. Summary (2026-10-26)
- **Decision: KILLED at step 0** (`step0_ceiling.txt`, `temp/loop19/e246_step0.py`, cached raw nodes of `fast_recall` (j1rc), VAL 873, in-sample). Best range-binned thresholds (near / mid / far) = (0.7, 0.3, 0.3) at false length <= base + 0.003 and vert100 <= base + 0.3: R@1_far 0.7334 (+0.0059 against the base point, gate +0.015), Q 0.7382 (-0.0009), false_len +0.0025, vert100 +0.09. The in-sample ceiling is below the gate, so an out-of-fold version cannot reach it.
- **What it says**: per-range admission thresholds do not move far recall at a fixed false-length budget; the far misses are not a threshold problem at the node level.
- **Weaknesses**: one model (j1rc), a coordinate search on a coarse grid (0.3-0.9), in-sample on VAL (optimistic, so a fortiori negative).
- **Next**: nothing; K194 stays in the register as "killed at step 0".
