# E261 Recall weight 6 (rc2) re-qualified on the reviewed vertex basis and at its own operating point

The card (read by `experiment.py` and `decide.py`; schema `autoresearch/schemas/experiment.schema.json`):

```json card
{
 "id": "E261",
 "card": "H22.E261",
 "title": "recall weight 6 (rc2, e18_D7_rc2) at its own operating point (tau, fork pruning) on the per-piece DP basis, against recall and against rc1's own threshold frontier",
 "kind": "variant",
 "deterministic": true,
 "metric": "lanes_found",
 "direction": "higher",
 "band": 0.01,
 "prediction": {
  "value": 0.022,
  "vs": "c"
 },
 "variant": "a",
 "control": "c",
 "arms": {
  "a": {
   "recipe": "rc2 at the point chosen on all 22 VAL scenes by temp/loop22/e261_cv.py (expected tau 0.8 / fork pruning 0.4); registry entry recall6 (run e18_D7_rc2, M 288 / K0 144, cache raw_val873_e18_D7_rc2_288.pkl through the loop-18 fallback); evaluate.py recall6 --compare recall",
   "base": "e18_D7_rc2",
   "run_ids": [
    "E261.g",
    "E261.cv",
    "E261.ev"
   ],
   "names": [
    "recall6"
   ]
  },
  "c": {
   "recipe": "recall (rc1, tau 0.7 / fork pruning 0.4), re-scored with tonight's code (E265.c, layer 1)",
   "base": "e123_D7_r3_rc1",
   "cards": [
    "runs/eval/recall/val_l22.json"
   ]
  },
  "t": {
   "recipe": "threshold-only control: rc1 at its best admissible point of the grid tau {0.05, 0.1, 0.2, ..., 0.9} x fork pruning 0.4 / 0.5 / 0.6 (registry entry recall_thr); evaluate.py recall_thr --compare recall",
   "base": "e123_D7_r3_rc1",
   "run_ids": [
    "E261.ev"
   ],
   "names": [
    "recall_thr"
   ]
  },
  "s": {
   "recipe": "seed spread of the reference: recall_s1 (rc1s1) at the registered point",
   "cards": [
    "runs/eval/recall_s1/val_l22.json"
   ]
  },
  "n": {
   "recipe": "equal-budget null f1eq (weight 1) and range-weighted rc4: report lines of e261_cv.py at tau 0.7 / 0.4 (no registry entry)"
  }
 },
 "rule": [
  {
   "metric": "lanes_found",
   "min_effect": 0.015,
   "note": "22-scene card (in-sample choice: label VAL, selection-CV, layer 3) AND the mean over the 20 held-out halves of e261_cv.py (lead check) must both reach +0.015 with the card's interval excluding 0"
  },
  {
   "metric": "R@1_far",
   "min_effect": 0.01,
   "note": "card and held-out mean"
  },
  {
   "metric": "Q",
   "max_loss": 0.005,
   "note": "and Q of a >= 0.735 absolute (card and held-out mean)"
  },
  {
   "metric": "topo_F1",
   "max_loss": 0.01
  },
  {
   "metric": "dup_frac",
   "max_loss": 0.03,
   "note": "rise <= 0.030 (the step accepted when recall replaced champion). Lead checks outside decide.py: false length of a <= 0.062; per-piece DP of a <= 14.3 on all frames and > 14.3 in at most 5 of the 20 held-out halves (referee); lanes found a - t >= +0.010 on the held-out mean (threshold-only control; decide.py cannot interval-judge a clause against t, tools item S6); CUT if a - t < +0.010 or rc2 has no admissible point; TIE if a - c in (-0.010, +0.015) or a guard fails by less than its band"
  }
 ],
 "budget_gpu_h": 0.0,
 "kill": "rung E261.g: the best admissible rc1 point (tau down to 0.05, pruning 0.4 / 0.5 / 0.6; admissible = Q >= 0.735, DP <= 14.3, FL <= 0.062, dup <= 0.1063) reaches lanes found >= 0.7802 (rc2's best admissible in-sample 0.7902 minus 0.010): CUT without the CV, E262 cancelled; layer 1 fails (rc1 0.7 / 0.4 does not give Q 0.7404, lanes found 0.7673, DP 12.82)",
 "evaluate": {
  "split": "val",
  "compare": "c"
 }
}
```


One file per experiment; it grows from card (P2) to detail plan (P3 start, block experiments only) to summary (P3 end). Machine-readable part: `experiment.json` next to this file (not written by the card author).

## 1. Card (written in P2, checked by ar-referee)
| | |
|---|---|
| lens | past critic |
| group | A3 sparse lane graph (gap 1: coverage at range), A9 evaluation (vertex-count basis, operating points) |
| sources | E219.rc1 / rc2 / rc4 / f1eq / rc1s1 (ledger), E233 (range-binned recall), E236 (matched-budget rerank), E246 (range thresholds killed), E257 section 5 (review 2026-10-31: per-piece DP basis, `vertex_basis.txt`), METRIC_RESPONSE.md (S / Q blind to parallel duplicates), closed question "larger recall weights" (2026-10-25); K184, K190 |
| status | scheduled (loop 22 plan: rung E261.g on the CPU now; referee fix applied in section 2) |

- **Statement** (falsifiable, one sentence): The existing, never-registered recall-weight-6 checkpoint `e18_D7_rc2`, at an operating point (tau, fork pruning) chosen out-of-fold on scene halves under tonight's constraints (Q >= 0.735, per-piece DP vertices <= 14.3 per 100 m, false length <= 0.062, duplicates at most +0.030 over `recall`), raises lanes found by >= +0.015 and R@1 far by >= +0.010 over `recall` (rc1 at its registered point) on the held-out halves, and by >= +0.010 lanes found over the best admissible threshold / pruning point of rc1 itself (the threshold-only control).
- **Why this is a past-critic card and not a re-opened question**: rc2 was rejected on 2026-10-25 (E219.rc2) by the rule "Q >= 0.745 AND lanes found >= 0.78", 0.010 stricter in Q than tonight's target (Q >= 0.735), at fork pruning 0.4 (the value tuned for rc1, K190), with vertices read as node ids (16.2, "over the budget"), on one seed. Since then (a) the review of E257 fixed the budget basis as per-piece Douglas-Peucker 0.3 m (the oracle's own count); (b) the brief sets Q >= 0.735 as a guard, not a gain. The closed question's own answer is "the weight buys lanes found, no Q gain": this card asks for no Q gain and trains nothing. If rc2 brings nothing that a lower tau / other pruning on rc1 cannot bring, the card closes the coverage side of the weight question as well (third answer).
- **Layer-1 / layer-3 numbers already measured by the card author** (scratch re-score through `evaluate.py`'s assembly path: `PI.assemble` with the registry assembly settings on the cached `~/exp_gcar_b123/p18/raw_val873_<run>_288.pkl`, `sparse_metrics.card_any(extra=True)` and `vert100_dp(.., 0.3)`; VAL 873; in-sample; nothing written to `runs/eval/`; script `frontier_scratch.py` and raw lines `frontier_numbers.txt` next to this file; the rc1 line at 0.7 / 0.4 equals the stored `runs/eval/recall/val.json` to 4 decimals):

| model, tau / fork pruning | Q | lanes found | R@1 far | P@1 far | topo F1 | false length | dup | vert100 node ids | per-piece DP |
|---|---|---|---|---|---|---|---|---|---|
| `recall` rc1, 0.7 / 0.4 (registered; reproduces the stored card) | 0.7404 | 0.7673 | 0.7714 | 0.7675 | 0.7423 | 0.0508 | 0.0763 | 15.53 | 12.82 |
| rc1, 0.3 / 0.4 (best admissible rc1 point by lanes found) | 0.7368 | 0.7757 | 0.7799 | 0.7559 | 0.7406 | 0.0559 | 0.0814 | 15.76 | 13.30 |
| rc1, 0.4 / 0.4 | 0.7379 | 0.7743 | 0.7784 | 0.7588 | 0.7413 | 0.0547 | 0.0804 | 15.71 | 13.18 |
| rc1, 0.5 / 0.4 | 0.7389 | 0.7726 | 0.7764 | 0.7611 | 0.7419 | 0.0536 | 0.0791 | 15.65 | 13.06 |
| rc1, 0.9 / 0.4 | 0.7422 | 0.7468 | 0.7508 | 0.7768 | 0.7423 | 0.0458 | 0.0697 | 15.39 | 12.25 |
| rc1, 0.7 / 0.5 | 0.7420 | 0.7673 | 0.7715 | 0.7676 | 0.7431 | 0.0507 | 0.0989 | 15.85 | 12.21 |
| rc1, 0.8 / 0.5 | 0.7426 | 0.7631 | 0.7672 | 0.7718 | 0.7437 | 0.0488 | 0.0964 | 15.77 | 12.07 |
| rc4 (range-weighted), 0.7 / 0.4 | 0.7360 | 0.7700 | 0.7777 | 0.7594 | 0.7420 | 0.0539 | 0.0804 | 15.43 | 12.88 |
| `recall_s1` rc1s1, 0.7 / 0.4 | 0.7376 | 0.7695 | 0.7707 | 0.7650 | 0.7412 | 0.0520 | 0.0771 | 15.51 | 12.94 |
| f1eq (weight 1, equal budget), 0.7 / 0.4 | 0.7367 | 0.7069 | 0.7238 | 0.7869 | 0.7438 | 0.0420 | 0.0558 | 15.10 | 11.23 |
| rc2, 0.5 / 0.4 | 0.7326 | 0.7956 | 0.7927 | 0.7378 | 0.7342 | 0.0629 | 0.0900 | 16.35 | 14.57 |
| rc2, 0.7 / 0.4 (the E219 point) | 0.7351 | 0.7929 | 0.7900 | 0.7440 | 0.7353 | 0.0599 | 0.0884 | 16.24 | 14.35 |
| **rc2, 0.8 / 0.4** | **0.7367** | **0.7902** | **0.7867** | 0.7493 | 0.7361 | 0.0577 | 0.0873 | 16.15 | **14.19** |
| rc2, 0.9 / 0.4 | 0.7387 | 0.7805 | 0.7768 | 0.7572 | 0.7375 | 0.0541 | 0.0832 | 15.98 | 13.73 |
| rc2, 0.7 / 0.5 | 0.7408 | 0.7929 | 0.7900 | 0.7441 | 0.7364 | 0.0599 | 0.1127 | 16.62 | 13.66 |
| rc2, 0.8 / 0.5 | 0.7420 | 0.7901 | 0.7867 | 0.7493 | 0.7371 | 0.0576 | 0.1111 | 16.52 | 13.51 |
| control E94 (stored graphs) | 0.7371 | 0.7999 | 0.8111 | 0.7576 | 0.7632 | 0.0622 | 0.1292 | 21.79 | 11.55 |

  Reading: (1) rc1's tau knob saturates: from tau 0.9 to 0.3 its lanes found move 0.747 -> 0.776 (R@1 far 0.751 -> 0.780) and never reach rc2's 0.790 - 0.796; (2) at exactly matched Q (rc1 tau 0.3: 0.7368, rc2 tau 0.8: 0.7367) rc2 has lanes found +0.0145, R@1 far +0.0068, topo F1 -0.0045, per-piece DP +0.9, false length +0.002, dup +0.006; at matched Q 0.739 (rc1 tau 0.5 vs rc2 tau 0.9) the edge is +0.008 / +0.000: in the high-Q region the weight is mostly a shift; its value is coverage in the Q 0.735 - 0.737 region that rc1 cannot buy with its threshold; (3) fork pruning 0.5 lifts Q on both models (rc2 +0.006, rc1 +0.002) but the pruned branches stay as unlinked pieces (`prune_forks`: geometry unchanged): false length and lanes found are unchanged, duplicates rise by +0.023 - 0.024, a gain the Q metric cannot see (METRIC_RESPONSE blind spot), so the duplicate guard below excludes it unless the referee decides otherwise; (4) the control is below the budget on the reviewed basis (11.55 per-piece DP against 21.8 node ids): its coverage is not bought with vertices but with false length (0.062) and duplicates (0.129); (5) rc4 (range weighting) sits on rc1's frontier (lanes found 0.770 at Q 0.736), nothing beyond.
- **Mechanism**: The recall weight lifts the trained validity of far nodes (E233: R@1 beyond 40 m 0.771 -> 0.790 from weight 3 to 6); rc1's own threshold cannot do the same because its far nodes below tau are few (the knob saturates at 0.773 lanes found, E246: far misses "not a threshold problem at the node level"). The precision rc2 loses (P@1 far 0.744 against 0.768, false length +0.009) is partly bought back by a stricter tau (0.8): about a third of the Q loss returns for a tenth of the coverage gain. The fork pruning tuned for rc1 (0.4) was never re-tuned for rc2, which produces more forks.
- **Prediction** (VAL 873, held-out halves of 20 seeded 11 / 11 scene splits, against `recall` on the same held-out frames): rc2 at its chosen point (expected tau 0.8 / pruning 0.4 in most splits): lanes found +0.022, R@1 far +0.015, Q -0.004 (absolute 0.737), topo F1 -0.006, per-piece DP 14.2, false length +0.007, dup +0.011; against the best admissible rc1 point (expected tau 0.3 / 0.4): lanes found +0.013, R@1 far +0.006 (in-sample +0.0145 / +0.0068; the out-of-fold edge shrinks). Probability of PASS about 0.6: the first clause is safe in-sample (+0.023), the threshold-control clause is the risk.
- **Decision rule** (deterministic: one checkpoint per arm, no training seed; paired scene bootstrap over the 22 VAL scenes; bands: Q 0.010 from the brief, lanes found and R@1 far 0.010 = 2.5 x their seed SD 0.004 / 0.003 from E240; tool `decide.py --experiment E261` with `deterministic: true`): **PASS** (rc2 is the prototype's coverage candidate, E262 starts) if, on the held-out halves (mean over the 20 splits) and on the 22-scene bootstrap of the point chosen on all scenes, lanes found a - c >= +0.015 with the interval excluding 0, R@1 far a - c >= +0.010, Q of a >= 0.735 and a - c >= -0.005, topo F1 a - c >= -0.010, per-piece DP of a <= 14.3, false length of a <= 0.062, dup a - c <= +0.030 (the step the project accepted when it adopted `recall` over `champion`: 0.076 vs 0.047), and lanes found >= +0.010 over the best admissible rc1 point. **TIE** if lanes found a - c lies in (-0.010, +0.015) or a guard fails by less than its band: `recall` stays, rc2 is reported as an equivalent. **CUT** (and a third entry to the closed recall-weight question: "the coverage of weight 6 is a threshold shift of weight 3") if lanes found over the best admissible rc1 point < +0.010, or rc2 has no admissible point.
- **Test design**:
  - Arms: a = `e18_D7_rc2` (weight 6, 4 u fine-tune of f1); c = `recall` (rc1, weight 3, same init and 4 u, registered point tau 0.7 / pruning 0.4); c' = `recall_s1` (seed spread of the reference: Q -0.003, lanes found +0.002); t = rc1 on the same grid (the threshold-only control, strong simple baseline); n = f1eq (weight 1, equal-budget null); r = rc4 (range-weighted, report only); x = control E94 (reported with its per-piece DP count; not in the rule).
  - Step 0 (layer 0 / 1): reproduce the stored `recall` card to 4 decimals (done: table row 1); assert `vert100_dp(gt, 0.3)` of the VAL ground truth (13.2, sparse_lanes README).
  - Step 1 (grid, CPU): a and t on tau {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9} x fork pruning {0.4, 0.5, 0.6}, frame records stored so that scene halves are scored without re-assembly.
  - Step 2 (out-of-fold): 20 random 11 / 11 splits of the 22 scenes (seeds 0 - 19, fixed before step 1 is read); on half A choose for a and for t the admissible point (all guards) with the highest lanes found; read on half B; report mean and spread. The point chosen on all 22 scenes becomes the registry entry `recall6` (in-sample, labelled so).
  - Controls: null f1eq; matched reference rc1 (identical frames, assembly code, settings except tau / pruning); seed twin of the reference rc1s1; equal budget by construction; threshold-only control t; sanity: rc2 at 0.7 / 0.4 reproduces the table.
  - Fidelity rung: layer 3 on cached outputs; the seeded layer-4 decision is E262.
- **Cost** and **kill criterion**: 0 GPU; CPU about 60 min at 2 threads (42 grid points x 2 models, about 40 s each on an idle box, 4 - 5 min each at load 13 as measured tonight); coding: one throw-away script in `temp/loop22/` (from `evaluate.graphs_of`, `bootstrap.frame_records`, `sparse_metrics.vert100_dp`); registry entries `recall6` / `recall6_dp` (not protected) after a PASS. Kill: step 0 does not reproduce; no admissible rc2 point in the grid; or the in-sample best admissible rc2 point has lanes found < best admissible rc1 point + 0.010 (out-of-fold cannot pass).
- **Speed class**: rc2 has rc1's network (288 / 144 with time recursion; 20 - 24 ms stored, E253 / loop-18 report); the assembly handles about 0.6 more node ids per 100 m (+0.2 - 0.4 ms expected, relative only until an idle session).
- **Differentiable path**: unchanged; tau and fork pruning are post-hoc settings of the existing assembly (fork pruning is a rule step of the delivered graph, the named gap, as for `recall`); nothing is added to training.
- **Inputs**: ready: raw caches `raw_val873_{e18_D7_rc2, e123_D7_r3_rc1, e18_D7_rc1s1, e18_D7_f1eq, e18_D7_rc4}_288.pkl` in `~/exp_gcar_b123/p18/`; cards in `runs/e123_ladder/cards/` (f1eq has none: its design and cfg equal rc1's); `evaluate.graphs_of`, `vert100_dp`, `bootstrap.paired`. To build: the grid / split script; registry entries after PASS.
- **Referee (2026-11-01, ar-referee)**: REVISE (minor; CPU only; G2-ready after fix 1). Falsifiable with metric, split and numbers: yes. Evidence checks out: rc2 at 0.7 / 0.4 reproduces the E219.rc2 ledger row (Q 0.735, lanes found 0.793, 16.2 node ids); the raw outputs are stored before assembly (`p16_infer.predict`), so re-scoring tau / pruning on the cache is exact. Closed question: no new mechanism, but admissible as a re-decision of an existing checkpoint under a changed frame (D28: Q as a guard; E257 section 5: per-piece DP 0.3 m basis) that trains nothing and claims only the coverage the closed answer already conceded; a PASS amends the closed row, it does not re-open the weight question. Fix 1 (needed): the threshold-only control t is censored at tau 0.3 (rc1 there is still admissible, Q 0.7368, DP 13.30): extend t's grid to tau 0.1 / 0.2 and stop only where a guard fails, otherwise the clause "rc2 >= best admissible rc1 + 0.010" is biased towards rc2. Notes: the 20-split design is a real cross-validation of a two-knob choice, but grid and guards were fixed after the full-VAL frontier was seen and no fresh frames exist (no TEST): label every number "VAL, selection-CV, layer 3"; the fresh evidence is E262's seed. DP 14.19 sits 0.11 under the budget: report the per-half DP and treat the point as inadmissible if it exceeds 14.3 in more than 5 of 20 held-out halves. Name the rc2 cache in the result (loop-18 fallback `raw_val873_e18_D7_rc2_288.pkl`, keyed without K0 by `builder_raw`). rc2 at its point leaves no vertex headroom: it cannot be stacked with E265 / E266 without a joint re-selection.

## 2. Detail plan (ar-planner, loop 22, 2026-11-01 ~01:30; machine-readable part `experiment.json`)
Restated: the existing rc2 checkpoint (`e18_D7_rc2`, recall weight 6), at an operating point (tau, fork pruning) chosen out-of-fold under tonight's guards, gives >= +0.015 lanes found and >= +0.010 R@1 far over `recall` on held-out scene halves AND >= +0.010 lanes found over rc1's own best admissible threshold / pruning point. Every number is labelled **"VAL, selection-CV, layer 3"** (grid and guards were fixed after the full-VAL frontier was seen; no fresh frames; the fresh evidence is E262's seed). The rc2 raw outputs are the loop-18 cache `~/exp_gcar_b123/p18/raw_val873_e18_D7_rc2_288.pkl` (reached by `builder_raw`'s fallback without K0); the result names it.

What would make it fail: rc1 at tau 0.05 - 0.2 reaches rc2's coverage (the weight is a threshold shift: the referee's censoring point); the chosen rc2 point sits on the 14.3 DP edge and fails it on many halves; the out-of-fold edge over t shrinks below +0.010.

Referee fix applied: t's grid goes down to tau 0.1 and, since it may still be admissible there, to 0.05 (the grid stops only where the guards fail). Notes applied: label above; per-half DP reported and the point counted inadmissible when its DP exceeds 14.3 in more than 5 of the 20 held-out halves; no stacking with S / R in the prototype (PLAN.md candidate W).

Admissible (all guards, fixed now): Q >= 0.735, per-piece DP 0.3 m <= 14.3, false length <= 0.062, dup <= dup of `recall` on the same frames + 0.030 (all frames: 0.0763 + 0.030 = 0.1063).
Grid (fixed now): tau {0.05, 0.1, 0.2, ..., 0.9} x fork pruning {0.4, 0.5, 0.6}, both models (60 points). Splits (fixed now): seeds 0 - 19; half A = `numpy.random.default_rng(seed).choice(sorted unique VAL scene ids, 11, replace=False)`, half B = the other 11.

| step | what exactly | where (file / function) | done when | budget |
|---|---|---|---|---|
| 0 commit | S5: commit this folder (`frontier_scratch.py`) by path | git | in HEAD | 2 min |
| 1 rung (cheap kill) | E261.g: `frontier_scratch.py e123_D7_r3_rc1:288:e123_D7_r3_rc1 0.7,0.2,0.1,0.05 fork_prune=P` for P = 0.4, 0.5, 0.6 (0.7 / 0.4 and 0.7 / 0.5 are the layer-1 lines: Q 0.7404 / 0.7420, lanes found 0.7673, DP 12.82 / 12.21). Kill reading with `frontier_numbers.txt`: if the best admissible rc1 point over all measured points reaches lanes found >= 0.7802 (rc2's best admissible in-sample 0.7902 at 0.8 / 0.4, minus 0.010): **CUT** (third answer to the closed recall-weight question), E262 cancelled, no CV | this folder | 12 lines; kill read and noted in the ledger | CPU 15 min |
| 2 code (throw-away, lead) | `temp/loop22/e261_cv.py`: for each model (rc2, rc1; f1eq and rc4 only at 0.7 / 0.4 as report lines) and grid point: `PI.assemble` with the asm of `recall` and the point's tau / fork_prune (cfg as `frontier_scratch.cfg_of`), per frame `bootstrap.frame_records`, `topo_metrics.frame_record` and the DP pair (vertices, length) of `bootstrap._vstats(g, False, 0.3)`; records pickled to `~/exp_gcar_b123/loop22/e261_records.pkl` (data, not git); layer 1 (exit 1 otherwise): pooled over all frames, rc1 0.7 / 0.4 gives Q 0.7404, lanes found 0.7673, R@1 far 0.7714, DP 12.82 and rc2 0.8 / 0.4 gives 0.7367 / 0.7902 / 0.7867 / 14.19; then the 20 splits: on A the admissible point with the highest lanes found for a (rc2) and for t (rc1) (ties: higher Q, then higher tau), read on B: a - c, a - t and t - c for lanes found, R@1 far, Q, topo F1, false length, dup, and DP of a; print per split, mean, SD, min, the count of halves with DP(a) > 14.3, the histogram of chosen points; and the choice on all 22 scenes (the registry points of `recall6` and `recall_thr`) | `temp/loop22/` (pooled metrics: `bootstrap.pooled(rec, idx)`, `topo_metrics.pooled`) | layer-1 asserts pass | lead, 30 min |
| 3 CV | E261.cv: `bash tools/env/_py14.sh temp/loop22/e261_cv.py` | as above | printout in the run log | CPU 60 - 70 min |
| 4 registry | `recall6` (run e18_D7_rc2, set M 288 / K0 144, tau and fork_prune of the all-22 choice, other asm of recall) and `recall_thr` (rc1 at its all-22 choice) | `models/registry.json` (lead; not protected entries) | `evaluate.py --list` shows them | 5 min |
| 5 cards | E261.ev: `evaluate.py recall6 --compare recall` and `evaluate.py recall_thr --compare recall` | evaluate.py | 2 cards | CPU 10 min |
| decide | PASS / TIE / CUT by the card's rule: `decide.py --experiment E261 --write` (22-scene card clauses) AND the held-out means of step 3 (same thresholds) AND the lead checks of `experiment.json` (FL, DP absolute and per half, a - t >= +0.010 held out); PASS starts E262 (GPU, 06:00) and makes `recall6` candidate W of PLAN.md | `decide.py` | verdict in the ledger rows E261.cv / E261.ev | ~04:30 |

Prediction of the rung (written before it runs): rc1 at 0.2 / 0.4 lanes found about 0.778 with Q about 0.735 (admissible or just not); at 0.1 and 0.05 Q < 0.735 at pruning 0.4, and at 0.5 / 0.6 the dup guard fails or binds; best admissible rc1 about 0.777 - 0.779, so the rung passes by about +0.012 (probability of a kill about 0.35).

## 3. Summary (lead, loop 22, 2026-11-01 04:40)
- **Runs**: E261.g (rung), E261.cv (20 scene-half splits, `e261_cv.json`, run log), E261.ev (cards `runs/eval/recall6/val.json`, `runs/eval/recall_thr/val.json`); code tags `lcr/E261.*`. **Decision: CUT** (`decide.py --experiment E261 --write`, `decision.json`); the card's TIE window (a - c in (-0.010, +0.015)) would give the same consequence: `recall` (rc1) stays, rc2 is not adopted, E262 is cancelled, candidate W is out.
- **Setup**: no training, no code in the product; rc2 (`e18_D7_rc2`, recall weight 6) and rc1 re-assembled on cached raw outputs over a grid of tau x fork pruning; the operating point chosen on one half of the 22 VAL scenes and read on the other half, 20 times; controls: `recall` (rc1 0.7 / 0.4, c), rc1's own best threshold (t), seed spread (`recall_s1`), equal-weight null f1eq and range-weighted rc4 (report lines).
- **Numbers** (VAL 873, selection-CV; the 22-scene cards are in-sample choices):

| | lanes found | R@1 far | Q | topo F1 | dup | DP / 100 m |
|---|---|---|---|---|---|---|
| `recall` (c, card `val_l22`) | 0.7673 | 0.7714 | 0.7404 | 0.7423 | 0.0763 | 12.82 |
| `recall6` (a, rc2 0.9 / 0.5) - c | +0.0133 | +0.0054 | +0.0023 | -0.0037 | +0.0295 | 13.08 |
| `recall_thr` (t, rc1 0.2 / 0.4) - c | +0.0105 | +0.0096 | -0.0049 | -0.0027 | +0.0068 | 13.48 |
| held-out mean a - c (13 of 20 splits with an admissible rc2 point) | +0.029 | +0.021 | -0.0075 | -0.0085 | +0.016 | > 14.3 in 10 of 20 halves (rc2 inadmissible on half A in 7) |
| held-out mean a - t | +0.0157 | +0.0075 | +0.0019 | -0.0031 | -0.0045 | |

- **What it says**: weight 6 does find more lanes at range, but the points that do so on held-out halves spend more than the vertex budget (10 of 20 halves; 7 had no admissible point) and lose Q (-0.0075); the point that respects the budget on all 22 scenes is only +0.013 lanes found and +0.005 R@1 far, under the bar, and at the edge of the duplicate guard. rc1's own threshold buys most of the same coverage (+0.0105) at a Q cost. Prediction +0.022, measured +0.013 (in-sample). Negative kept: the recall weight is a coverage lever only at a vertex and Q price (as loop 19 found for weight 3).
- **Weaknesses / next**: one seed of rc2; 7 of 20 halves had no admissible rc2 point at all (the guards are tight for rc2); `decide.py` has no clause for a TIE window (tools note). Next: nothing on this branch; gap 1 continues through the assembly (E265 KEEP) and the seed ring (E266 / E267).
