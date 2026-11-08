# E280 The close-node step with a learned radius per pair of nodes (E279 with a context-dependent radius; baseline B, D55 / D58)

Card of 2026-11-08, written by the lane chat at Mirko's word ("Yes, write the card"), revised the same day after a fresh critic pass (`ar-critic`, mode cards: REVISE, three blocking points, all checked against the sources and taken, section 8). Status: **CLOSED on 2026-11-08 by its own gate, before any code or GPU hour: the CPU pre-check (section 3) failed both conditions on both seeds, result in section 3a. Nothing coded, no GPU job queued.** GPU only in Mirko's windows. No TEST. Nothing pushed.

**Why this card.** The soft step of E277 / E279 damps a node by every stronger node within ONE fixed radius. The radius is a trade-off: 1.5 m gives junction F1 0.263 / 0.268 but lanes found 0.604 / 0.619 (E277, CUT by the guards on seed 0), 1.25 m gives lanes found 0.618 / 0.632 and junction F1 0.223 / 0.218 (E279, a weak KEEP). One constant for the whole map cannot know where a close pair is two real nodes that the lane graph needs and where it is a redundant node. This card lets the network choose the radius of each pair from the context.

**What is known about the step, and what is NOT (E275 section 7; corrected after the critic).** The gain of the step is mostly a density reduction, not the removal of duplicates: the duplicate share `dup_frac` does not move (0.232 -> 0.234, 0.192 -> 0.201), 65 % of the dropped nodes are counted true by the scorer, 93 - 95 % of the true nodes lost have another GT node within 1.5 m, the dropped nodes lie mostly on straight stretches (20.5 % / 19.6 % in the intersection mask against a base rate of 22.6 %), and the junction F1 of the hard step rises with the radius (seed 0: 0.196 / 0.225 / 0.271 / 0.280 at 1.0 / 1.25 / 1.5 / 1.75 m; lanes found 0.644 -> 0.627 at 1.5 m). Two readings fit this and the card does not know which holds: **H1** (the card's hope) a part of the close real pairs matters for lanes found and can be told from the nodes that are thinned for the junction gain, so a gate keeps the recall and most of the junction gain; **H0** the junction gain IS the thinning of real but redundant nodes, and the recall it costs is the price of the very same nodes, so no gate can have both. The pre-check of section 3 decides between them as a necessary condition (an oracle gate on B's stored predictions); only if it passes is code written. The prediction is set low for that reason.

**D55 status.** The gate is part of the forward path: differentiable, trained by the losses that see the damped existence, no rule step added. Outside the network stay only what E277 / E279 already named: the threshold e >= 0.5 and the order inside the damping (who is stronger: a comparison without gradient). The gate is a learned LOCAL RADIUS (a radius per pair inside a fixed band), not a free classifier of pairs: the structure of the step is kept, its one constant becomes a function.

## 0. The card (what the tools run)
```json card
{
 "id": "E280",
 "card": "",
 "title": "the lane head trained with a learned radius per pair of neighbouring nodes in the soft close-node step (start radius 1.25 m = the E279 step, band 0.65 - 1.85 m, smoothness 0.1 m, 3 passes), read out at e >= 0.5 without a hard step, against the E279 head (fixed 1.25 m), a fixed 1.375 m head, the E277 head (fixed 1.5 m) and the baseline B",
 "kind": "block",
 "deterministic": false,
 "metric": "junction_F1",
 "direction": "higher",
 "band": 0.03,
 "prediction": {"value": 0.015, "vs": "c"},
 "arms": {
  "c": {"recipe": "matched reference = the E279 runs e279_w_s0 / e279_w_s1 (fixed soft step 1.25 m, same phase-2 start, same seeds, same recipe otherwise); scoring cards runs/e279/<run>.plain_t0.5/card.json (no new training)"},
  "b": {"recipe": "baseline B = E273 runs e273_p_s0 / e273_p_s1 read out as before; cards runs/e273/<run>/card.json (no new training); the guards of E279 are measured against it"},
  "e": {"recipe": "the E277 runs (fixed soft step 1.5 m): cards runs/e277/e277_u_s<S>.plain_t0.5/card.json; reported, not judged"},
  "v": {"recipe": "from the E273 phase-2 checkpoints phases 3 4 5 6 7 as E273 / E279 with `--pair-gate 1.25 0.6 0.1 3` (new): radius of a pair R_ij = 1.25 + 0.6 * tanh(g(pair features)), i.e. inside [0.65, 1.85] m; g is a small MLP whose last layer starts at zero, so the model starts as the E279 step (checked by a test); scored at e >= 0.5 with no hard step (runs/e280/<run>.plain_t0.5/card.json)"},
  "t": {"recipe": "twin of c, seed 0 only: the E279 recipe again from the same phase-2 checkpoint with a different training seed (another data order), E280.t_s0: it measures the paired noise of a run with the same start; run FIRST"},
  "r": {"recipe": "fixed soft step at 1.375 m (the middle between E279 and E277, `--soft-nms 1.375 0.1 3`), both seeds, run only if v is a KEEP-candidate: it is the control that a learned radius has to beat, because a fixed radius between the two known points would give a trade point on the same line"},
  "s": {"recipe": "one learned scalar radius for all pairs (`--pair-gate-kind scalar`), start 1.25 m, both seeds, optional, only if v is KEEP: does the gain come from context or only from a better constant? reported, not judged"},
  "a": {"recipe": "control of the assignment, scored inside every card"}
 },
 "control": "c",
 "variant": "v",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "junction_F1", "vs": "c", "min_effect": 0.03},
  {"metric": "lanes_found", "vs": "c", "max_loss": 0.015},
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "lane_acc_err", "vs": "a", "min_effect": 0.03},
  {"metric": "lane_acc_exact", "vs": "a", "max_loss": 0.01}
 ],
 "budget_gpu_h": 12,
 "kill": "a loss that is not finite; or the gate of phase 3 (node precision within 2 m below 0.75 or node recall within 2 m below 0.5) failing: the run stops (exit 3); or an arm that needs more than 1.5 x the E279 time (about 80 min for phases 3-7); or the start check failing (at initialisation the model must equal the E279 step); or the pre-check of section 3 not passing (no code, no GPU job at all)",
 "evaluate": {"split": "val"}
}
```
The JSON holds the clauses that the decision tools can judge; the guards against B and the order of the verdicts are in the rule below (by hand, as in E277 / E279).

**Verdict rule (by hand, written now, before any code or run; to be implemented as `qgraph_run_eval.py verdict280` with a test per outcome).** VAL 873 frames, final checkpoint, in-sample, no TEST. Per seed: dJ, dL, dT, dF = v minus c in junction F1, lanes found, topo F1, F1@1.0 (v and c are the runs of the same seed). Tolerances: junction band 0.03; lanes found 0.015, topo F1 0.01, F1@1.0 0.01 are FLOORS: if the twin t differs from c on a metric by more than half the floor, the tolerance of that metric (both seeds) is twice that difference (declared now, so that it is not chosen after seeing v). Order of the verdicts, the first that applies wins:
1. **No verdict (invalid run):** a kill criterion fired or the start check failed; the ledger row says which.
2. **CUT:** an assignment guard is broken (`lane_acc_exact` v - g below -0.01 on a seed, or `lane_acc_err` v - g below +0.03 on the mean of the seeds), or dJ is below -0.03 on a seed, or dL is below -0.03 on a seed.
3. **KEEP-candidate:** dJ above +0.03 on BOTH seeds, and dL, dT, dF not below minus their tolerances on both seeds, and the guards of E279 against B hold on both seeds (F1@1.0 not below B - 0.01, lanes found not below B - 0.03; E279 itself holds them by only 0.0003 and 0.004 on seed 0, so a miss here is a coin toss of the noise). A KEEP-candidate runs arm r; it becomes **KEEP (the learned radius replaces the fixed 1.25 m step)** only if v dominates r on both seeds: (junction F1 of v at least r - 0.01 and lanes found of v at least r + 0.01) or (junction F1 of v at least r + 0.02 and lanes found of v at least r - 0.005). Otherwise the verdict is **TIE with r** (a fixed 1.375 m step does as well; no gate needed; a new fixed radius is Mirko's decision).
4. **TIE (E279 stays: simpler, no extra parameters):** dJ inside +-0.03 on both seeds and no loss beyond a tolerance on a seed; the guards against B are not asked (a variant equal to E279 would trip them half of the time by noise).
5. **Otherwise INCONCLUSIVE** (goes to Mirko): a junction gain bought with a loss beyond a tolerance (a different operating point, as E277 was), a KEEP-candidate that fails only a guard against B, or seeds on different sides (one above the band, the other inside).
- **Reported, not judged:** E277 and arm r as reference points; node recall / precision, vertices per 100 m after Douglas-Peucker, `dup_frac`, long-link precision; the lane signature (`gt.json`) of v against E279's; arm s.
- **Mechanism check (reported):** the learned radius per pair type (section 3: two real nodes / a thinned node), median and 10 / 90 %, per seed and per phase; a KEEP with medians less than 0.2 m apart is reported as "gain of unknown origin"; pairs on a bound (within 0.05 m of 0.65 or 1.85 m) are counted, and more than 95 % of the pairs on a bound is reported as a saturated gate.
- **Cost check (reported):** ms per frame of the model, batch 1, idle GPU, cached features, against E279's: the gate has about 450 parameters and runs on 8 x 2500 pairs per frame; the design is dropped if the whole head gets more than 10 % slower.

**Prediction (a number):** the junction F1 gain over c is +0.015 on both seeds (range -0.01 to +0.04), lanes found at the level of E279 (0.618 / 0.632), topo F1 0.618 - 0.625, vertices per 100 m 20 - 23, so the most likely verdict is **TIE**; my estimate of a KEEP-candidate is about one in four, of KEEP about one in six. Why it can miss in either direction: H0 above (no gain possible); the loss may use the gate only as an extra knob to lower all nodes near others; the gradient to the gate is lopsided (a damped real node carries a large focal-BCE term and pulls the radius down; a damped redundant node with small e' gives almost none), so R may shrink everywhere and the model drift toward the plain B read-out; the paired noise is unmeasured (E279's two trainings were resumed from checkpoints after host memory errors, so c itself is an interrupted run with a changed data order).

## 1. Where we are (VAL 873, per seed, in-sample; from the cards of E273 / E275 / E277 / E279)
| seed 0 / seed 1 | B | B + hard step 1.5 m | soft 1.5 m (E277) | soft 1.25 m (E279 = c) |
|---|---|---|---|---|
| topo F1 | 0.597 / 0.595 | 0.626 / 0.625 | 0.621 / 0.624 | 0.618 / 0.620 |
| junction F1 | 0.130 / 0.120 | 0.271 / 0.249 | 0.263 / 0.268 | 0.223 / 0.218 |
| junction P / R (seed 0) | 0.072 / 0.655 | 0.179 / 0.556 | 0.176 / 0.515 | 0.141 / 0.544 |
| F1@1.0 | 0.832 / 0.821 | 0.828 / 0.816 | 0.818 / 0.820 | 0.823 / 0.822 |
| lanes found | 0.644 / 0.630 | 0.627 / 0.614 | 0.604 / 0.619 | 0.618 / 0.632 |
| vertices per 100 m after Douglas-Peucker (budget 14.3) | 28.9 / 31.5 | 19.3 / 20.1 | 19.1 / 18.8 | 21.4 / 20.9 |
| node recall / precision | 0.833 / 0.873, 0.833 / 0.868 | 0.793 / 0.915, 0.793 / 0.910 | 0.787 / 0.916, 0.790 / 0.911 | 0.805 / 0.907, 0.804 / 0.901 |
Hard step on B at 1.25 m (E275 radius curve, seed 0 / seed 1): topo F1 0.624 / 0.622, junction F1 0.225 / 0.211, lanes found 0.635 / 0.622, F1@1.0 0.830 / 0.819. E279 is a weak KEEP (margins on seed 0: topo gain +0.0210 against 0.02, F1@1.0 -0.0097 against -0.010, lanes found -0.026 against -0.030). Mirko's decision whether E279 replaces the hard read-out is open (no D-id); E280 is compared with E279 either way. Seed-to-seed spread of lanes found of one recipe: 0.013 - 0.015 (B, hard step, E277, E279), up to 0.028 (E276 arm t); of F1@1.0 about 0.011 - 0.012 (B, hard step).

## 2. Design
- **The step today** (`qnodes.soft_suppress_logit`): e'_i = e_i * prod over the 8 neighbour cells j that are stronger than i of (1 - e'_j * k_ij), k_ij = sigmoid((R - d_ij) / tau), d_ij the distance of the predicted node positions, R one constant (1.25 / 1.5 m), 3 passes, only the 3 x 3 neighbourhood.
- **The change.** R becomes R_ij = 1.25 + 0.6 * tanh(g(f_ij)) in [0.65, 1.85] m; g is a small MLP (10 -> 16 -> 16 -> 1, about 450 parameters, last layer initialised to zero, run outside autocast), so the model starts as the E279 step exactly and the difference v - c at the start is only the gate and the changed random-number stream. The 3 x 3 argument of the old function (two cells apart are at least 2 m apart) holds only for R clearly below 2 m: with R <= 1.85 and tau 0.1 a pair two cells apart at d = 2.0 m still gets k <= 0.18 and is neglected; the test measures the error against a 5 x 5 reference at R = 1.85. Everything else of the step is unchanged (order by e, passes, tau, the clamp-free logit formula).
- **Pair features f_ij** (i = the weaker cell that is damped, j = the stronger neighbour; from the node head's own outputs, detached so that the gate is a separate small reader and adds no new path into the head; the gradient into the head still flows through e, e_j and the positions as in E277 / E279): the distance d_ij / 2 m; the displacement p_j - p_i along and across the direction of node i (unit(dir_raw), two numbers, / 2 m); the cosine of the two node directions; the existence probabilities e_i and e_j; the split and merge flag probabilities of i and of j (`qtargets.FLAGS` order, split = 2, merge = 3; four numbers). Ten numbers, no picture features. Note that e_i is the head's own prediction of "this cell has a GT node": the pre-check therefore compares the ten features with distance AND e_i, e_j, not with distance alone (section 3).
- **What trains it.** No new labels. The gate is a submodule of the node head and trains wherever the node head trains: **phases 3, 5 and 7** (phase 5, `PHASE_TRAIN[5] = ("nodes_s",)`, fits the whole node head at lr 5e-4 with the node loss on; phases 4 and 6 freeze it). The loss on the damped existence (`n_e`) acts in phases 3, 5, 7; in phase 7 the gate also receives the gradient of the link and assignment losses, because the damped existence feeds the link token (`qlinks.py`) and the assignment (`qgraph.py`, `e_act`, not detached when the nodes train). The setting and the gate weights travel with the checkpoint (`args['pair_gate']`, `qpredict.load_model`) like `soft_nms`; `QGraph.node_stage` stays the one place that applies the step (forward path, gates, timing).
- **Not changed:** E277 / E279 code stays as it is (their checkpoints must reproduce); the new function is a copy with a per-pair radius, tested equal to the old one when the gate outputs zero.
- **Alternative not in this card:** a one-to-one matching loss (each GT node claimable by one predicted node, DETR-style) that trains the head to deliver the thin row itself; it stays on the list (E277 section 6) and would be a separate card with its own rule. If H0 holds, it is the only route that does not put the thinning into a radius.

## 3. Step 1: the CPU pre-check (P11), before any code for the head
Input: the stored predictions of B on the 873 VAL frames, `temp/e276/predfull_e273_p_s0.pkl` and `..._s1.pkl` (per frame 512 node slots with `idx` (cell), `e`, `pos`, `dirv`, `flags`, links, sections; no forward pass needed), and the node targets of GT l4 per cell (`qtargets`, as the E276 analyses used them). Script in `temp/e280/` (scratch; if a card or record names it as how a number was made it is promoted with a test).
- **Pairs and types.** Neighbour cells (3 x 3) with the stronger cell j, e of both at least 0.3. Type A "two real nodes": both cells carry a GT node. Type B "thinned": only the stronger cell carries one. Type C "wrong order": only the weaker cell carries one (reported and counted, left out of the AUC: they exist in training and pull the gate toward small radii). Neither carries one: left out. Reported: the number of pairs per type and per distance bin (below 1.0 / 1.0 - 1.25 / 1.25 - 1.5 / 1.5 - 1.85 m), the share of type A pairs that carry a split / merge flag or lie in the intersection mask.
- **Test 1 (signal).** AUC for type A against type B inside the decision band d in [1.0, 1.85] m of three gradient-boosting models: on d alone; on d, e_i, e_j; on all ten features; two folds by scene (fit on one half of the scenes, score on the other), per seed.
- **Test 2 (oracle, a necessary condition).** The hard greedy step at R = 1.5 m on B's dump with one exception: a type-A pair is not damped (both nodes kept). Scored by the E273 scorer on CPU like the earlier read-out analyses (`exp_link_threshold.py` re-reads the links of a full dump). An oracle is a sanity check, not an upper bound, and B's head was not trained with the step (E277 moved its node recall from 0.833 to 0.787), so passing is necessary, not sufficient.
- **Gate: code is written only if BOTH hold on both seeds:** (1) AUC(all ten) minus AUC(d, e_i, e_j) is at least 0.05 in both folds; (2) the oracle read-out reaches junction F1 at least the hard step at 1.25 m plus 0.03 (0.255 / 0.241) with lanes found at least that step's minus 0.005 (0.630 / 0.617). Otherwise the card is closed as "no signal beyond the head's own existence probability" or "H0: the recall cost and the junction gain are the same nodes" (a CUT before any GPU hour) and the next candidate is the matching loss. In-sample VAL, a design check, not a verdict. Cost: CPU only, about 2 h.

## 3a. Result of the pre-check (run 2026-11-08 16:06 - 16:18, CPU, both seeds in parallel; gate written before the run: **both conditions failed on both seeds, the card is closed**)
Command: `wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 bash tools/env/_py14.sh temp/e280/p11_precheck.py <seed>` (scratch script, kept in the git tag `lcr/e280-scratch-1`, not promoted: used once). Input: the stored predictions of B (`predfull_e273_p_s0/s1.pkl`, 873 VAL frames) and the lane GT of the l4 store (signature 0702a065c6). Reproduction gate passed exactly: the plain, hard 1.25 m and hard 1.5 m read-outs of the script equal the stored E275 cards (difference 0.0000 in topo F1, junction F1, F1@1.0, lanes found). A subset run on 150 frames showed the same picture before the full run. Result files (scratch): `temp/e280/p11_s0.json`, `p11_s1.json`.

**Pairs in the decision band d in [1.0, 1.85) m** (seed 0 / seed 1): A two real nodes 18 813 / 19 380, B one thinned node 23 976 / 24 196 (so A is 44.0 % / 44.5 % of the A + B pairs: the close pairs are about half real). Per distance bin, A and B: 1.0 - 1.25 m 2 127 / 2 190 and 4 119 / 4 249; 1.25 - 1.5 m 4 163 / 4 078 and 5 827 / 6 040; 1.5 - 1.85 m 12 523 / 13 112 and 14 030 / 13 907. Only 6.5 % / 5.8 % of the A pairs have a GT split or merge node, and the head's own split / merge probability does not tell A from B (A 12.1 % / 10.7 % against B 11.2 % / 10.1 % of the pairs).

**Test 1 (can the pair be told apart?), AUC of A against B in the band, two folds by scene (fold 1 / fold 2), gradient boosting:**
| features | seed 0 | seed 1 |
|---|---|---|
| distance d alone | 0.565 / 0.548 | 0.585 / 0.553 |
| d, e_i, e_j (distance and the head's own existence probabilities) | 0.717 / 0.716 | 0.730 / 0.728 |
| all ten features | 0.719 / 0.727 | 0.731 / 0.736 |
| gain of the other seven features | +0.003 / +0.011 | +0.001 / +0.008 |
The gate asked for at least +0.05 in both folds on both seeds: **failed by a factor of 4 or more** (the largest gain is +0.011). Almost all of the little signal there is sits in e_i and e_j; geometry, direction and flags add nothing, and the distance alone is nearly useless (AUC 0.55 - 0.59).

**Test 2 (oracle: the hard step at 1.5 m that spares every A pair), VAL 873, seed 0 / seed 1:**
| read-out | nodes delivered | junction F1 | junction P | lanes found | topo F1 |
|---|---|---|---|---|---|
| plain (B) | 150 305 / 151 265 | 0.130 / 0.120 | 0.072 / 0.067 | 0.644 / 0.630 | 0.597 / 0.595 |
| hard 1.25 m | 139 797 / 140 374 | 0.225 / 0.211 | 0.140 / 0.132 | 0.635 / 0.622 | 0.624 / 0.622 |
| hard 1.5 m | 136 631 / 137 287 | 0.271 / 0.249 | 0.179 / 0.166 | 0.627 / 0.614 | 0.626 / 0.625 |
| **oracle: hard 1.5 m, A pairs spared** | 140 889 / 141 563 | **0.193 / 0.178** | 0.115 / 0.107 | 0.633 / 0.620 | 0.618 / 0.616 |
The gate asked for a junction F1 of at least 0.255 / 0.241 (hard 1.25 m plus 0.03) with lanes found at least 0.630 / 0.617: the lanes condition holds (0.633 / 0.620), **the junction condition fails clearly (0.193 / 0.178, it is even below the plain hard step at 1.25 m by 0.03)**.

**Reading (H0 holds as far as this check reaches).** Against the hard step at 1.5 m (the oracle's own radius), sparing the real close pairs gives back about 4 300 of the 13 700 / 14 000 nodes it removes (31 %), and that costs 0.079 / 0.071 of junction F1 (0.271 -> 0.193, 0.249 -> 0.178; junction precision 0.179 -> 0.115, 0.166 -> 0.107) while lanes found rises by only 0.007 / 0.006 (0.627 -> 0.633, 0.614 -> 0.620). **The oracle point is worse than the plain fixed radius of 1.25 m on both axes and on both seeds** (junction F1 0.193 / 0.178 against 0.225 / 0.211, lanes found 0.633 / 0.620 against 0.635 / 0.622): knowing exactly which close pairs are real does not help. The real close pairs are where the false junctions come from, and they are only a small part of where the recall goes; the rest of the lanes-found cost of the step is carried by other nodes (types B, C, N), which this check did not examine. The pairs cannot be told apart by anything the head knows beyond e_i, e_j. The card's hope H1 is not supported.
**What this check does not show:** B's head was not trained with the step (the oracle is a necessary condition, as the card said), only R = 1.5 m was tried for the oracle, the labels are cell-occupancy types of the GT and not what a trained gate would be shown, all numbers are in-sample VAL. None of that changes the decision: the gate was written as a necessary condition and it failed by a wide margin; a failed necessary condition closes the card. Decision: **CUT before the GPU, by the written gate (not "by hand")**. The next candidate in the section 2 note is the one-to-one matching loss (a separate card with its own pre-check).

## 4. Controls and checks
- c, b, e, a from existing cards; the paired comparison is per seed (same seed number, same phase-2 start as E279); the twin t gives the paired noise of the start; r gives the fixed-radius point between E279 and E277.
- Layer 0 (code tests, `training/grid/test_pair_gate.py`, new): at initialisation the new function equals `soft_suppress_logit` at radius 1.25 on random frames (to 1e-6) and the model with the gate equals the E279 model with the same weights; the gate's gradient is non-zero in phases 3, 5, 7 and zero in 4 and 6; a hand-set gate (large positive output for parallel same-direction pairs, negative for diverging pairs) damps the first and keeps the second; the radius stays inside [0.65, 1.85] for any output; the 5 x 5 reference bound at R = 1.85; features equal for a frame alone and inside a batch, and on the borders; the gate runs outside autocast; the checkpoint round trip (`load_model`) and an old checkpoint stays plain; `qgraph.py` selftest: every phase finite with a gradient into the trainable groups; `test_verdict280.py`: the written rule on fake cards, one test per outcome (invalid, CUT, KEEP-candidate, KEEP, TIE with r, TIE, INCONCLUSIVE) including the null result (TIE), the trade point (INCONCLUSIVE), seeds on different sides, a b-guard miss of a KEEP-candidate, and the raised tolerances.
- Layer 1: verdict277 reproduces the E279 verdict (KEEP) from the E279 cards before verdict280 is used.
- Layer 2: the null is c; the mechanism controls are t (noise), r (fixed intermediate radius), s (constant instead of context).
- Records: the scoring dump gets the radius per damped pair (needed for the mechanism check; the E277 / E279 dumps have none); the trainer logs the radius statistics at the end of every phase; the run records name the lane signature (`gt.json`: the l4 store 0702a065c6, which E277 / E279 did not write) and the check against E279's; `qgraph_timing.py` gets the gate.
- After training: the 20 frames with the largest difference between the E279 and the E280 node rows, drawn next to each other, and the radius maps of the gate on them, are looked at before the summary.
- A critic pass (`ar-critic`, mode code) on the diff before the first GPU job.

## 5. Steps, gates, cost
| step | what | gate / done when | cost |
|---|---|---|---|
| 0 | this card; `experiment.py check E280`; critic pass on the card (done, REVISE, section 8) | card valid | CPU + agent |
| 1 | P11 pre-check (section 3) | both gate conditions on both seeds | CPU about 2 h |
| 2 | code: `PairGate`, `learned_suppress_logit` in `qnodes.py`, `QGraph(pair_gate=...)`, `qpredict.load_model`, `qgraph_train.py --pair-gate` (and `--pair-gate-kind scalar`), the radius export and logging, `qgraph_timing.py`, `verdict280`; tests of section 4; critic pass (mode code) on the diff | tests green, no blocking finding | CPU + agent, about half a day |
| 3 | GPU smoke: 20 steps per phase, no card, never a verdict | finite loss, radii logged | about 0.5 GPU h |
| 4 | twin t_s0 (E280.t_s0), then v for seeds 0 and 1 (E280.v_s0, E280.v_s1; two slots) | gate of phase 3, finite loss | twin about 1.4 GPU h, v about 1.4 GPU h per seed |
| 5 | score twin and v (plain@0.5 and close:1.5@0.5, dumps with the radii; ids E280.tq_s0, vq_s0, vq_s1) | cards written | about 0.2 GPU h each |
| 6 | `verdict280`, mechanism check, ms per frame (idle GPU), ledger notes | verdict written | CPU + 0.1 GPU h |
| 7 | arm r (E280.r_s0, r_s1 and scoring) only if v is a KEEP-candidate; arm s only if v is KEEP | | about 3.2 GPU h each |
Unconditional about 5.4 GPU h; with r about 8.6; with s about 11.8 (budget 12). Windows: the night of 2026-11-08 is planned for the BEV chat (`t23.lane_bce`) and the Image Encoder chat (t22 stage B2); the first lane window after steps 1 and 2 is a later one (Monday 08:30 - 17:30 at the earliest). The GT chat's "live checks done" for the l5 merge is waited for before a lane job (the lane defaults still give the l4 store, unchanged).

## 6. Risks and fallbacks
| risk | effect | what we do |
|---|---|---|
| H0: the junction gain is the thinning of real nodes and costs the same nodes | no gate helps | the oracle of section 3 stops it before the GPU; next candidate: the matching loss |
| node geometry and flags carry no signal beyond e_i, e_j | no gain over a fixed radius | test 1 of section 3; fallback: picture tokens in the gate (a second card) |
| the loss uses the gate only to lower e near others, or the lopsided gradient shrinks R everywhere | v drifts toward B's read-out or toward lower recall | the tolerances against c decide; the radius statistics per phase and the mechanism check show it |
| the gate saturates on a bound | no information | reported as a saturated gate; the band is the first thing to revisit |
| the paired noise is unmeasured and c is an interrupted run | a real change of 0.01 - 0.02 cannot be told from noise on two seeds | twin t measures it first; the tolerances are raised by the declared rule; INCONCLUSIVE is the honest outcome |
| a fixed radius between E279 and E277 reaches the same trade point | "better than a fixed radius" is not shown by KEEP-candidate alone | arm r decides KEEP |
| training instability (the gate adds a coupling between cells) | NaN / divergence | the trainer's guards; the gate of phase 3 stops the run; the last layer starts at zero |
| the in-sample label | all numbers are VAL; the radius band and the start were chosen from the VAL curves of E274 / E275 | stated in every table; a held-out check needs Mirko's recorded OK |

## 7. Summary
**CUT before the GPU by the written gate of section 3** (2026-11-08): on both seeds the extra pair features add at most +0.011 AUC over the head's own existence probabilities (needed +0.05), and an oracle that spares the real close pairs reaches junction F1 0.193 / 0.178 (needed 0.255 / 0.241), wins back only 0.007 / 0.006 of lanes found against the hard 1.5 m step and is worse than the fixed 1.25 m radius on both axes. The junction gain of the close-node step is the thinning of real, redundant nodes (H0); a learned radius per pair cannot separate it from the recall cost. Nothing coded, no GPU hour spent. Details and the table in section 3a. The 12 GPU h of the budget are free; the next candidate is the matching loss (own card).

## 8. Critic pass (`ar-critic`, mode cards, 2026-11-08; verdict REVISE; read-only; every finding below was checked by the lane chat against the source before it was taken)
Blocking, all taken: (1) the premise contradicted E275 (`dup_frac` unchanged, density reduction, junction F1 rising with the radius) and the pre-check could not test it: the card now states H0 / H1, compares the ten features with distance AND e_i, e_j in the decision band, makes the oracle read-out a gate, reads the stored dumps instead of a forward pass, and sets the prediction low; (2) the decision rule was at the noise level (E279 itself passes the guards against B by 0.0003 and 0.004 on seed 0, so a variant equal to E279 would be called CUT in many draws), its KEEP could be reached by a fixed radius between E279 and E277, it mislabelled the null result and had overlaps: the rule now has a written order, measures the paired noise with a twin first, treats the guards against B as a condition of KEEP only, adds arm r and maps the null result to TIE; (3) the gate trains in phases 3, 5 and 7, not only 3 and 7 (`PHASE_TRAIN[5] = ("nodes_s",)`), and in phase 7 also receives the link and assignment gradients: sections 2 and 4 corrected. Non-blocking, taken: the start radius 1.25 m (v then equals c at the start), the radius band [0.65, 1.85] with its 3 x 3 argument and a saturation epsilon, the gate outside autocast, the lane signature record, the missing tasks (radius export, per-phase logging, mechanism check, timing script), the budget sum, the B and E277 junction P / R cells, the wording of the interruptions, the topo prediction. Not taken: none. Could not be checked by the critic: the contents of the dump files (looked at since: 512 slots per frame with `idx`, `e`, `pos`, `dirv`, `flags`, links), the oracle and the pre-check themselves (not run).
