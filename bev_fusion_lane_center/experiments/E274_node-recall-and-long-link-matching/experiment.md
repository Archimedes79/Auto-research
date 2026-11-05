# E274 Node recall and long-link matching on the pyramid head: arm r (recall weights where the nodes are weak) and arm p (r + a dual-softmax loss over the long links), against the E273 checkpoints

Card of 2026-11-05, revised after the critic pass of the same day (blocking findings B1-B5 fixed before any run). E273 is CUT (topo F1 0.596 and F1@1.0 0.827 against the best graph so far d = recall_sp0_dp 0.7405 / 0.8781, guards -0.03; the assignment gain +0.071 holds). The E273 rule sends the work back to the nodes and links; the diagnostics below changed the scope of the link arm, so the new numbers get a card of their own. Priority: D57 (2026-11-05), Spurmitte work first. GPU only inside Mirko's windows (nights 22:00-07:00, Monday-Friday 08:30-17:30).

## 0. The card (what the tools run)
```json card
{
 "id": "E274",
 "card": "",
 "title": "recall weights at event cells and in intersections (arm r) and a dual-softmax loss over the long links (arm p = r + it) on the E273 pyramid head, against the E273 checkpoints (c) and the best graph so far (d)",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.02,
 "prediction": {"value": 0.05, "vs": "c"},
 "arms": {
  "c": {"recipe": "control, no new training: the E273 final checkpoints runs/e273_p_s0 and runs/e273_p_s1 (scoring cards runs/e273/<run>/card.json); seed 0's phases 2-4 come from earlier launches of the same run id (E273 summary), so the seed twin is loose"},
  "c_thr": {"recipe": "matched reference, no new training: c scored with a lower node threshold so that its node recall equals r's (in-sample on VAL: labelled); seed 0 sweep already in temp/e274/thr_sweep_e273_p_s0.json"},
  "r": {"recipe": "the E273 recipe (phases 2-7, same steps, same seeds) with two changes: the positives of the node existence loss x3 at event cells (--node-w-evt 2) and x2 at cells inside the grown intersection mask (--node-w-inter 1); the pixel event loss stays as in E273"},
  "p": {"recipe": "r's phase-3 node model (runs/e274_r_s<k>_p3) as the start, then phases 4-7 with r's flags plus the long-link dual-softmax loss (--long-match 0.5): the long-link candidates of one source node compete, so do those of one destination node, each with a 'none' entry; multi-positive NLL"},
  "d": {"recipe": "reference of the graph, no new training: recall_sp0_dp (Q 0.7424, F1@1.0 0.8781, topo F1 0.7405, junction F1 0.2730)"},
  "g": {"recipe": "control of the assignment, as E273 (E272 chain + geometric assigner on the same lane-level map tokens), scored inside every arm's card"}
 },
 "control": "c",
 "variant": "r",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "min_effect": 0.02},
  {"metric": "lane_acc_err", "vs": "g", "min_effect": 0.03},
  {"metric": "lane_acc_exact", "vs": "g", "max_loss": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01}
 ],
 "budget_gpu_h": 10,
 "kill": "a loss that is not finite; after phase 3 (the node gate, enforced by the trainer with --gate-node-prec 0.75) the node precision within 2 m (the gate's definition: delivered nodes at e > 0.5 against the true node positions) below 0.75 (E273: about 0.93 on 16 frames), or the E273 gate of phase 3 failed (node F1 on the cell grid below the E271 read-out - 0.05, median position error above 0.5 m); a step longer than 1.5 x its estimate (section 4)",
 "evaluate": {"split": "val"}
}
```
Verdict rule (by hand, `qgraph_run_eval.py verdict274`, tested by `evaluation/tests/test_verdict274.py`; `card.validate` cannot read the assignment numbers). Headline: `topo_F1` (the lane suite's function, VAL 873 frames, the E273 scorer), gain over the control of the same seed. **Arm r against c:** KEEP when the gain exceeds +0.02 on both seeds and the guards hold: assignment `lane_acc_exact` p - g not below -0.01, `lane_acc_err` p - g not below +0.03 (the E273 gain is kept; mean of the two seeds), `F1@1.0` not below c - 0.01 on either seed. **TIE** when both gains are inside +-0.02. **CUT** when a gain is below -0.02 on any seed or a guard is broken. Otherwise INCONCLUSIVE (add a seed). **Arm p against r** (the link effect, same nodes): the same rule with the headline `junction_F1` and band 0.03, and the extra guard `topo_F1` not below r - 0.01 on either seed. **Matched reference:** if c_thr gains more than +0.02 topo F1 over c on a seed, a KEEP of r needs r to beat c_thr by more than 0.02 as well (so that re-calibration is not mistaken for learning); the seed-0 sweep already shows the opposite (section 1: a lower node threshold lowers the topology). **Reported, not judged:** the system pass (topo F1 and F1@1.0 within 0.03 of d: 0.7105 / 0.8481), node recall and precision, long-link precision and recall, vertices per 100 m after Douglas-Peucker (budget 14.3: both arms deliver about twice that; the sparsification is arm s of a later card), duplicates, ms per frame. **Predictions (ranges, because the oracle probe removes nodes at random and the head's misses are clustered):** r: node recall (scorer, 2 m) 0.83 to 0.87-0.90, recall inside intersections 0.73 to 0.80-0.85 (diagnostic definition), topo F1 +0.03 to +0.08 (point +0.05, 0.596 to 0.65; the read-out sweep of section 1 warns that recall without precision does not help), F1@1.0 0.827 to about 0.86, junction F1 0.125 to about 0.16, node precision kept above 0.80; p: long-link precision 0.18 to 0.30, junction F1 +0.06 over r, topo F1 unchanged within 0.01. Whether p avoids the failure mode of E273 attempt v6 (an all-link row softmax: lanes found fell from 0.627 to 0.436) is a hypothesis; the cause of v6's failure was not isolated.

## 1. Where we are (all VAL, measured on the E273 checkpoints; scratch scripts `temp/e274/`, kept in a scratch tag)
Two definitions of node recall appear below and must not be mixed: the **scorer's** (`qgraph_eval.node_counts`, 873 frames, both seeds: 0.833) and the **diagnostic's** (`diag_nodes2.py`: a true node counts when any delivered node lies within 2 m, 300 frames, seed 0, exact map: 0.895; not one-to-one).
- **Recall explains the topology gap in the oracle, not the junction gap.** The oracle graph (VAL GT, every third frame = 291 frames) with nodes removed at random, scored by the same code: topo F1 0.967 with all nodes, 0.708 with 10 % removed, 0.583 with 17 % removed (the head's node recall is 0.83), F1@1.0 0.892, lanes found 0.682, junction F1 0.632 at 17 %. The head scores 0.596 / 0.827 / 0.637 / 0.125. So recall alone explains topo F1 and lanes found in the oracle, a part of the F1@1.0 gap, and almost none of the junction F1 gap. To reach the topo guard (0.7105) node recall has to be about 90 % (0.708 at 10 % removed with perfect links and no false nodes).
- **Where the nodes are lost** (diagnostic definition): recall 0.895 overall; **inside the grown intersection mask 0.726 against 0.939 outside** (11,344 of 54,322 true nodes, about 54 % of all misses); event cells 0.821 against plain cells 0.906; beyond 35 m 0.852 against 0.936 below 20 m. The misses are gaps, not slips: the median distance from a missed true node to the nearest delivered node is 3.8 m and 63 % have none within 3 m. The offset of the true node from its cell centre matters only a little (recall 0.907 below 0.5 m, 0.847 at 1.1-1.5 m).
- **Under-confidence at the cell level** (`diag_nodes.py`, exact cell): 81 % of the missed cells (10,997 of 13,654) have an existence probability between 0.2 and 0.5; at threshold 0.3 cell recall rises from 0.75 to 0.90 and 81 % of all false cells touch a true cell in the 3 x 3 neighbourhood (up to about 4 m; 76 % of the additional false cells).
- **The read-out threshold sweep without retraining** (`thr_sweep.py`, seed 0, exact map, 873 frames, link threshold 0.5, `temp/e274/thr_sweep_e273_p_s0.json`): node threshold 0.5 / 0.4 / 0.3 / 0.2: node recall 0.833 / 0.908 / 0.948 / 0.969, node precision 0.873 / 0.771 / 0.662 / 0.554, F1@1.0 0.832 / 0.867 / 0.876 / 0.882, lanes found 0.644 / 0.783 / 0.834 / 0.856, but **topo F1 0.597 / 0.548 / 0.473 / 0.407, junction F1 0.130 / 0.095 / 0.071 / 0.055 and duplicates 0.232 / 0.351 / 0.420 / 0.453**. More nodes from a lower threshold bring false and duplicate links: the oracle probe (nodes removed, no false nodes) is a ceiling for recall, not a prediction. The topo guard needs recall AND precision, i.e. a cleaner existence map, not a lower threshold. A 3 x 3 non-maximum suppression at the read-out is measured in `thr_sweep_nms.py` (summary).
- **The pixel stage's events over-fire** (`pixel_eval.py`, 300 frames, two seeds; peak above 0.5 against a true peak above 0.5, two-cell tolerance): the event heat maps have high recall and low precision (F1 split 0.37, merge 0.44, stop 0.57, exit 0.54, start 0.54, end 0.49), against lane centres 0.95 and road 0.93. The loss at event nodes is in the node stage (scorer recall of the type flags: stop 0.73, exit 0.70, split 0.57, merge 0.63), so scaling the pixel event loss would not target it: **the pixel event loss is not changed in r** (an earlier draft had `--ev-w 3`; the critic pass showed that the evidence does not support it).
- **Long links** (scorer): precision 0.18, recall 0.49, duplicates 0.21 against 0.077 for d: the candidates of one entry compete with each other and with the entries of the same exit, which independent sigmoids do not model; the all-link row softmax of E273 attempt v6 was worse (0.47 topo F1), but it changed the short links too. The long-link-only version is new.
- **The plan changed on the evidence:** the "path-level link loss" of the E273 recommendation (a loss on whole chains of short links) is dropped for now: the product of the link probabilities along a path is the sum of the per-link losses, and the oracle probe says chain breaks are mostly node gaps. What is left is the junction: an entry-to-exit assignment problem.

## 2. Design (what the arms change; code in `training/grid/qgraph.py` `CFG`, `long_match_loss`; trainer flags in `training/train/qgraph_train.py`; defaults reproduce E273: asserted in the self-test)
- **r.** `node_loss`: the existence focal loss gets a per-cell weight on the positives, `1 + node_w_evt * (cell holds an event flag) + node_w_inter * (cell inside the grown intersection mask)` (alpha 0.5 times it), the negatives are unchanged. Nothing changes in the read-out (e >= 0.5, p >= 0.5), the model, the pixel loss or the other losses. The mechanism is a bias towards recall at exactly the cells where recall is worst; precision is guarded by the node gate (kill criterion). r bundles the two weights (events, intersections): a KEEP is not attributable to one of them (no ablation in this card).
- **p.** `long_match_loss`: over the long-link candidates (`is_long`), a dual softmax with one fixed 'none' logit (0) per source node and per destination node; target = the mass of the true successors (multi-positive, no clamp), or 'none' for a node without a true long link; weight 0.5, added to the E273 link loss (the balanced BCE for the long links stays). The read-out still thresholds the independent link probability (a named output step). Start: r's phase-3 node model, so that p differs from r only in the link loss (phases 4-7 see a fresh random data order).
- **Commands** (`temp/e274/start_arm.sh`): r: `--node-w-evt 2 --node-w-inter 1 --gate-phases 3 --gate-node-prec 0.75`; p: the same plus `--long-match 0.5 --phases 4 5 6 7 --resume-from runs/e274_r_s<k>_p3/model.pt`. Same everywhere: TRAIN only (8,414 frames), seeds 0 and 1, steps of E273 (3000 / 6000 / 5000 / 2000 / 1500 / 2000), batch 4, width 48, bf16, the E271 start of the 1 m pixel branch, map errors random in training. The phase-2 gate (centre F1 not below the E271 head - 0.02 = 0.897; E273 reached 0.910 / 0.908) is read but not blocking in r, which does not touch phase 2 except through the random stream; a centre F1 below 0.88 is stopped by hand.

## 3. Controls and checks
- c = the E273 checkpoints, scored once (cards exist); c_thr = c at a lower node threshold (matched reference). The scorer, the control g and the reference d are the E273 ones (reproduction test of d in `evaluation/tests`).
- Layer 0: `qgraph.py` self-test with the switches (finite loss and gradient; the weight map: an event cell weighs 3 x, an intersection cell 2 x, other cells 1 x, measured on the gradient of the existence logits; the dual softmax against known cases: small for right logits, large for wrong ones, a source without a true link goes to 'none', two true successors share the mass, two frames are kept apart, a true logit far below its competitors still gets a gradient; switched off again = the E273 loss to 1e-6), plus the critic's brute-force check (loss and gradient equal to 1e-8) and the bit-equality of the default path against the committed E273 code. `test_verdict274.py` (the rule on fake cards).
- Layer 2: the oracle probe of section 1 is the ceiling for recall (a sanity check, not a bound); the null is the E273 head itself.
- Statistics: two seeds, noise band 0.02 on topo F1 (the E273 seeds differ by 0.002); scene bootstrap for the assignment guard as E273.

## 4. Steps, gates, cost
| step | what | gate / done when | cost |
|---|---|---|---|
| 0 | switches, tests, trainer flags, verdict function (done 2026-11-05) | self-tests pass | CPU |
| 1 | critic pass on the diff and the card (done 2026-11-05; blocking findings fixed) | no blocking finding left | agent |
| 2 | r_s0 and r_s1 in parallel through `run.py` (ledger E274.r_s0, E274.r_s1; tag lcr/E274.r_s*) | node gate after phase 3 (precision floor); verdict inputs after phase 7 | about 2.4 GPU h per seed (E273 seed 1: phase 2 about 65 min at 1.0-1.3 s per step, phases 3-6 0.19-0.20 s per step, phase 7 0.68 s per step), both seeds at once |
| 3 | score r and c_thr on VAL 873, pictures, verdict r against c | rule of section 0 | 0.5 GPU h |
| 4 | p_s0 and p_s1 from r's phase-3 models (ledger E274.p_s0, E274.p_s1) | phases 4-7 | about 1.1 GPU h per seed |
| 5 | score p, verdict p against r, videos, report | rule of section 0 | 0.3 GPU h |
Estimate 8-9 GPU h in all, budget 10; replaced by measured values after step 2. **A job starts only if it ends inside Mirko's window:** the launcher (`temp/e274/launch_window.sh`) checks start + estimate (r 2.7 h, p 1.4 h with margin) against the end of the window, for the p jobs too, the GPU is free, and no other chat's job runs (D57: a waiting lane job goes first; running jobs are never touched).

## 5. Risks and fallbacks
| risk | effect | what we do |
|---|---|---|
| recall weights buy false nodes | precision falls, links and density get worse | node gate after phase 3 (precision within 2 m below 0.75 stops the run); fallback: smaller weights |
| events are not learnable from the pixels (no paint inside intersections) | recall stays near 0.73 there | arm r's TIE/CUT says so; next: stop lines and crosswalks as a GT layer, or a map prior for the long links (the map-conditioned long-link logit of the E273 plan) |
| the dual softmax hurts the calibration of the link probability | the read-out threshold 0.5 gives too few / too many links | read the long-link precision and recall; fallback: weight 0.2 or a threshold in the rule |
| node recall rises but topo F1 does not | the links limit topo F1 after all, or duplicates | CUT / TIE of r by the rule; read the non-maximum suppression sweep; arm p and the map-conditioned logit become the main line |
| the phase-2 centre F1 gate (0.897) trips | a failed gate after 1.1 h | the gate is not blocking for r; stopped by hand below 0.88 |
| the sparsification is missing | both arms deliver about twice the vertex budget | named; arm s of a later card |

## 6. Assumptions I chose
1. Weights 3 / 2 and 0.5 are first guesses; no tuning on VAL before the run (the numbers of section 1 are diagnostics on the E273 checkpoints, not selections). 2. The topo F1 band is 0.02. 3. TEST is not used; a TEST check of a finished design needs Mirko's recorded OK. 4. The map is the lane-level map of E273 (lane GT with simulated errors). 5. Out of this card: the sparsification (arm s), temporal fusion by cross-attention to earlier nodes, the route attribute, the query-decoder reference (CGNet), the GT additions discussed in the report (events with or without painted evidence, occlusion by boxes), an ablation of the two node weights. 6. The scripts of section 1 (`diag_nodes.py`, `diag_nodes2.py`, `pixel_eval.py`, `thr_sweep.py`, `thr_sweep_nms.py`) are scratch kept in the git tag `lcr/e274-scratch`; they are promoted into a code folder with a test when a second card uses them.

## 7. Summary (2026-11-05; VAL 873 frames, TEST untouched; numbers from `runs/e273/` and `runs/e274/` cards, ledger rows E274.*)
**Verdicts (by hand, the rule of section 0, `verdict274`): arm r against c = CUT, arm p against r = CUT, arm p against c = CUT.** Baseline B (= c) stays the baseline of the lane head (D58); nothing of r or p is adopted.

| VAL 873, mean of 2 seeds | c = B | c_thr (seed 1, node thr 0.4) | r | p | d |
|---|---|---|---|---|---|
| topo F1 | 0.596 | 0.551 | 0.557 | 0.580 | 0.741 |
| F1@1.0 | 0.827 | 0.863 | 0.860 | 0.857 | 0.878 |
| junction F1 | 0.125 | 0.088 | 0.121 | 0.123 | 0.273 |
| lanes found | 0.637 | 0.768 | 0.767 | 0.744 | - |
| node recall / precision | 0.833 / 0.871 | 0.909 / 0.761 | 0.883 / 0.813 | 0.882 / 0.813 | - |
| duplicates (dup_frac) | 0.21 | 0.286 | 0.35 | 0.30 | 0.077 |
| long-link precision / recall | 0.18 / 0.49 | - | 0.18 / 0.51 | 0.22 / 0.48 | - |
| assignment p-g, exact map (guard >= -0.01) | -0.007 | - | -0.017 | -0.016 | - |
| assignment p-g, map error 1-2 m (guard >= +0.03) | +0.071 | - | +0.061 | +0.060 | - |

- **r.** Topo F1 -0.039 / -0.038 against c (band 0.02), the exact-map guard broken (-0.0173 / -0.0160), F1@1.0 +0.033 and the map-error gain kept. Reading: the weights move the operating point, they do not make the existence map cleaner. The matched reference c_thr (c at a lower node threshold, seed 1, in-sample) has the same topo F1 (0.551 against 0.557) and lanes found (0.768 against 0.756); r has the better junction F1 (0.121 against 0.088) and the better node precision (0.814 against 0.761), but more duplicates (0.34 against 0.29). Predictions: topo +0.05 missed (-0.039), junction F1 0.16 missed (0.121); node recall 0.883, F1@1.0 0.860 and node precision above 0.80 hit.
- **p.** The dual-softmax loss raised the long-link precision (0.18 to 0.22) and lowered the duplicates (0.35 to 0.30), the topo F1 rose +0.025 / +0.020 over r, but the junction F1 gain over r is +0.003 / +0.0002 (predicted +0.06, band 0.03). Against c: topo -0.015 / -0.018 (inside the band), the exact-map guard broken (-0.0164 / -0.0156). The long-link loss is not the junction lever; where the junction F1 is lost is not the competition between long-link candidates.
- **The assignment guard.** The gain with a 1-2 m map error holds in all arms (c +0.071, r +0.061, p +0.060). The exact-map guard breaks in r and p (about -0.016 against -0.007 for c): more duplicate nodes make the sections ambiguous for the assigner. The read-out step below was not yet scored on the assignment: that is a step of the new card.
- **Gate caution (found 13:57).** The live GATE line printed at the end of phase 4 or 5 and the phase-7 record are different models (the phase-7 gate re-runs gates 2-6 on the final model). Junction F1 is 0.16-0.18 live and 0.11-0.126 final for c, r and p alike (joint refinement, phase 7, lowers it and raises the vertices per 100 m from 22-24 to 28-32; 150-frame gate, not verified on 873). My first reading of the p run (a junction gain of about +0.05 over r) compared a live line with a final record and was wrong; the 873-frame scoring above is not affected.
- **Read-out diagnostic, in-sample on VAL (not a verdict).** `temp/e274/sweep_gpu.py` (scratch, scripts only, tag `lcr/e274-scratch-2`; the sweep outputs `sweep_<run>.json` are not kept, the numbers are in the tables here): one prediction per model, then several read-outs. The position rule (greedy by existence probability, a node is dropped when a kept node lies within R m, applied before the link read-out) with R = 1.5 m, mean of the two seeds:

| VAL 873, in-sample | topo F1 | F1@1.0 | junction F1 | lanes found | node recall / precision |
|---|---|---|---|---|---|
| c = B plain (thr 0.5) | 0.596 | 0.827 | 0.125 | 0.637 | 0.833 / 0.871 |
| B + rule, thr 0.5 | 0.626 | 0.822 | 0.260 | 0.621 | 0.793 / 0.912 |
| B + rule, thr 0.4 | 0.634 | 0.858 | 0.242 | 0.740 | 0.847 / 0.857 |
| B + rule, thr 0.3 | 0.619 | 0.867 | 0.222 | 0.779 | 0.877 / 0.795 |
| r + rule, thr 0.5 | 0.614 | 0.856 | 0.239 | 0.738 | 0.836 / 0.868 |
| B + rule R = 2.5 m, thr 0.4 | 0.204 | 0.480 | 0.254 | 0.220 | 0.458 / 0.894 |
| d (reference) | 0.741 | 0.878 | 0.273 | - | - |

The rule doubles the junction F1 of B and gives +0.03 to +0.04 topo F1; the gap to d on the junction F1 shrinks from 0.15 to 0.03, the gap on topo F1 stays 0.11. With the rule r is not better than B (0.614 against 0.634 topo F1 at equal F1@1.0): the existence weights add nothing once close nodes are removed. The R = 2.5 m rule removes half the nodes (2 m grid) and the topology collapses. The 3 x 3 cell rule (also in the sweep) deleted about 85 % of the nodes: an indexing error in my scratch code, **discarded**. The duplicate share `dup_frac` does not fall with the rule (0.19 to 0.20 for seed 1), so "duplicates" is a hypothesis for what the rule repairs; which nodes it drops is not yet looked at. Selection freedom: two radii, three thresholds, on the set that is also scored: in-sample, labelled so.
- **Consequences.** (1) The reweighting line (r) and the long-link competition line (p) are closed for the junction problem; closed_questions.md gets no entry yet (the read-out result changes the question). (2) A name for the rule and a clean test are the card E275 (`experiments/E275_node-readout-suppression/`); the soft, trained replacement (D55) follows as its own card after E275's analysis. (3) Open: an ablation of the two weights (events, intersections) was not run; seed 0 of c shares phases 2-4 with earlier launches (loose twin).
- **Cost.** Job hours: r 2 x 3.6 h (estimate 2.4 h each: at the 1.5 x kill line, not past it), p 2 x 1.8 h (estimate 1.1 h), scoring and sweeps about 1.3 h: 12 job hours against the 10 budgeted, about 6.7 h of wall clock because two jobs ran at once. The estimates of section 4 were too low for r.
