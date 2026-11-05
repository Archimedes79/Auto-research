# E275 Close-node suppression as a named read-out step on the baseline lane head (E273 checkpoints, D58), with a look at what it drops

Card of 2026-11-05, written after the E274 verdicts (arm r CUT, arm p CUT). It follows from an in-sample diagnostic of E274 (section 1): dropping every delivered node that lies within 1.5 m of a node with a higher existence probability doubles the junction F1 of the baseline and raises the topo F1 by 0.03. That diagnostic was a scratch sweep with no rule written before it and one invalid variant (the 3 x 3 cell rule, an indexing error). This card turns it into a named step with a written rule, tests, the missing checks (assignment, what is dropped, sensitivity to the radius) and a reproduction. Priority: D57 (Spurmitte first). GPU only inside Mirko's windows (nights 22:00-07:00, Monday-Friday 08:30-17:30). No new training.

**D55 status.** The step is a hard selection on the output, applied to the delivered nodes before the links are read: a named rule step, not part of the differentiable path. It is a diagnostic and a stop-gap. The goal is a trained, soft replacement (section 6, a later card E276); this card does not claim the head is network-only with the step in place.

## 0. The card (what the tools run)
```json card
{
 "id": "E275",
 "card": "",
 "title": "close-node suppression (radius 1.5 m, greedy by existence probability, applied to the nodes with e >= threshold before the link read-out) on the E273 baseline checkpoints, against the same checkpoints without it",
 "kind": "block",
 "deterministic": true,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.02,
 "prediction": {"value": 0.03, "vs": "c"},
 "arms": {
  "c": {"recipe": "control, no new training: the E273 final checkpoints runs/e273_p_s0 and runs/e273_p_s1 read out as before (node threshold 0.5, link threshold 0.5); scoring cards runs/e273/<run>/card.json"},
  "n": {"recipe": "the same checkpoints and thresholds, plus the step: among the nodes with e >= 0.5 keep a node only if no kept node with a higher e lies within 1.5 m; the links are read between the kept nodes; scoring cards runs/e275/<run>_n/card.json"},
  "n4": {"recipe": "as n at node threshold 0.4 (the operating point that gave the best sweep numbers; chosen on VAL, in-sample): reported, not judged"},
  "rn": {"recipe": "the E274 arm r checkpoints with the step at node threshold 0.5: reported, not judged (do the existence weights add anything once close nodes are removed?)"},
  "c_thr": {"recipe": "matched reference: c at node threshold 0.4 without the step (so that n4 is compared with the effect of the threshold alone); reported"},
  "d": {"recipe": "reference of the graph, no new training: recall_sp0_dp (Q 0.7424, F1@1.0 0.8781, topo F1 0.7405, junction F1 0.2730)"},
  "g": {"recipe": "control of the assignment, as E273/E274 (E272 chain + geometric assigner on the same lane-level map tokens), scored inside every card"}
 },
 "control": "c",
 "variant": "n",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "min_effect": 0.02},
  {"metric": "junction_F1", "vs": "c", "min_effect": 0.03},
  {"metric": "lane_acc_err", "vs": "g", "min_effect": 0.03},
  {"metric": "lane_acc_exact", "vs": "g", "max_loss": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "lanes_found", "vs": "c", "max_loss": 0.03}
 ],
 "budget_gpu_h": 1,
 "kill": "the step deletes more than 20 % of the delivered nodes at threshold 0.5 on any seed (the in-sample sweep: about 9 %; the invalid 3 x 3 rule deleted 85 %), or its output depends on the order of the input nodes, or the reproduction of step 2 misses the sweep numbers of E274 section 7 by more than 0.005: stop and search the code before any other number is read",
 "evaluate": {"split": "val"}
}
```
Verdict rule (by hand, written now; two calls of `qgraph_run_eval.py verdict274` for the headline (topo F1, band 0.02) and for the junction F1 (band 0.03), the lanes-found guard by hand): **KEEP the step as the named read-out of the lane head** when the topo F1 gain over c exceeds +0.02 AND the junction F1 gain exceeds +0.03 on both seeds, and the guards hold: assignment `lane_acc_exact` n - g not below -0.01 and `lane_acc_err` n - g not below +0.03 (mean of the two seeds, as E274), `F1@1.0` not below c - 0.01 and `lanes_found` not below c - 0.03 on either seed. **TIE** when both gains are inside their bands (then no step: the cheaper variant). **CUT** when a gain is below its negative band on any seed or a guard is broken. Otherwise INCONCLUSIVE (the step was kept to one operating point; a further seed is not available without retraining, so INCONCLUSIVE goes to Mirko). **Reported, not judged:** n4, rn, c_thr, the radius curve, node recall and precision, duplicates, long-link precision and recall, vertices per 100 m after Douglas-Peucker, the system pass (topo F1 and F1@1.0 within 0.03 of d: 0.7105 / 0.8481), ms per frame of the step. **In-sample label:** the radius 1.5 m, the threshold 0.5 and the 0.4 / 0.3 variants were seen on VAL in the E274 sweep; the effect (junction F1 four times the band) is far larger than the selection freedom (two radii, three thresholds), but every number of this card is in-sample on VAL until the held-out check of section 6.

**Predictions (the reproduction is not one; these are):** (1) the exact-map assignment guard holds for n (n - g not below -0.01; c is at -0.007) and for rn (rn - g not below -0.01; r is at -0.017: the step repairs the section ambiguity of the duplicate nodes); (2) at least 40 % of the dropped nodes lie inside the grown intersection mask (the base rate of true nodes there is 21 %); (3) the share of true nodes that have another true node within 1.5 m (the recall the step cannot keep) is below 3 %; (4) the radius curve (1.0, 1.25, 1.5, 1.75, 2.0 m, thr 0.5) has its junction F1 maximum between 1.25 and 1.75 m and collapses at 2.0 m (the grid spacing); (5) ms per frame of the step below 1 ms.

## 1. Where we are (E274 section 7, VAL 873, mean of two seeds, in-sample)
| | topo F1 | F1@1.0 | junction F1 | lanes found | node recall / precision |
|---|---|---|---|---|---|
| c = B plain (thr 0.5) | 0.596 | 0.827 | 0.125 | 0.637 | 0.833 / 0.871 |
| B + step R 1.5 m, thr 0.5 | 0.626 | 0.822 | 0.260 | 0.621 | 0.793 / 0.912 |
| B + step R 1.5 m, thr 0.4 | 0.634 | 0.858 | 0.242 | 0.740 | 0.847 / 0.857 |
| r + step R 1.5 m, thr 0.5 | 0.614 | 0.856 | 0.239 | 0.738 | 0.836 / 0.868 |
| B + step R 2.5 m, thr 0.4 | 0.204 | 0.480 | 0.254 | 0.220 | 0.458 / 0.894 |
| d | 0.741 | 0.878 | 0.273 | - | - |
- The step deletes about 9 % of the delivered nodes at thr 0.5 (node recall -0.04, precision +0.04, counted by the scorer: about half of the deleted nodes were counted false). The duplicate share `dup_frac` does not fall with it (seed 1: 0.192 to 0.201), so "duplicate nodes" is a hypothesis for what the step repairs; step 3 looks at the dropped nodes.
- The exact-map assignment of arms r and p is below the guard (-0.017 / -0.016, c -0.007); the step has not been scored on the assignment.

## 2. Design
- **The step.** Input: the delivered node list of a frame (position, existence probability e, type flags, direction). Keep the nodes with e >= threshold; sort by e descending (stable, ties by index); go through them and keep a node unless a kept node lies within R metres (Euclidean, BEV metres); the dropped nodes are removed from the node list before the links are read, so no link ends in one. Nothing else changes: the model, the thresholds of the links (0.5), the scorer and the assignment are those of E273. Equivalent to the scratch rule of E274 for thresholds >= 0.2 (a node below the threshold can only suppress nodes of lower e, which are below the threshold too).
- **Code.** `evaluation/qgraph_readout.py` (new; promoted from `temp/e274/sweep_gpu.py` at its second use: `suppress_close_nodes(nodes, radius, threshold)`), a flag of the scorer (`qgraph_run_eval.py score --readout close:1.5 --exp e275`) and a test (`evaluation/tests/test_qgraph_readout.py`). The 3 x 3 cell rule is not promoted (invalid).
- **Tests (layer 0).** Two nodes 1.0 m apart: the stronger stays. Two nodes 2.0 m apart: both stay. A chain at 0, 1.0, 2.0 m with e 0.9 / 0.8 / 0.7: nodes 0 and 2.0 stay (the third is 2.0 m from the kept first). Equal e: the lower index stays. Order of the input list does not change the output set (random permutations). Nodes below the threshold are never returned and never suppress. Empty and one-node frames. The output is a subset of the input and no two returned nodes are closer than R. Brute-force comparison with an O(n^2) reference on random frames. Link read-out between kept nodes only (no link to a dropped node).
- **Reproduction (layer 1).** The step on the stored predictions of seed 1 at R 1.5 m, thr 0.5 must give the sweep numbers of E274 (topo F1 0.625, junction F1 0.249, node recall 0.793, precision 0.910; tolerance 0.005); likewise seed 0 (0.626 / 0.271).
- **What is dumped for the analysis.** One prediction pass per model (B s0, s1, r s0, s1), the predicted nodes of the 873 frames are written to `temp/e275/pred_<run>.pt` (scratch): the analysis of step 3 and the radius curve run on CPU from these files, so the GPU is needed once per model.

## 3. Controls and checks
- c and c_thr come from existing cards (c_thr: seed 1 from the sweep, seed 0 `thr_sweep_e273_p_s0.json`, re-scored through the card scorer so that it has an assignment number). g, d, the scorer and the bootstrap are those of E273 / E274.
- Layer 2: the step on the oracle (the VAL GT nodes with a small random jitter) must delete only the true pairs that lie within R (that is prediction 3: the recall ceiling of the step); the null is c itself.
- Statistics: both E273 seeds; the step is deterministic, so the two seeds are two models, not two runs of the step. Paired scene bootstrap for the assignment guard as E273; for topo F1 and junction F1 the suite tallies are used per scene if the functions allow it (checked in step 1), else the two-seed agreement is the only evidence and the summary says so.
- Dropped nodes (step 3, per seed): count and share of the delivered nodes; share counted true / false by the scorer's matching; share inside the grown intersection mask and at event cells; distance to the stronger neighbour (histogram); e of the dropped node and of its suppressor; the same for the true nodes lost (recall -0.04); 20 frames as pictures (delivered nodes, dropped nodes marked, GT).
- The share of GT nodes that have another GT node within 1.5 m (the ceiling).

## 4. Steps, gates, cost
| step | what | gate / done when | cost |
|---|---|---|---|
| 0 | this card; `experiment.py check E275` | card valid | CPU |
| 1 | `qgraph_readout.py`, scorer flag, `test_qgraph_readout.py`, `verdict275` wrapper (two `verdict274` calls + lanes-found guard) with a test on fake cards; critic pass (`ar-critic`, mode code) on the diff | tests green, no blocking finding | CPU + agent |
| 2 | prediction pass for B s0, B s1, r s0, r s1 with the dump; score c-with-step, n4, rn, c_thr cards; reproduction check | reproduction within 0.005; kill criterion not hit | about 0.5 GPU h, inside a window |
| 3 | analysis of the dropped nodes and the radius curve from the dumps (CPU), pictures | numbers for the five predictions | CPU |
| 4 | verdict by the rule above; ledger rows E275.n_s0, E275.n_s1 (through run.py: the scoring command, `--no-card` not used because the verdict feeds from it); summary | verdict written | 0.1 GPU h |
Estimate 0.6 GPU h, budget 1. A job starts only inside Mirko's window, the GPU is free, no other chat's job runs, and the BEV system chat knows (it was told at 18:43 that nothing of mine runs before 22:00; D57: a waiting lane job goes first, running jobs are never touched).

## 5. Risks and fallbacks
| risk | effect | what we do |
|---|---|---|
| legitimate nodes closer than 1.5 m (a split or merge point with one node per lane) are merged | recall ceiling, lanes found and F1@1.0 fall | prediction 3 measures the ceiling; the lanes-found guard (-0.03) and F1@1.0 guard stop a bad step; fallback: a smaller radius (the curve) or suppress only nodes of the same direction and type |
| the gain is an artefact of VAL | the step does not carry over | in-sample label; the held-out check of section 6 before the step enters a headline; no TEST use before Mirko's recorded OK |
| the step hides a defect that training should fix (D55) | the head is not network-only | named in the card and the report; E276 replaces it |
| the assignment gets worse (fewer sections near a junction) | guard broken | the guard is in the rule; the summary shows where |
| the dropped nodes are mostly true (counted by the scorer) | the metric gain comes from a scoring effect of the duplicates, not from a better graph | step 3 reports it; pictures of 20 frames decide whether the delivered graph is better to look at |

## 6. What comes after (not in this card)
- **E276, the soft replacement (D55).** The step is the limit of a trained one. Options, to be decided after step 3 shows what is dropped: (a) a set-based node loss (one prediction per true node, the others get the 'none' target; a one-to-one matching inside the loss is allowed by D55); (b) a repulsion loss on pairs of nodes within R where the GT has one node (differentiable, cheap); (c) a soft suppression inside the forward path, e'_i = e_i * prod over j with e_j > e_i of (1 - e_j * k(d_ij)), the ordering without gradient, trained with the node loss on e' (the hard rule is its limit). Recommendation: (c) with (b) as the cheap first arm. Needs a night window (training, about 2 x 3.6 h per seed in the E274 timing).
- **Held-out check.** The step (or its replacement) on a set it has not been selected on: TEST, once per finished design and only with Mirko's recorded OK (a grant under a decided D<n>); so it waits for E276.
- **The joint refinement (phase 7)** lowers the junction F1 from about 0.17 to 0.12 in every variant at the gate (150 frames, E274 section 7): a separate lever (keep the node and link losses active in phase 7, or a shorter phase), to be tested on 873 frames before it is believed.
- **A topo F1 error breakdown** (missing, false and fragmented links) after the step: the remaining gap to d is 0.11.

## 7. Summary
Open.
