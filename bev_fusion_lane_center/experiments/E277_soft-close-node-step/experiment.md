# E277 The close-node suppression as a soft step inside the lane head, trained end to end (baseline B, D55 / D58)

Card of 2026-11-08 01:30, written alone (D60: Mirko is away, ask nothing; D57 / D61: this chat has priority). It follows E275 (the hard step, KEEP: topo F1 +0.03 and junction F1 +0.13 on both seeds, vertices per 100 m 28.9 -> 19.3, in-sample) and the failed sparse-target arm of E276 (arm s failed its phase-3 gate: the head cannot place nodes on a sparse target, the probability smears over about three cells per target node). The hard step is a rule step on the output (D55: "no rule step in the delivered graph"); this card puts it into the forward path as a differentiable step and trains the head with it, so that the delivered nodes at e >= 0.5 need no step after the network. GPU only in Mirko's windows. No TEST. Nothing pushed.

**D55 status.** After this card the thinning of the node row is part of the network (soft damping of a node by every stronger node within 1.5 m, passes in the forward path, a loss on the damped probability). The links and the assignment already take the damped existence. What remains outside the network is only the threshold e >= 0.5 and the greedy order inside the damping (a comparison without gradient, named).

## 0. The card (what the tools run)
```json card
{
 "id": "E277",
 "card": "",
 "title": "the lane head trained with the soft close-node step in its forward path (radius 1.5 m, smoothness 0.1 m, 3 passes), read out at e >= 0.5 without a hard step, against the baseline B read out as before and against B with the hard step of E275",
 "kind": "block",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.02,
 "prediction": {"value": 0.025, "vs": "c"},
 "arms": {
  "c": {"recipe": "control B = E273 runs e273_p_s0 / e273_p_s1, read out as before (node threshold 0.5, link threshold 0.5); scoring cards runs/e273/<run>/card.json (no new training)"},
  "h": {"recipe": "B with the hard step of E275 (radius 1.5 m, threshold 0.5): the cards runs/e275/<run>.close1.5_t0.5/card.json (no new training); the bar the soft step has to reach"},
  "u": {"recipe": "from the E273 phase-2 checkpoints (runs/e273_p_s<S>_p2/model.pt) phases 3 4 5 6 7 as E273 (the same steps, learning rates, losses, no tier weights) with `--soft-nms 1.5 0.1 3`: after the node head the existence logit is replaced by the damped one (qnodes.soft_suppress_logit), the node loss, the active-node choice of the links, the section stage and the assignment all see the damped existence; scored at e >= 0.5 with no hard step (runs/e277/<run>.plain_t0.5/card.json)"},
  "uh": {"recipe": "u with the hard step on top (close:1.5@0.5): reported, not judged (is anything left for the hard step to remove?)"},
  "g": {"recipe": "control of the assignment, as E273 / E274 / E275, scored inside every card"}
 },
 "control": "c",
 "variant": "u",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "min_effect": 0.02},
  {"metric": "junction_F1", "vs": "c", "min_effect": 0.03},
  {"metric": "lane_acc_err", "vs": "g", "min_effect": 0.03},
  {"metric": "lane_acc_exact", "vs": "g", "max_loss": 0.01},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "lanes_found", "vs": "c", "max_loss": 0.03},
  {"metric": "topo_F1", "vs": "h", "max_loss": 0.01}
 ],
 "budget_gpu_h": 4,
 "kill": "a loss that is not finite; or the gate of phase 3 (node precision within 2 m below 0.75, node recall within 2 m below 0.5, or the E273 node gate) failing on an arm: the run stops (exit 3) and the ledger row says which; or an arm that needs more than 1.5 x the E273 time for phases 3-7 (about 77 min at 0.19 - 0.24 s per step)",
 "evaluate": {"split": "val"}
}
```
**Verdict rule (by hand, written now; `qgraph_run_eval.py verdict277 --var e277_u_s0.plain_t0.5 e277_u_s1.plain_t0.5 --ctl e273_p_s0 e273_p_s1 --hard e273_p_s0.close1.5_t0.5 e273_p_s1.close1.5_t0.5`).** It is the rule of E275 for the soft step against the baseline (`verdict275` with the read-out `plain_t0.5`): the topo F1 gain over c above +0.02 AND the junction F1 gain above +0.03 on both seeds, and the guards held (assignment `lane_acc_exact` u - g not below -0.01 and `lane_acc_err` u - g not below +0.03 on the mean of the seeds, `F1@1.0` not below c - 0.01 and `lanes_found` not below c - 0.03 on either seed); plus one comparison with the hard step: the topo F1 of u is not below that of h by more than 0.01 on either seed. **KEEP (the soft step replaces the hard read-out)** when both hold. **TIE (the soft step helps, the hard step stays the read-out)** when the first holds and u trails h by more than 0.01 on a seed. **CUT** when a gain over c is below its negative band on a seed or a guard is broken. Otherwise INCONCLUSIVE (to Mirko: no further seed without retraining); a null result (both headlines inside their bands) is INCONCLUSIVE here, not TIE (the word TIE is reserved for "helps, but trails the hard step"). **Reported, not judged:** uh, node recall and precision, vertices per 100 m after Douglas-Peucker, duplicates, long-link precision and recall, the system pass (topo F1 and F1@1.0 within 0.03 of d: 0.7105 / 0.8481), ms per frame of the model (the step is 8 shifted products on the 50 x 50 grid; negligible). **In-sample label:** the radius 1.5 m was chosen on VAL in E274 / E275, the threshold 0.5 is the card's; every number is in-sample on VAL until a held-out check (TEST, only with Mirko's recorded OK).

**Prediction (a number):** u reaches the hard step: topo F1 0.62 +- 0.01 on both seeds (c + 0.025); junction F1 0.22 or more (h: 0.27 / 0.25; the head can also learn to place nodes so that fewer pairs are damped); lanes found 0.60 - 0.64; vertices per 100 m 18 - 22; exact-map assignment guard held (h: -0.0066 / -0.0058). Why it could miss: a GT node pair within 1.5 m (32 % of the GT nodes) cannot both stay, so the loss on the damped probability has a floor on those pairs (the network may spread the pair or lose one node); the hard step lost 4 points of node recall for the same reason.

## 1. Where we are (E275, VAL 873, per seed, in-sample)
| | topo F1 | junction F1 | F1@1.0 | lanes found | vertices per 100 m | node recall / precision |
|---|---|---|---|---|---|---|
| c = B plain | 0.597 / 0.595 | 0.130 / 0.120 | 0.832 / 0.821 | 0.644 / 0.630 | 28.9 / 31.5 | 0.833 / 0.873, 0.833 / 0.868 |
| h = B + hard step | 0.626 / 0.625 | 0.271 / 0.249 | 0.828 / 0.816 | 0.627 / 0.614 | 19.3 / 20.1 | 0.793 / 0.915, 0.793 / 0.910 |
The step thins the dense node row (65 % of the dropped nodes are counted true by the scorer, 93 - 95 % of the true nodes lost have a GT node within 1.5 m): it is a density reduction that also cleans the junction links, not a removal of false nodes only. Mirko's video note 5 ("the point density is very high"). Arm s of E276 tried the same through the target and failed the phase-3 gate (3.6 nodes per 100 m at e >= 0.5, recall 0.15 against the sparse GT; at e >= 0.3 the head delivers 50 per 100 m with recall 0.91: the position along a straight lane is not predictable, the probability smears).

## 2. Design
- **The step** (`qnodes.soft_suppress_logit`, tests `training/grid/test_soft_nms.py`): e'_i = e_i * prod over the 8 neighbour cells j with e_j > e_i of (1 - e'_j * k_ij), k_ij = sigmoid((1.5 - d_ij) / 0.1) with d the distance of the predicted node positions (cell centre + offset), the damped e'_j of the previous pass in the product from the second pass on (3 passes: exact for chains of three nodes), ties by the lower cell index. The ordering is a comparison without gradient (the order of the hard rule); the gradient reaches e_i, the stronger e_j and both positions. Only the 3 x 3 neighbourhood matters for a radius up to 2 m (two cells apart are at least 2 m apart).
- **Where it sits.** `QGraph(soft_nms=(1.5, 0.1, 3))` replaces `no['e']` right after the node head (`no['e_raw']` keeps the head's own logit); the node loss, the active-node choice of the link stage (the top-K by e), the assignment's e and the read-out all use the damped existence; the setting travels with the checkpoint (`args['soft_nms']`, `qpredict.load_model`). Training: `qgraph_train.py --soft-nms 1.5 0.1 3` (default off = E273).
- **Tests (layer 0).** The hard limit: on 183 random frames (12 x 12 cells, saturated probabilities) the kept set of the soft step at smoothness 0.002 and 8 passes equals the greedy hard rule of E275 (`suppress_close_nodes`) in every frame; the chain case (one pass drops the third node, two passes keep it like the hard rule); ties; flipping the grid flips the result; damping only; isolated nodes unchanged; the gradient reaches the stronger node's logit (negative), the weaker node's own logit, the positions; borders and bf16 input; the model switch adds no parameter, the default model has no `e_raw` and its logit is the plain one, the soft model's `no['e']` is the damped logit of the plain model's; `load_model` restores the setting and an old checkpoint stays plain; `qgraph.py` selftest: every phase 3-7 finite with a gradient into the trainable groups. `test_verdict277.py`: the written rule on fake cards. Known limit: with probabilities of 0.6 - 0.7 a chain of three nodes is damped only partly (a node of e 0.69 behind one of 0.71 keeps 0.2), so the soft step equals the greedy rule only where the head is saturated; this is what "soft" means and is why the card compares with the hard step instead of assuming it.
- **Reproduction (layer 1).** A hard-step reading of the baseline reproduces the E275 cards (done: difference 0). The soft step on the baseline's own predictions is not an arm (the baseline has no damped training).
- **Seeds.** Both E273 seeds (0, 1): two trained models per arm, as E274 / E276; the baseline cards c and h exist already.

- **Critic pass (`ar-critic`, mode code, 2026-11-08 01:30 - 01:45; the function itself checked against an independent loop implementation, output and gradients, B = 1 / 2 / 3).** Blocking, all fixed after the first two jobs had started: (1) the phase-3 gate (`qgates._node_stats`) and (2) the timing script (`qgraph_timing.py`) called the raw node head: both now call `QGraph.node_stage`, the one place that applies the step (the forward path, the gates and the timing use it; a gate test in `test_soft_nms.py`); (3) `verdict277` labelled a null result TIE where this card says INCONCLUSIVE: fixed with a test. Non-blocking, taken: the clamp at 1e-6 cut the gradient beyond a logit of 13.8 (now a clamp-free formula, a test at logits +-14.5 and +-30), a model-level test with a hand-built node head that pins the pass count and the radius of the model setting (the first model test could not see them), a batch of three frames and the 50 x 50 grid against the frame-by-frame result, the production setting against the hard rule (the gap bound), argument checks (tau 0, no pass, a radius above 2 m), the `ckpt.pt` fallback of `load_model` reads the run's `config.json`, `verdict277` checks that the hard-step cards are the control's models, the prediction of the JSON and the prose agree. **Consequence for the running jobs:** E277.u_s0 (started 01:12) and E277.u_s1 (01:19) run the code of before the fixes: their phase-3 gate read the raw head (so the gate lines in their logs are not a reading of what the arm delivers; the scoring cards are the judge) and the trained models carry the 1e-6 clamp (a dead gradient only beyond a logit of 13.8; the scoring uses the fixed function, which gives the same values up to 1e-5 inside that range).

## 3. Controls and checks
- c, h and g from existing cards (E273 / E275). The paired comparison of u with c and h is per seed (same seed number, same phase-2 start).
- Layer 2: not applicable (a read-out inside the network); the null is c.
- After training: the pictures of the 20 frames with the most damping (delivered nodes of u next to those of h on the same frames) and the count of nodes that u delivers at e in [0.2, 0.5) against c (the delivered, damped existence of u against the plain one of c, from the full dumps; the raw logit `e_raw` is not exported) are looked at before the summary.

## 4. Steps, gates, cost
| step | what | gate / done when | cost |
|---|---|---|---|
| 0 | this card; `experiment.py check E277` | card valid | CPU |
| 1 | code, tests (done), critic pass (`ar-critic`, mode code) on the diff | tests green, no blocking finding | CPU + agent |
| 2 | train u for seeds 0 and 1 (ids E277.u_s0, E277.u_s1; the scheduler of the night, two slots) | gate of phase 3 (node precision within 2 m >= 0.75, recall within 2 m >= 0.5), finite loss | about 1.4 GPU h per seed |
| 3 | score u (read-outs plain@0.5 and close:1.5@0.5; ids E277.uq_s0, E277.uq_s1) | cards written | about 0.2 GPU h per seed |
| 4 | `verdict277`, ledger notes, summary | verdict written | CPU |

## 5. Risks and fallbacks
| risk | effect | what we do |
|---|---|---|
| the loss on the damped probability fights the GT pairs (32 % of the GT nodes have a neighbour within 1.5 m) | recall of u below h's, lanes found falls more than the guard | the card reports node recall; fallback: a smaller radius (1.25 m) or the pairs excluded from the positives; not tried tonight |
| training instability (the damping couples cells; the gradient reaches the stronger node's probability) | NaN / divergence | the trainer's guards (clipping, roll-back); the gate of phase 3 stops the run |
| u only learns to push all nodes below 0.5 near others (lower recall) instead of placing them better | junction F1 gain without topo gain | topo F1 against h decides (rule); node recall reported |
| the soft step is not exactly the hard rule (mid-range probabilities: about 1 % of the cells differ at saturated probabilities, about 7 % at 0.5 - 0.9) | differences to h that are no defect | compared with h on the graph numbers, not assumed equal |
| the position of a non-GT cell gets a gradient only through the radius term (the offset loss acts on positives), so the head may pull a false node onto a stronger neighbour instead of lowering its probability; a damped GT node of a pair stays an active link candidate in training (teacher forcing adds 2 to GT cells) but needs e' > 0.05 at inference | false nodes that survive as one doubled node; links trained on nodes that the read-out no longer delivers | look for it in the comparison pictures; the cards report node precision and recall |

## 6. What comes after (not in this card)
- If KEEP: the delivered chain has no rule step between the node head and the output except the thresholds; E276's sparse idea revisited as a set-based node loss (a one-to-one matching inside the loss, allowed by D55) so that the head may place a node anywhere along a straight lane.
- The long links (7 - 8 % of the delivered links, precision 0.17 - 0.19, 27 - 32 % of the wrong links, E276 section 7a P9) are the next lever for the topology; their share in the topo F1 is large (without them the topo F1 falls by 0.12, P10).

## 7. Summary
Open: training not started (code and tests done 2026-11-08 01:30; critic pass and the jobs next).
