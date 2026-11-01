# Spurmitte (lane centre) E2E set-up - what is pulled over from the BEV system (2026-11-01)

Asked by Mirko: prepare a Spurmitte experiment in a prepared set-up, take over what the BEV system (`bev_fusion/`) does that helps, differentiable E2E designs only. Basis: `bev_fusion/training/architecture.md` (five layers), `bev_fusion/plan.md` 8.5 (the BEV chat's Spurmitte plan: dense head first, D26), `training/architecture.md` of this project (24 stages on the main path: 8 learned, 2 fixed-differentiable, 2 frozen, 12 rule steps; BEV: 14 stages, 0 rule steps).

## What is pulled over, by E2E value
| # | BEV concept | use here | state |
|---|---|---|---|
| 1 | dense differentiable output interface, hard selection only for the metric (`interface.py`, `extract.py`) | the builder's output as a contract (`training/models/p16_interface.py`); later soft graph tokens for the route head (E269); the hard assembly stays a layer-5 step outside the loss | contract built; E269 proposed |
| 2 | slot tables, `check_config`, `tests/test_slots.py` (gradient reaches every trainable parameter) | `evaluation/drivers/e2e_gradient_report.py` measures which groups receive gradient and whether the input features do; slot tables for the rule-step replacements come with the first replacement, not before | report built (selftest passed); first real run waits for free memory |
| 3 | equivalence test old code against new | the golden numbers and the E260 graph hashes already do this for any restructuring; no new tool | existing |
| 4 | trainable front end through a shared trunk (D26), modality dropout | the builder on the BEV trunk features S, frozen then trainable (stage S4 of the BEV plan) | later, step 2 |
| 5 | precompute + visibility maps, one coverage rule for loss, evaluation, execution (D36 / D37) | lane loss and R@1 far only where the camera sees (D36: lanes are camera-only); the 45 - 48 m ring finding of E266 | later; the BEV chat owns the rule |
| 6 | world-aligned temporal memory, sequence training | replaces the hard-match time recursion (a rule step) with a recurrent token state | later, step 3 |
| 7 | loss machinery (Kendall / UW-SO, gradient-conflict cosines, EMA, warm start) | the builder's hand-set weights; the recall weight is a coverage lever, not a Q lever | later |
| 8 | board / champion file, trainsuite watch, keep3, GT signature | one table across arms with per-column bands | later |

## Measured today: gradient report of `e123_D7_r3_f1` (2 VAL frames, CPU float32, the trainer's loss; indicative, two frames only)
- **No tensor of the builder is silent**: all 101 parameter tensors of the nine module groups get gradient from the builder loss. So inside the network the gradient is complete; the E2E gap is what lies after and before it: the route head trains on stored token files, the consumer on the assembled graph, the assembly runs after the network, the GaussianCaR features and the dense head are cached and frozen.
- Share of the gradient norm (it follows the loss scales, not importance): seeds 81.7 %, ray read-out 7.3 %, stem 6.3 %, projections 3.9 %; context, junction head, child tokens, rescoring and link head together under 1 %.
- **Gradient on the input channels** (what a trainable front end, D26, would receive): mean |grad| per channel 8.8e-6 (GaussianCaR decoder PCA, 32 channels) and 9.7e-6 (map logits, 7) against 3.1e-4 for the dense-head heat and 5.9e-5 for the direction field. The frozen features get a small but nonzero signal; the dense-head channels, which are stored arrays today, would get the largest. Step 2 has to normalise for that scale before it compares arms.
- Consequence for E269: gate G1 is not "the builder groups light up" (they already are) but "the route loss alone reaches the seeds, ray read-out and link groups, which it did not in E202 / E214 / E232".

## G1(1) of card E269, measured (route loss alone through the old joint path, `e123_D7_r3_j1`, 2 TRAIN frames, CPU; indicative)
The route loss reaches the builder strongly through the node tokens `h` (seeds group 7.3 against 8.3 for the builder's own loss, stem 3.3 against 2.2, child tokens 1.0 against 0.16, `h` itself 20 times), but almost orthogonally to the builder objective (cosines -0.06 ... +0.05 except the seeds group, 0.51). Exactly zero: node positions, validity, link logits, junction head, rescoring, link head (34 tensors). So the gap is the missing position / validity / link path, and the loud near-orthogonal gradient on the shared features is a new candidate explanation of the three negative joint fine-tunes. Details in the card.

## Steps
- **Step 0 (no behaviour change), done today:** `p16_interface.py` (contract of `Builder.forward`, selftest), `e2e_gradient_report.py` (selftest passed; the real run on `e123_D7_r3_f1` is in the section above). Still to do, needs a critic pass because the files are shared: call `check_outputs` on the first batch in `p16_train.py` and in `p16_infer.predict`.
- **Step 1, card E269 (revised after three critic passes):** first a cheap proxy, a straight-through gradient from the route loss to the node positions with the old head, read-out and evaluator unchanged; the full soft-node design (validity, links, node read-out) is the follow-up E270, written only if the proxy shows an effect. Order: gradient report of the route loss alone through the old path (no new code), then the build, then four trainings.
- **Step 2:** builder on the BEV trunk S (frozen, then trainable image features), against the champion on frozen GaussianCaR features on the same frames. Risk: the GaussianCaR features carry the lane skill (E27 +0.24 F1), the BEV maps are weaker; coordinate with the BEV chat (D26 plan 8.5 S4) and the machine lock (D19).
- **Step 3:** replace rule steps in order of measured value (fork pruning J 0.265 -> 0.409 first, then the time recursion, anchoring, relocation), each reproducing the reference before it counts.

## Not taken over
The BEV refiner (GT matching inside the network), the frozen cached camera heads as a design goal, `select_boxes` / `decode` as part of a main path.

## Open for Mirko
1. The E2E lane work in the BEV code (CentreHead on S, plan 8.5, decided D26) and the lane builder on S later are one ladder: confirm that this project prepares the contract, the gradient report and the route experiment, and the BEV chat the dense head.
2. Is Douglas-Peucker (D20) the accepted layer-5 sparsification, or must the vertex budget come out of the network?
