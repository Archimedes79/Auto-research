# E269 Soft graph tokens: let the route loss reach positions, validity and edges

Card of 2026-11-01, status **proposed**. First E2E experiment of the Spurmitte set-up (`experiment_execution/plans/spurmitte_e2e_setup_2026-11-01.md`). Block experiment: new code, so the detail plan (section 2) is written when the loop reaches it. `experiment.json` is written when the card is accepted.

## 0. What the tools run (the fields of `experiment.json`)
| field | value |
|---|---|
| id / card / title | E269 / - / soft graph tokens: the route loss reaches the graph |
| kind | block |
| deterministic | false (trained arms, two seeds each) |
| metric, direction, band | route success (failure half: night, blur, three cameras off), higher; band 0.03 (route-head seed spread: up to 0.03 on clean frames, 0.05 - 0.09 under failure, so paired seeds and the paired scene bootstrap decide) |
| prediction | arm a - arm c = +0.03 route success in at least two of the three failure types |
| arms | a: node-wise route head on soft graph tokens, joint with the builder (builder at 0.3x learning rate, route loss ramped in); c: the same route head, every input from the builder detached (equal parameters, equal budget); b: reference, the stored-token route head of the registry (no training) |
| control | c |
| rule | route success a - c >= +0.03 in at least 2 of 3 failure types AND Q and J of the assembled clean-frame graph no worse than -0.005 / -0.02 against c |
| budget_gpu_h, kill | about the budget of E232 (4800 steps per arm, 2 seeds, 4 runs); gate G1 and the kill criterion below |
| evaluate | split val (873 frames) + the VAL failure series; compare c; TEST never |

## 1. Card
| | |
|---|---|
| lens | system review (the E2E gap table of `training/architecture.md`) |
| group | A4 differentiable selection and training regime, A7 navigation context to the planner |
| sources | E202, E214, E232 (three negative joint fine-tunes), `training/architecture.md` rows "Route input tokens along edges", "Joint builder + route fine-tune (closed)"; bev_fusion layers 4 / 5 (`../../bev_fusion/training/architecture.md`) |
| status | proposed |

- **Statement:** a route head that reads a differentiable, validity-weighted view of the builder's nodes lets the route loss improve the builder, where the same loss through the hard assembled graph did not (three negatives).
- **Mechanism:** the earlier joint runs ran the assembly inside the loop; the route loss reached only the node tokens `h`, because the route input takes, per vertex, the node of highest validity (hard, numpy) and the cross-sections from the assembled edges. Positions, path validity `pi` and link logits received no route gradient. Here the route logit is made per beam node: token = `h_i`, the SD-map raster and the map channels sampled at `pos_i` (bilinear, differentiable in the position), `pi_i`, and two rounds of message passing along the parent edges and the link edges weighted by the link probability. The loss is the route BCE at the nodes, weighted by `pi_i * valid_i`; the target (the nearest ground-truth lane point's route flag) is computed with stop-gradient. The vertex-level route flag of the delivered graph is read from the node logits after the network (the hard node-to-vertex map is post-processing, outside the loss).
- **Prediction:** route success a - c = +0.03 or more in at least two of night, blur, three cameras off (VAL failure series); Q and J on clean frames inside -0.005 / -0.02 of c.
- **Decision rule:** KEEP if both clauses hold with the paired scene bootstrap interval above 0 on route success; CUT if a - c <= 0 in all three types (a fourth negative closes the joint line for good); otherwise TIE, take the cheaper arm (c). INCONCLUSIVE (seeds disagree) means a third seed, not an adoption.
- **Test design:** arms a / c / b as above, 2 seeds for a and c, start from `e123_D7_r3_j1` (no time recursion, 192 / 96), 4800 steps at the E232 recipe; controls: the equal-parameter detached arm c (separates "more parameters or a different head" from "the gradient"), the stored-token head b (reference), a seed twin.
- **Gate G1 (before any training):** `evaluation/drivers/e2e_gradient_report.py` extended to the route loss alone: the groups seeds, ray read-out, link head and junction head must show a nonzero gradient from the route loss in arm a, and none in arm c. If a group stays silent the code is wrong; that is a bug, not a result.
- **Cost / kill criterion:** coding 2 - 3 days, GPU about E232's. Kill: G1 fails; Q falls by more than 0.010 at one third of the steps; non-finite loss.
- **Inputs:** ready: the builder checkpoints, the SD library, the route-head code, `p16_interface.py` (contract), `e2e_gradient_report.py` (G1 base). To build: the node-wise route head with message passing, the soft target, the joint trainer wiring, the node-to-vertex read-out for the metric.
- **Starts from:** `e123_D7_r3_j1`; the champion (with recursion) only after a KEEP.
- **Referee:** not yet (`ar-critic`, mode cards, before it is scheduled). Machine lock (D19): the GPU arms wait for the bev_fusion trainings.

## 2. Detail plan
Written by the planner when the loop reaches the card.

## 3. Summary
Open.
