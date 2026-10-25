# Loop 17: plan (written 2026-10-24 16:50, before any run)

**Parameters.** The /loop prompt of 2026-10-24 (pasted again, same text as loops 13 - 16) named no stop time, no mail and no shutdown. Taken over from the last instruction of the user: STOP_AT = Sunday 2026-10-25 11:20 (the stop of loop 16), SHUTDOWN = no (withdrawn 2026-10-24 13:40), MAIL = no. Final report ready by 10:30. TEST (21 scenes) is NOT touched; the open request of loop 16b (one TEST run of the route head along the edges and the fast configurations) stays a decision for the user.

**Mission (user text, condensed).** Fully differentiable lane-graph chain (centre lines -> sparse graph -> temporal fusion -> map assignment -> navigation attribute to the planner), only differentiable candidates, concentrated on sparsification, temporal fusion and map matching; building blocks from the literature; systematic development and fine-tuning experiments; deep evaluation with hypotheses; result must be able to propagate a graph with navigation information into a planner; English report. Standing rules from the user: the method must be very good AND very fast (every experiment carries a time per frame), no red (non-differentiable) blocks in architecture drawings, every experiment has an architecture drawing + numbers + result images.

## Where loop 16 stopped (facts that drive this plan)

| fact | source |
|---|---|
| Champion f1 = D7 (pieces and links, teardrop fan, lattice seeds, geometric dedup, global context, warm start, junction head, time recursion, fork relocation, beam 288 / 144): VAL S 0.746 vs control E94 0.724, TEST S 0.728 vs 0.702; 13.5 vs 21.8 vertices per 100 m | loop 16 |
| Route head asked along the edges: VAL route success 0.889 (2 seeds) vs 0.798 vertex-based, control 0.885; consumer error at 20 m 0.26 m | loop 16b |
| Speed: 31 ms per frame (f1), 25 ms (j1 192 / 96) vs 628 ms for the control | loop 16b |
| Weak points: junction F1 J 0.26 vs 0.40 (shallow forks, geometry-only separation fails), lane recall 0.70 vs 0.80 | loop 16 / 16b |
| NEVER TESTED: the champion chain under sensor failure (night, blur, three front cameras off) and under map error with the new route head. Every earlier chain (E57, E105, E114, E118) was; the new one is the candidate for the E2E network, so its robustness is the open question that decides whether it can replace the control | this plan |
| Time recursion in the builder is a tie on clean frames (S 0.720 - 0.725 vs 0.72), but it was never tested where memory should matter (failure phase) | loop 16 |

## Hypotheses and experiments (ids E130 +; tags as in e2e_catalogue.py)

| id | question | hypothesis | design | success criterion |
|---|---|---|---|---|
| E130 | How much does the champion lose under failure? | It loses at least as much as E94 (night F1 0.34 vs clean 0.88), because both read the same 42 channels and the same dense head | champion f1 and the control on the VAL failure series (dark, blur, three front cameras off; first half of every scene clean, second half perturbed); S, F1@1, J, vertices, lanes found, route success with the route head along the edges, map error 0 / 2 m, ms per frame | numbers per phase and type, paired scene bootstrap |
| E131 | Does the training-free spatial hold of the input (E112: per pixel the decayed warped memory where its heat is higher) carry over to the builder? | yes: most of the loss of the chain under failure is lost input, not a weak builder (E114: night route 0.09 -> 0.44) | hold applied to the 42-channel input in front of the builder (decay 0.95); also EMA + monitor hold (E105) | failure-phase S and route success vs E130, clean-phase cost < 0.01 |
| E132 | Does the builder's own time recursion (prior from the previous frame's graph) help under failure, with and without E131? | it helps in the failure phase although it is a tie on clean frames (the prior is exactly a memory of lane structure) | champion with `temporal` on / off, with the previous frame fused (E131) or raw | failure-phase S difference with scene bootstrap |
| E133 | Can failure-frame training repair what the hold cannot (E118: fine-tune on failure series, carries partly to unseen types)? | yes for night and blur, not for three front cameras off | fine-tune the builder 1 - 2 u on failure frames of the VAL scenes that are NOT in the evaluation half (scene-disjoint split of the VAL scenes), clean frames mixed in (53 %) to bound the clean cost | held-out failure S, clean S change, other failure type left out |
| E134 | A learned temporal fusion (differentiable gate between warped memory and current input; GRU-like cell on the 42-channel input) vs the training-free hold | the learned gate can also switch off a bad modality and keep the clean one; it beats maxdecay on mixed failures | small conv gate g = sigmoid(conv(x, warp(mem), heat ratio)); y = g * warp(mem) + (1 - g) * x; trained on failure frames with the builder loss, builder frozen then unfrozen | failure-phase S vs E131, clean cost, parameters, ms |
| E135 | Map matching under failure with the new route head | the route head along the edges loses less than the vertex head because the edges carry the map-relevant geometry | route heads of loop 16b (2 seeds) on the E131 / E133 graphs, map error 0 / 2 m, wrong map, no map | route success per type, seed noise stated |
| E136 | Joint end-to-end fine-tune: builder + route head + consumer with route and planner loss through the graph (the gradient reach of the gate G4) | no gain on clean frames (E119, E121c), possible gain under failure where the loss can teach the builder what the route needs | 2 seeds, 1 u joint after the heads are trained separately | route success, consumer error, S; honest negative result is a result |
| E137 | Latency of the full robust chain | hold and gate cost < 3 ms | time per stage, idle GPU, batch 1, cached foundation features | ms per frame table |

Order and time budget (to STOP_AT 11:20, 18 h left at 17:00): harness p17_series.py (2 h) -> E130 / E131 / E132 (2 h) -> E133 / E135 (3 h) -> E134 (3 h) -> E136 (2 h) -> E137 + report with two review rounds (4 h). At most 3 GPU jobs. Two seeds for any route-head comparison (its noise is up to 0.03). Whatever is not finished by 08:30 is written up as open.

## Literature (parallel, agents)

Three agents write cards in the strict format of `report_final/text/lit16/SCHEMA.md` to `report_final/text/lit17/`: (A) streaming / temporal vector-map and lane-topology memory (query propagation, BEV memory, state-space memory, 2025 - 2026), (B) robustness of BEV perception and mapping to sensor failure plus test-time adaptation and efficiency, (C) SD-map / navigation conditioning, map matching and lane-level assignment inside E2E planners. Synthesis in `reports/Literature_Update_Loop17_EN.md`.

## Tracking

Live state: `out/loop2/e17/LOOP17_STATE.md` (phase, running jobs, next action, decision log, ledger rows). Code: `scripts/p17_*.py`, results `out/loop2/e17/<run>/`, cards `out/loop2/e17/cards/`. Every finished run goes into the experiment catalogue (`scripts/p17_catalogue.py` -> e2e_catalogue loader, ids E130 +). Report: chapter 9 added to `reports/Loop16_Differentiable_Lane_Graph_Report_EN.pdf` as `reports/Loop17_Robust_Differentiable_Lane_Graph_Report_EN.pdf` (builder `scripts/build_loop17_report.py`).
