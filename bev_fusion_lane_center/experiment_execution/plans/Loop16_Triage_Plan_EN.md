# Loop 16 plan: systematic exploration and triage of fully differentiable lane-graph builders

Camera + radar BEV, lane centre-line graph with navigation route (nuScenes) · plan, 2026-10-23 · builds on `Growth_and_Link_Architectures_EN.md` (designs G, L, H) · nothing in this plan has been run yet

> **Reading guide.** Section 1 fixes what the plan does and the assumptions I made. Sections 2 - 4 say which designs enter and how they were chosen. Sections 5 - 8 are the triage machinery (schedule, scoring, refinement, loop protocol). Section 9 lists the weaknesses of the scheme itself. Costs are estimates from measured full-run times (E94 trace network 10 k iterations = 85 min alone, E120 8.4 k steps = 70 min, single RTX 3060 12 GB) and are re-measured in the gate step.

## 1. What the plan does, and the assumptions

- **Only fully end-to-end differentiable designs enter** (section 2). A design is admitted by a static and a dynamic test, not by a claim.
- **Search scheme = successive halving** with refinements (your rule): all designs are tried and refined once; from the second rung on the worst 50 % are removed and the rest continue with 4 refinements; this repeats until one design is left.
- **Assumption on "and so on".** I read it as: every rung after the first gives each survivor 4 refinements, while the training length of each run doubles per rung (1 u, 2 u, 4 u, 8 u; u = unit run). That is the usual successive-halving resource growth, but the *number* of design iterations stays 4. The alternative "4, then 16, then 64 refinements" is not affordable on one GPU (appendix).
- **Assumption on the pool.** 8 designs, chosen as an orthogonal array (section 3), so that one halving ladder 8 -> 4 -> 2 -> 1 fits; 16 designs would need about 10 h for rungs 0 - 2 alone (appendix).
- **Coverage design (added after the review of the experiment table).** The search space is covered in layers, each by an orthogonal array so that every factor level is tested equally often: L8 over the seven architecture factors A - G (rung 0), a forked L4 over three sparsification factors per survivor (rung 1), a forked L4 over three hypothesis / cascade factors (rung 2), time, joint training and the route / planner chain (rung 3). A coverage ledger (10.7) records which level was tested how often; nothing is left to chance.
- **A refinement** is one hypothesis-driven change set (at most two coupled parameters) applied to a design, followed by a continuation or re-training run and an evaluation. Section 6 says how it is chosen.
- **Consequence for the loop.** The full ladder needs about two nights (figure). Night A: harness, gate, rungs 0 and 1. Night B: rungs 2 and 3, joint end-to-end fine-tuning, confirmation on TEST. The plan keeps its state in a file, so the second night continues where the first stopped; the stop time and shutdown come from the `/loop` prompt.

![schedule](assets/p16_triage_schedule.png)

## 2. Admission gate: what "fully differentiable" means here

A design is admitted only if all four checks pass; failing designs are removed before rung 0 and are not part of the halving.

| check | how | pass criterion |
|---|---|---|
| G0 spec tests | the geometry, survival, selection, tree and merge code of the design passes the spec tests of `Hypothesis_Tree_Spec_and_Tests_EN.md` (T01 - T16, T21, T22; exact, as in the numpy reference) | all applicable tests pass |
| G1 static op scan | script scans the training path of the design for forbidden operations | none found outside label / input masks |
| G2 gradient reach | one batch, loss = geometry + validity + (link) + route + planner; backward; gradient norm per parameter group (stem / readers, seed and snap, pattern head, step or link heads, token heads, revisit, route head, consumer) | every group finite and non-zero |
| G3 planner reach | backward of the planner loss alone | non-zero gradient at pattern / snap / link parameters and at the validity head |
| G4 position sensitivity | add a small bias to the pattern output and to a seed position; compare autograd slope with finite differences | within 20 % |

Forbidden on the training path: argmax / argmin / argsort / round on predicted quantities, hard top-k or NMS on predictions, `torch.multinomial`, hard Gumbel, straight-through estimators, Hungarian assignment in the loss, `detach()` or `no_grad` on geometry that is meant to carry gradient. Allowed: bilinear sampling, softmax, sigmoid, expectation, masks built from inputs and labels, gathers of selected items with gradient through their scores, k-nearest candidate lists built from positions, the n best endpoints per tip with near-duplicate suppression and the global top-N_max on validity as a budget (all three selection only, flagged: gradient through the selected items, not through the choice; strictly soft alternative: sparsemax support), truncated backpropagation through the last 3 - 4 rounds (every parameter still receives gradient), top-k and thresholds at inference only.

Excluded from the pool and why:

| not admitted | reason |
|---|---|
| E94 as it is (argmax step, teacher forcing, peak seeds) | not differentiable at the decisions; kept as reference only |
| soft step on the fixed fan but with peak-picked seeds | seeds still discrete, positions of the start do not receive gradient |
| MapTR / LaneSegNet-style set decoder with Hungarian matching | assignment outside the gradient; E32 (matched set decoder) already lost 0.058 F1 on this data |
| Gumbel / straight-through step choice, policy-gradient growth | biased or high-variance gradient; the expectation over rays is the exact alternative |
| rule tracer, ICP registration, rule assembly | no gradient |

## 3. The solution space and the pool (L8 array)

Seven two-level factors span the designs G, L and their mixtures. A Taguchi L8 array picks 8 of the 128 combinations such that every level appears four times and every pair of levels twice, so main effects can be estimated from the 8 runs of rung 0.

| factor | level 1 | level 2 | what it tests |
|---|---|---|---|
| A construction | G: grow from tips | L: spray, snap, link | sequential against simultaneous building |
| B pattern package | fixed fan, +-45 deg at 2 / 4 / 6 m (E94-like 9 x 3), single expectation read-out | teardrop of the user sketch: full circle at 4 m, forward lobe to 24 m, shared rings, long rays verified by their samples (survival read-out), n = 3 best endpoints as hypotheses | whether the package pays; in L the same envelope is the candidate cone and the connector is verified by the survival product. The parts are separated later by flips (verification off, n = 1, far-ray arc re-aim as an extra) |
| C seeds | jittered lattice | k-means anchors | seed distribution |
| D duplicates | geometry only (density term + soft pointer / repulsion) | learned pair head | learned against geometric duplicate removal |
| E context | local attention (relative-pose bias, 12 m radius) | global graph-biased attention over all tokens | information exchange range |
| F init | free operation from step 0 | teacher-assisted warm phase (expert states for G, jittered ground-truth pieces for L), then free | exposure bias and slow starters |
| G revisit | none | E120-style revisit of the whole set | late correction |

| design | A | B | C | D | E | F | G | in words |
|---|---|---|---|---|---|---|---|---|
| D1 | G | fixed | lattice | geometry | local | free | none | simplest growth |
| D2 | G | fixed | lattice | learned | global | warm | revisit | growth with all context |
| D3 | G | teardrop | anchors | geometry | local | warm | revisit | adaptive growth, cheap context |
| D4 | G | teardrop | anchors | learned | global | free | none | adaptive growth, no help |
| D5 | L | fixed | anchors | geometry | global | free | revisit | pieces, fixed cone |
| D6 | L | fixed | anchors | learned | local | warm | none | pieces, learned duplicates |
| D7 | L | teardrop | lattice | geometry | global | warm | none | pieces, adaptive cone |
| D8 | L | teardrop | lattice | learned | local | free | revisit | pieces, adaptive, learned dedup |

- **Control arm R** (not a candidate, not halved): the existing chain E94 + E120 trained at each rung's fidelity with seeds 0 and 1. It gives the noise band, the fidelity curve (how a known system scores after 1 u, 2 u, 4 u, 8 u, 15 u) and the benchmark.
- **Hybrid H** is not in the array because it needs both parts. It plays a challenge match at the end of rung 1: the best G design and the best L design of rung 0 / 1 are recombined (L pieces, G growth from open ends, ray-or-piece softmax), warm-started from both parents, with 2 refinements at 2 u (4 u in total). If its S exceeds that of the lower-ranked of the two rung-1 survivors by more than delta, it takes that place; otherwise it is dropped. Rung 2 therefore always starts with 2 designs and the halving ladder stays 8 -> 4 -> 2 -> 1.
- **Implementation of G (and of the growth part of H): the verified-ray hypothesis tree** (`Hypothesis_Tree_Spec_and_Tests_EN.md`): b = 2 children per tip in a fixed-shape tree, a beam of M nodes per level, a cascade of hypothesis networks per level (A generate -> B refine -> C rescore, deep supervision), tree -> DAG by merge and edge split. In the trials of rung 0 only stage A runs; stage B, stage C, b = 3, per-level weights and the time recursion are refinement flips (section 6), the time recursion not before rung 3. A numpy reference with an oracle verification and 18 passing spec tests exists.
- **Literature defaults and baseline arms** (`Concept_Hypothesis_Tree_Lane_Graph_EN.md`, sections 5 - 6): from rung 0 the children are trained by multiple-choice learning with a score head and annealing (M3, K38), the survival loss treats rays leaving the BEV or entering occlusion as censored (M4, K41), and stages are not detached (K43). Beside the ladder, not halved: B0 control R, B1 geometric link on pieces (K30), B2 nodes + pairwise edges with verified edges (K48), B3 single-score line verifier on the same rays (K40). The mixes M1, M5, M6, M8, M9 are refinement flips; M2 is part of H.
- **Caveat of the array:** with 8 runs the main effects are aliased with two-factor interactions (resolution III). Supported effects are hypotheses; the refinement flips of section 6 are the confirmation.

## 4. Fidelity, budget and sizing

Unit run u: about 5 min of training plus evaluation on the fixed VAL subset, warm-started from the shared pre-trained stem. Full-length run = 15 u (70 - 85 min measured). Two jobs at a time at 1.5 x throughput (5 - 6 jobs oversubscribed the GPU in loop 15: 10 s per step).

| rung | designs | runs per design | fidelity | design runs (u) | control (u) | wall (h) |
|---|---|---|---|---|---|---|
| gate | 8 + R | smoke 100 iterations | - | - | - | 0.5 |
| 0 | 8 | trial + 1 refinement | 1 u | 16 | 2 | 1.0 |
| 1 | 4 (+ H challenge) | 4 forked refinements (L4 array, section 6) | 2 u | 32 (+ 4) | 4 | 2.0 (+ 0.1) |
| 2 | 2 | 4 forked refinements (L4 array, section 6) | 4 u | 32 | 8 | 2.2 |
| 3 | 1 | 4 refinements, joint fine-tune, 2-seed confirmation | 8 u, 15 u | about 76 | - | 4.2 |

Rung 3 joint fine-tune: dense head (live from the cached features), builder, route head and consumer are trained together on the sum of geometry, validity, link, route and planner losses; the gate checks G2 and G3 are re-run with the dense head in the list of parameter groups.

Refinements are continuations from the parent checkpoint wherever the change allows it (loss weights, schedules, extra inner pass, revisit on, regularisers); structural changes (pattern ladder, seed type, dedup type) restart from the shared warm start and are limited to one per design and rung. A finalist therefore accumulates about 26 u (130 min) of training before rung 3.

## 5. Scoring, cut and decision rules (fixed before rung 0)

**Evaluation sets.** Triage uses VAL only: a fixed subset of 250 frames (every 3rd or 4th frame, same for all designs) at rungs 0 - 1, the full VAL (873 frames) from rung 2. TEST (833 frames, 21 scenes) is touched once, for the rung-3 winner and the control, because 21 scenes are too few for repeated selection.

**Metrics per run** (graph evaluation of `p10_infer`): F1 at 0.5 / 1 / 1.5 m, precision, recall, lanes found, false length fraction, vertex error (mean, median, along / across), split and merge F1 (6 m window), vertices per 100 m of lane, duplicate rate, validity calibration, drift (vertex error against position in the chain, G only), hypothesis survival rate and frontier size per round (G only), candidate recall and link precision / recall (L only), cost in seconds, stability flags.

**Composite (the rank key):** `S = 0.30 F1@1 + 0.20 F1@0.5 + 0.20 J + 0.15 Sp + 0.15 (1 - FL)` with J = mean(split F1, merge F1), Sp = min(1, 14.3 / vertices per 100 m) (14.3 = Douglas-Peucker oracle at 0.3 m), FL = false length fraction. Weights follow the three weaknesses of B (position accuracy, junction topology, sparsity) and are fixed here; per-metric ranks are always shown beside S.

**Reference values (measured):** VAL F1@1 / F1@0.5 / lanes found / false length: E94 0.883 / 0.697 / 0.800 / 0.062; E120a 0.893 / 0.726 / 0.813 / 0.057 (vertex error 0.39 m). TEST: E94 0.851 / 0.661, E120a 0.863 / 0.694. Split / merge F1 0.36 / 0.46 (E94) against 0.69 / 0.65 (rule system, final graph). 22 vertices per 100 m (E94). The control arm re-computes S at every fidelity; S is only compared within a rung.

**Noise band delta.** From the two seeds of the control arm: delta = 2 x the spread of S (route-head seed noise was up to 0.03; the builder's own noise is unknown and is the first thing measured).

**Cut rule.** At the end of a rung rank by S (VAL, same fidelity); remove exactly floor(n / 2) designs.
1. *Kill switch (before ranking):* NaN, divergence, or F1@1 below 0.5 after the rung's last refinement removes a design; if more than half fail, no further cut at this rung.
2. *Ties inside delta* are decided by lower cost per run, then by the larger gain of the last two refinements.
3. *Metric-leader protection (rung 0 only):* the best design on J, on Sp or on F1@0.5 is not cut if it would otherwise fall in the bottom half; the next-worst design is cut instead (screening must not discard a specialist on one noisy ranking).
4. *Plateau check:* if the control arm's fidelity curve is still rising by more than delta between the last two fidelities, the cut is made but the rung's survivors list is annotated "fidelity-limited" and the next rung keeps the shared warm start.

**Main-effect analysis (rung 0).** For each factor the mean of S (and of each metric) at level 2 minus level 1 over the four runs per level; standard error about 0.71 x sigma of the control; "supported" if |effect| > 2 SE. Written to the state file and used by the refinement rule F.

**Success criteria for the final design** (all on the control's fidelity-matched comparison and then on TEST): S above the control by more than delta, and at least two of: mean split / merge F1 >= 0.50 (E94 0.41); vertices per 100 m <= 17 (E94 22); lanes found +0.03 over the control at equal false length. Then the route head (two seeds, as in E104b) and the consumer E117 run on the output, planner loss on and off.

**Sparsity-quality frontier (reported for every run, not part of S).** The loop looks for good sparsification, and a single composite hides the trade-off. For every run the validity threshold is swept (0.1 ... 0.9) and the curve F1@1 against vertices per 100 m is stored; scalars read from it: `F1@Sp17` = F1@1 at 17 vertices per 100 m or fewer (E94: 22; Douglas-Peucker oracle 14.3; the Imai-Iri minimum over valid chords is lower still: 13.2 against 14.1 per 100 m for Douglas-Peucker on 60 test frames at 0.3 m, `scripts/oracle_vertices.py`; the greedy longest chord equals the Imai-Iri count there, so the floor is about 13), `FL@R85` = false lane length fraction at lane recall 0.85 (the fixed-error-budget view of RoadTracer, K14; "better short than false" in one number), and the vertex count at equal F1@1 against the control. A run is on the frontier of its rung if no other run has both fewer vertices per 100 m and a higher F1@1; frontier points go to 10.9 and into the report figure.

## 6. Refinement protocol

Order of priority for the next refinement of a design (the first rule that applies):

| type | trigger | action |
|---|---|---|
| S stabilise | loss spike, NaN, gradient-norm ratio between loss terms above 100 | lower learning rate, clip, longer warm phase |
| F factor flip | main effect supported and the design is on the worse level, flip feasible | flip that factor (warm if possible), the flip is the confirmation of the effect |
| D diagnosis | worst diagnostic below | action from the table |
| H hyper-parameter | none of the above | sweep the most sensitive weight one step (position loss, density, validity gamma) |

Diagnosis table:

| diagnostic (automatic, every run) | action |
|---|---|
| drift grows along the chain (G) | more rounds in the backprop window, larger own-state share, recurrent token supervision |
| along-lane error much larger than across | step-length target (Douglas-Peucker rule, 0.3 m), vertex-count penalty |
| lane recall low for short lanes | more seeds or lower validity threshold, focal gamma up |
| false length or duplicate rate high | density weight up, learned pair head, repulsion step |
| split / merge F1 low, link recall low | image samples along the connector, one more link iteration, path-level term |
| candidate recall low (L) | wider cone, larger k |
| hypotheses multiply (frontier at the budget, duplicate hypotheses) | NMS radius up, n down, coverage merge, sparsemax support |
| corner cutting or too many vertices (sag), long steps fail | verification sigma and sample spacing on long rays, tolerant variant for gaps |
| forks missed (split recall low) | b = 3 instead of 2, lower prune threshold |
| children off the lane by 0.5 - 1 m | switch on stage B (refine) |
| false hypotheses alive, sibling conflicts | switch on stage C (rescore with tree attention) |
| curved lanes cut short (tree runs out of levels) | more levels D or a larger beam M |
| vertices per 100 m high | step-length target, sparsity penalty |
| validity badly calibrated | focal loss, temperature |

**Rung 0:** one refinement per design, chosen by the priority table and the diagnosis table above (types S, F, D, H), from the trial.

**Rungs 1 and 2: planned forks instead of a free chain.** The four refinements of a survivor are four *parallel branches* from the same parent checkpoint, laid out as an orthogonal array L4 over three two-level factors; each branch trains for the rung's fidelity, the branch with the best S (rules of section 5) becomes the parent of the next rung, the other three are kept in the ledger with their scores. Cost is unchanged (4 runs x fidelity), all branches have had the same extra training, so no control correction is needed, and the effect of every factor level is measured in the same rung on the same design. The branches of the four survivors are pooled for main effects (8 runs per level at rung 1, 4 at rung 2).

| branch | factor 1 | factor 2 | factor 3 |
|---|---|---|---|
| b1 | level 1 | level 1 | level 1 |
| b2 | level 1 | level 2 | level 2 |
| b3 | level 2 | level 1 | level 2 |
| b4 | level 2 | level 2 | level 1 |

| rung | factor | level 1 | level 2 | basis |
|---|---|---|---|---|
| 1 | SPF1 step-length rule | emergent from the verification only (the longest chord that stays on the lane) | plus an expected-vertex-count penalty with a Lagrangian pull to a target density (Douglas-Peucker 14.3 per 100 m on TEST, the Imai-Iri floor is about 6 % lower) | K02 K25 K129 K134 K142 K144 K121 |
| 1 | SPF2 validity / keep gate | validity = survival x direction prior | plus a learned keep head with a hard-concrete gate and an expected-L0 penalty (the E120e keep gate as reference); the DAG-merge edges get the same gate with an expected-kept-edges term | K129 K135 K134 K15 K09 |
| 1 | SPF3 reach of the ray ladder | rings to 24 m | rings to 12 m (E94 reached 6 m; the oracle chord is 7 - 8 m) | K01 K25 |
| 2 | HF1 stage B (refine) | off | on (small teardrop around every child, curve-aligned reads) | K07 K22 K36 |
| 2 | HF2 stage C (rescore) | off | on (attention over siblings, ancestors, committed points; no detach) | K22 K43 K44 K34 |
| 2 | HF3 hypotheses per step | b = 2, hard top-b | b = 3 with the MCL training of the children (K38); sparsemax support as the soft-selection ablation in a later loop | K03 K38 K39 |

For L designs the factors are applied analogously (SPF1: spacing target of the relaxation step; SPF3: maximum link gap; HF1: connector refinement; HF2: graph-biased iterations; HF3: candidates per piece end). A fork that fails the gate or diverges is replaced by the next row of the array only if at least two branches of the survivor remain; otherwise the survivor is carried with the remaining branches and the loss of coverage is written into 10.7.

**Rung 3:** diagnosis-driven again: time recursion on (K06 K24), joint end-to-end fine-tuning with the dense head, route head and consumer, map-fault and failure protocols (E87, E118 recipes), two seeds.

Rules for all rungs: a branch that diverges or falls below the floor is a KILL row, not an omission; two refinements of the same family with gains below delta are reported as "no effect" (that is a result for the report); the diagnosis table stays the guide for rung 0 and rung 3 and for choosing which survivor level to flip further in a later loop.

## 7. Loop protocol (what each wake-up does)

State lives in `out/loop2/e123_ladder/TRIAGE_STATE.md` plus one JSON card per run (experiment ids: E122 = control arm, E123.Dk.rj.nn = design k, rung j, refinement nn).

1. Read the state file (phase, rung, designs alive, runs queued / running / done, pids).
2. Parse finished runs, write the triage card (metrics, diagnostics, flags, S, rank), update the ledger (lineage: parent run, refinement type and change).
3. If the rung is complete: run the cut (section 5), write the main-effect table, create the next refinement for every survivor by the rules of section 6.
4. Launch at most 2 GPU jobs (wrapper scripts, Start-Process wsl as in loop 15), check memory below 11 GB; write pids to the state file.
5. Schedule the next wake-up (20 - 25 min), same prompt; at the stop time of the prompt write the report and shut down; mail only on explicit instruction.

**At the end of every rung** the loop writes, besides the cut log, (a) the coverage ledger 10.7, (b) the pooled effects table 10.8, (c) the frontier points 10.9, (d) `out/loop2/e123_ladder/rung_<r>_summary.md` with the ladder row of every design and three figures (S per design and branch, frontier F1@1 against vertices per 100 m with the oracle line at 14.3, main effects with the noise band), and (e) the new catalogue entries: every finished run is appended to `scripts/e2e_catalogue.py` (ids E124 onward, tag E / P / C / D) and its basis (SP stage, BB, K ids) to `scripts/e2e_basis.py`, then `build_experiment_table.py` is run, so that `reports/Experiment_Table_EN.md` always states what was tried on which basis. New techniques are added to the register 10.0 (K ids) before they are used.

Already there: `p16_ref.py` (numpy reference of the hypothesis tree with an oracle verification) and `test_p16_spec.py` (18 spec tests, all pass). Scripts to write in the harness phase: `p16_builder.py` (unified builder with switches A - G; its geometry code must pass the spec tests), `p16_gate.py` (G1 - G4), `p16_train.py` (run one design at a fidelity, writes `result.json`), `p16_diag.py` (diagnostics), `p16_triage.py` (cards, cut, main effects, plan), `p16_report.py` (figures and tables for the final report). While the harness is written, the GPU runs the control arm and the fidelity curve (no new code needed).

Data hygiene: all designs share data order, subset and seed 0; the control uses seeds 0 and 1. The builder is trained from labels and cached features, so it does not read E94 graphs; the route head and consumer on top of a design are trained on out-of-fold builder outputs (the in-sample caveat of E120 / E121).

## 8. Outputs at the end

Every result in the report names its basis by the technique ids of section 10.0. Ladder table (designs, lineage, S per rung, cut decisions), main-effect table with the confirmation flips, gate results, fidelity curve of the control arm, the winner on VAL and TEST against the control, route and planner numbers with two seeds, qualitative frames (as in loop 15), and the list of limits. English, architecture drawing of the winner, numbers table, images, key images in chat.

Report skeleton and where each part comes from: (1) task and constraints; (2) building blocks (table BB of `Experiment_Table_EN.md`, extended by the blocks found); (3) literature and register (10.0, `lit_p16_tree.md`, sparsification review); (4) earlier results as the starting point (the sparsification map of `Experiment_Table_EN.md`); (5) search design (L8, forks, coverage ledger 10.7); (6) results by rung (10.1 - 10.3, figures of the rung summaries); (7) effects of the sparsification and hypothesis factors (10.8) and the frontier (10.9); (8) the best design: architecture drawing, spec-test and gate results (10.6), VAL and TEST numbers against the control, route and planner numbers with two seeds, qualitative frames; (9) what did not work and why (KILL and CUT rows with reasons); (10) constraints and limits in short sentences; (11) next steps from the diagnosis. Every claim names its basis by K ids.

## 9. Weaknesses of the scheme

- **Low fidelity favours fast starters.** Free roll-out designs may look bad at 1 u; factor F and the shared warm start counter this, the control's fidelity curve measures it, and a plateau check annotates the cuts.
- **Noise.** The builder's seed noise is unknown until the control seeds finish; the first cut can be mostly noise. Hence the noise band, the tie rules and the metric-leader protection.
- **Array aliasing.** Resolution III confounds main effects with interactions; a wrong flip costs one refinement.
- **Unequal training.** Continuations give a design more training than a restart; the number of refinements is equal by rule, the control curve is subtracted.
- **Small data.** 212 training scenes (8,414 samples; 255 scenes in all, split 212 / 22 / 21); VAL subsets of 250 frames; TEST of 21 scenes used once.
- **Implementation risk.** Eight variants of a new builder in one harness is a lot of code; the gate step is where bugs surface, and a design that cannot be made to pass is replaced by the neighbouring row of a second array instead of being patched.
- **Fixed GPU.** One RTX 3060; two jobs at a time; the throughput factor 1.5 is an assumption.
- **Scope.** Exact poses and a simulated SD map, as in all earlier loops.

## 10. Tracking: results by round

The tables below are the template; the loop keeps the live copy in `out/loop2/e123_ladder/TRIAGE_STATE.md` and fills one rung per round. Every decision of section 5 leaves a row in 10.2 and 10.3, so a design's path (score, refinements, continued or cut, and why) can be read off one line of 10.1.

### 10.0 Technique register (the basis of every design and run)

Every design, refinement and run names the techniques it rests on by these ids, so that the report can state what was tried on which basis. Links only where the source was opened or seen in a search during this work; "from memory" = not re-checked. K30 - K52 come from the literature review `report_final/text/lit_p16_tree.md` (2026-10-23); the concept `reports/Concept_Hypothesis_Tree_Lane_Graph_EN.md` maps them to the mixes M1 - M12.

| id | technique | source | status | used for |
|---|---|---|---|---|
| K01 | teardrop pattern: short rays in all directions, long narrow forward rays, samples on shared rings | user sketch 2026-10-23; `Hypothesis_Tree_Spec_and_Tests_EN.md` M1 | internal | factor B level 2 |
| K02 | verification along a ray, survival / discrete-time hazard (same algebra as volume-rendering transmittance) | user 2026-10-23; spec M2; NeRF (Mildenhall et al., ECCV 2020) | NeRF from memory | factor B level 2 |
| K03 | n best endpoints as hypotheses; next N vertices with validity in a local window | user 2026-10-23; [CenterLineDet](https://arxiv.org/pdf/2209.07734) | checked | factor B level 2, tree |
| K04 | deformable sampling at predicted offsets, reference points refined per layer | [Deformable DETR](https://arxiv.org/pdf/2010.04159) | checked | pattern head, snap, factor E |
| K05 | anchor queries whose extent modulates the attention | [DAB-DETR](https://arxiv.org/pdf/2201.12329v4) | checked | snap of L (oriented boxes) |
| K06 | sparse instances with fixed + learnable keypoints, recurrent instance propagation with ego compensation | [Sparse4D](https://arxiv.org/pdf/2211.10581), [v3](https://arxiv.org/pdf/2311.11722) | checked | factor C level 2 (anchors), time M6 |
| K07 | look-up at the current estimate, update, repeat | [RAFT](https://arxiv.org/pdf/2003.12039) | checked | inner loop, stage B |
| K08 | expectation over a softmax (soft-argmax, integral regression) and its bias | [Integral Human Pose Regression](https://alphaxiv.org/abs/1711.08229); [Gu et al. ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Gu_Removing_the_Bias_of_Integral_Pose_Regression_ICCV_2021_paper.pdf) | checked | factor B level 1 (single expectation) |
| K09 | learned duplicate removal as a relation module | [Relation Networks for Object Detection](https://arxiv.org/pdf/1711.11575) | checked | factor D level 2 |
| K10 | lane segments + topology head | [LaneSegNet](https://proceedings.iclr.cc/paper_files/paper/2024/hash/06d86daa7001b7ea755ac4c80dc93497-Abstract-Conference.html) | checked | factor A level 2 (L) |
| K11 | DAG of lane nodes with continue / fork / merge states | [DAGMapper](https://arxiv.org/pdf/2012.12377) | checked | factor A level 1 (G), tree |
| K12 | path-level supervision instead of pieces | [LaneGAP](https://arxiv.org/pdf/2303.08815v3) | checked | path term of L |
| K13 | vertex and edge maps, GNN adjacency; local candidate edges | [InstaGraM](https://arxiv.org/html/2301.04470v3); SAM-Road (read in loop 8, `loop2/STATE.md`) | checked / internal | candidate lists of L |
| K14 | iterative graph growth guided by a CNN decision | [RoadTracer](https://arxiv.org/pdf/1802.03680) | checked (abstract) | factor A level 1 (G) |
| K15 | matching-free point-to-polyline refinement of a traced graph | E120, loop 15 (`out/loop2/LOOP15_INDEX.md`); MapTR / MapTRv2 (`report_final/text/lit_extra15.md`) | internal | losses, factor G (revisit) |
| K16 | learned trace network: fan sampling, argmax steps, ring wavefront | E94, loop 10 (`report_final/text/04e_loop10.md`) | internal | factor A level 1 lineage, factor B level 1 |
| K17 | route head (cross-attention to SD-map tokens), planner consumer | E104b, E117, loops 10 - 14 | internal | rung 3 chain |
| K18 | successive halving / Hyperband | Jamieson and Talwalkar 2016; Li et al. 2017 | from memory | triage schedule |
| K19 | orthogonal arrays (Taguchi L8) for factor screening | classical design of experiments | from memory | pool of designs |
| K20 | sparsemax (sparse, differentiable selection) | Martins and Astudillo 2016 | from memory | strictly soft selection variant |
| K21 | soft non-maximum suppression | Bodla et al. 2017 | from memory | factor D level 1, endpoint suppression |
| K22 | cascaded refinement with stage-wise losses | Cascade R-CNN (Cai and Vasconcelos 2018) | from memory | cascade stages A / B / C |
| K23 | scheduled sampling, DAgger (own states replace expert states) | Bengio et al. 2015; Ross et al. 2011 | from memory | factor F level 2 |
| K24 | log-odds fusion of evidence over time | occupancy-grid mapping (classical) | from memory | time M6 |
| K25 | Douglas-Peucker simplification (oracle vertex density) | classical (1973) | from memory | sparsity target, metric Sp |
| K26 | Hermite connector curve | standard spline | - | link head of L |
| K30 | geometric link prior: endpoint distance / heading -> learned link probability | [TopoLogic](https://arxiv.org/pdf/2405.14747) | checked | L link head, M2, baseline B1 |
| K31 | explicit endpoint detection, point-lane attention | [TopoPoint](https://arxiv.org/abs/2505.17771) | checked | M1 junction head |
| K32 | junction-aware queries, Bezier-space connection | [CGNet](https://arxiv.org/abs/2407.11337) | checked | M1 |
| K33 | fine-grained keypoint priors, topological denoising | [TopoFG](https://arxiv.org/abs/2511.12590) | checked | M1, M7 |
| K34 | joint predecessor / successor modelling | [UniTopo](https://arxiv.org/abs/2605.08911) | checked | M9, stage C |
| K35 | training-only topology sequence decoder | [Topo2Seq](https://arxiv.org/pdf/2502.08974) | checked (abstract) | M8 |
| K36 | lane / Bezier-aligned deformable attention | [LaneSegNet](https://arxiv.org/abs/2312.16108v1), [TopoBDA](https://arxiv.org/abs/2412.18951) | checked | M5 stage B |
| K37 | sampling along a lane prior, cross-level refinement | [CLRNet](https://arxiv.org/pdf/2203.10350), [LaneATT](https://arxiv.org/abs/2010.12035v2) | checked | M5 |
| K38 | multiple-choice learning with score head and annealing | [rMCL](https://arxiv.org/pdf/2311.01052), [aMCL](https://arxiv.org/abs/2407.15580) | checked | M3 (default) |
| K39 | entmax / sparsemax; soft top-k by optimal transport | [entmax](https://arxiv.org/abs/1905.05702v1), [soft top-k](https://papers.nips.cc/paper/2020/hash/ec24a54d62ce57ba93a531b460fa8d18-Abstract.html) | checked | soft-selection ablation (replaces K20 link) |
| K40 | line verification by pooling along the segment (LoI) | [L-CNN](https://arxiv.org/pdf/1905.03246), [HAWP](https://ar5iv.org/html/2003.01663) | checked | baseline B3 |
| K41 | discrete-time hazard / survival with right censoring | [survival review](https://arxiv.org/pdf/2305.14961) | checked via review, formulas from memory | M4 (default loss) |
| K42 | cascade with increasing thresholds (= K22, now checked) | [Cascade R-CNN](https://arxiv.org/pdf/1712.00726) | checked | M6 |
| K43 | look forward twice, contrastive denoising | [DINO](https://arxiv.org/pdf/2203.03605) | checked | M6, M7, no detach default |
| K44 | quality estimation head for calibrated confidence | [Sparse4D v3](https://arxiv.org/pdf/2311.11722) | checked | M6 |
| K45 | temporal map propagation and stream-query denoising, memory | [StreamMapNet](https://arxiv.org/pdf/2308.12570), [SQD-MapNet](https://arxiv.org/pdf/2401.09112), [MapTracker](https://arxiv.org/pdf/2403.15951) | checked | M11 time |
| K46 | stability metrics for online maps (mAS) | [Stability under scrutiny](https://arxiv.org/abs/2510.10660) | checked | time metrics |
| K47 | landmark and reachability metrics (forks, merges) | [RNTR](https://arxiv.org/pdf/2402.08207) | checked | J metric, report |
| K48 | nodes + pairwise edge classifier | [SAM-Road](https://arxiv.org/pdf/2403.16051), [Relationformer](https://ar5iv.labs.arxiv.org/html/2203.10202) | checked | M10, baseline B2 |
| K49 | map uncertainty consumed by the planner; ego-status caveat of open-loop metrics | [UncAD](https://arxiv.org/abs/2504.12826), [ego status](https://arxiv.org/pdf/2312.03031) | checked | M12, planner evaluation |
| K50 | restricted + free exploration (DAgger) for growers | [iCurb](https://arxiv.org/pdf/2103.17118) | checked | warm start option, factor F |
| K51 | generate candidates, then grow from dangling ends | [DeH4R](https://arxiv.org/abs/2508.13669) | checked | hybrid H |
| K52 | SD / satellite map prior for topology | [SMART](https://arxiv.org/pdf/2502.04329) | checked (abstract) | route head context |
| K60 | frozen camera-radar BEV features, detection-pretrained | CRN (Kim et al., ICCV 2023) [arXiv 2304.00670](https://arxiv.org/abs/2304.00670) | loop 1 foundation | E15 - E24 |
| K61 | frozen camera-radar BEV features, map-pretrained (Gaussian camera branch + point-transformer radar branch) | GaussianCaR (ICRA 2026) [arXiv 2602.08784](https://arxiv.org/abs/2602.08784) | loops 2 - 15 foundation | E27 onward |
| K62 | U-Net / FCN dense head | Ronneberger et al. 2015 [arXiv 1505.04597](https://arxiv.org/abs/1505.04597) | classic | E17, E37 dense head |
| K63 | ConvNeXt V2 encoder | Woo et al., CVPR 2023 [arXiv 2301.00808](https://arxiv.org/abs/2301.00808) | classic | E18 |
| K64 | temporal self-attention over ego-aligned BEV features | BEVFormer (Li et al., ECCV 2022) [arXiv 2203.17270](https://arxiv.org/abs/2203.17270) | from memory | E105 (as a fixed warp) |
| K65 | view lifting with depth distributions | Lift-Splat-Shoot (Philion and Fidler, ECCV 2020) [arXiv 2008.05711](https://arxiv.org/abs/2008.05711) | from memory | block BB06 |
| K66 | dataset | nuScenes (Caesar et al., CVPR 2020) [arXiv 1903.11027](https://arxiv.org/abs/1903.11027) | dataset | all experiments |
| K67 | fuser and decoder of the foundation: cross-modal fusion, dense prediction transformer, point transformer v3 | CMX; DPT; Point Transformer V3 [arXiv 2203.04838](https://arxiv.org/abs/2203.04838); [arXiv 2103.13413](https://arxiv.org/abs/2103.13413); [arXiv 2312.10035](https://arxiv.org/abs/2312.10035) | part of K61 | E27 |
| K68 | linear probing then full fine-tuning | LP-FT (Kumar et al., ICLR 2022; arXiv 2202.10054) [arXiv 2202.10054](https://arxiv.org/abs/2202.10054) | checked loop 6 (abstract) | E73, E76 |
| K69 | surgical fine-tuning of chosen layers | Lee et al., ICLR 2023; arXiv 2210.11466 [arXiv 2210.11466](https://arxiv.org/abs/2210.11466) | checked loop 6 (abstract) | E73 |
| K70 | weight-space interpolation of fine-tuned and original model | WiSE-FT (Wortsman et al., CVPR 2022; arXiv 2109.01903) [arXiv 2109.01903](https://arxiv.org/abs/2109.01903) | checked loop 6 (abstract) | E73 |
| K71 | model soups: averaging fine-tuned weights | Wortsman et al., ICML 2022; arXiv 2203.05482 [arXiv 2203.05482](https://arxiv.org/abs/2203.05482) | checked loop 6 (abstract) | E73, E76 |
| K72 | L2-SP: penalty towards the pre-trained weights | Li et al., ICML 2018; arXiv 1802.01483 [arXiv 1802.01483](https://arxiv.org/abs/1802.01483) | checked loop 6 (abstract) | E73 |
| K73 | SD map as raster canvas and vector tokens; noise-trained against shifted priors | TopoSD (Yang et al., arXiv 2411.14751) [arXiv 2411.14751](https://arxiv.org/abs/2411.14751) | checked loop 7 (html) | E78, E80 |
| K74 | navigation-map polylines by cross-attention, prior treated as noise, BEV dropout | NavMapFusion (Monninger et al., WACV 2026; arXiv 2512.03317) [arXiv 2512.03317](https://arxiv.org/abs/2512.03317) | checked loops 7, 14 (abstract) | E78, E87 |
| K75 | assignment of perceived lanes to SD roads (path-aware attention, beam search) | Online Navigation Refinement (Wan et al., arXiv 2507.07487) [arXiv 2507.07487](https://arxiv.org/abs/2507.07487) | checked loops 7, 14 | E77, E83, E104, E117, E119 |
| K76 | spatial alignment module and sensor/prior gate | TrajTopo (Jia et al., arXiv 2411.17161) [arXiv 2411.17161](https://arxiv.org/abs/2411.17161) | checked loop 7 (html) | E78, E80 |
| K77 | SD-map prior in lane-segment queries, denoising, past-frame consistency | Coherent Online Road Topology (Pham et al., IROS 2025; arXiv 2507.01397) [arXiv 2507.01397](https://arxiv.org/abs/2507.01397) | checked loops 14, 15 | E78 |
| K78 | global raster memory accumulated from past predictions | HRMapNet (Zhang et al., ECCV 2024; arXiv 2409.00620) [arXiv 2409.00620](https://arxiv.org/abs/2409.00620) | checked loop 7 (html) | E79, E80, E105 |
| K79 | propagate confident queries; fast and slow latent memory | TopoStreamer (arXiv 2507.00709); FASTopoWM (arXiv 2507.23325) [arXiv 2507.00709](https://arxiv.org/abs/2507.00709); [arXiv 2507.23325](https://arxiv.org/abs/2507.23325) | checked loops 7, 14 | E80, E118 |
| K80 | tri-mode training (no prior, previous prediction, map), displaced priors | Uni-PrevPredMap (Peng et al., RA-L 2026; arXiv 2504.06647) [arXiv 2504.06647](https://arxiv.org/abs/2504.06647) | checked loops 7, 8 (html) | E78, E80 |
| K81 | benchmark of camera / LiDAR corruptions for online map models | MapBench (Hao et al., NeurIPS 2024 D&B; arXiv 2406.12214) [arXiv 2406.12214](https://arxiv.org/abs/2406.12214) | checked loop 7 (html) | E57, E79, E80 |
| K82 | maps that stay usable with missing views or sensors: reconstruction, distillation, masked-input training, gating | SafeMap (arXiv 2507.00861); SEFMAP (arXiv 2602.21589) [arXiv 2507.00861](https://arxiv.org/abs/2507.00861); [arXiv 2602.21589](https://arxiv.org/abs/2602.21589) | checked loops 8, 14 (abstract) | E65, E118 |
| K83 | graph tensor encoding of road networks, single pass | Sat2Graph (He et al., ECCV 2020; arXiv 2007.09547) [arXiv 2007.09547](https://arxiv.org/abs/2007.09547) | checked loop 8 (abstract) | read, not used |
| K84 | graph grown vertex by vertex with a transformer, imitation with corrective expert actions | RNGDet / RNGDet++ (arXiv 2202.07824; 2209.10150) [arXiv 2202.07824](https://arxiv.org/abs/2202.07824); [arXiv 2209.10150](https://arxiv.org/abs/2209.10150) | checked loops 8, 10 | E92, E94 |
| K85 | global nodes then local repair; extended line for occlusion (SAM-Road++ is named in the loop-8 text but its arXiv id 2411.16733 is another paper: not cited) | GLD-Road (arXiv 2506.09553) [arXiv 2506.09553](https://arxiv.org/abs/2506.09553) | checked loop 8 and link check 2026-10-23 | E84 |
| K86 | lane topology as a chain of graph expansions (autoregressive, quantised) | SeqGrowGraph (Xie et al., ICCV 2025; arXiv 2507.04822) [arXiv 2507.04822](https://arxiv.org/abs/2507.04822) | checked loops 13, 15 | E46, E47, E113 |
| K87 | online map uncertainty consumed by a trajectory predictor | Gu et al., CVPR 2024; arXiv 2403.16439 [arXiv 2403.16439](https://arxiv.org/abs/2403.16439) | checked loops 8, 12 (abstract) | E109, E115 |
| K88 | planner with map queries into the motion stage, multi-stage training | UniAD (Hu et al., CVPR 2023; arXiv 2212.10156) [arXiv 2212.10156](https://arxiv.org/abs/2212.10156) | checked loops 8, 15 | E86b, E121 (interface) |
| K89 | vectorised scene representation; ego query cross-attends map queries | VAD (Jiang et al., ICCV 2023; arXiv 2303.12077) [arXiv 2303.12077](https://arxiv.org/abs/2303.12077) | checked loops 8, 15 | E66, E121 (interface) |
| K90 | sparse scene representation with anchor polylines | SparseDrive (Sun et al.; arXiv 2405.19620) [arXiv 2405.19620](https://arxiv.org/abs/2405.19620) | checked loops 8, 15 | E66, E121 (interface) |
| K91 | predicted map elements as scene-graph nodes for the planner | GraphAD (IJCAI 2025; arXiv 2403.19098) [arXiv 2403.19098](https://arxiv.org/abs/2403.19098) | checked loop 8 (html) | interface reference |
| K92 | dense probabilistic map plus route map for planning | MP3 (Casas et al., CVPR 2021; arXiv 2101.06806) [arXiv 2101.06806](https://arxiv.org/abs/2101.06806) | checked loop 8 (html) | interface reference |
| K93 | route compliance metric and counterfactual commands | NaviHydra / Closing the Navigation Compliance Gap (arXiv 2512.10660) [arXiv 2512.10660](https://arxiv.org/abs/2512.10660) | checked loops 8, 14 | E117d |
| K94 | goal point from a candidate set, then trajectory generation | GoalFlow (Xing et al., CVPR 2025; arXiv 2503.05689) [arXiv 2503.05689](https://arxiv.org/abs/2503.05689) | checked loop 14 (abstract) | idea for the consumer, not built |
| K95 | map features gated by ego state, trained jointly with the planner | arXiv 2509.13926 (MAP25) [arXiv 2509.13926](https://arxiv.org/abs/2509.13926) | checked loop 14 (abstract) | E115, E117 |
| K96 | camera-radar sparse queries for detection, tracking, mapping and planning | SpaRC-AD (arXiv 2508.10567) [arXiv 2508.10567](https://arxiv.org/abs/2508.10567) | checked loop 14 (abstract) | reference for the radar claim |
| K97 | planner tokens: lane and route-lane polylines | Diffusion Planner (arXiv 2501.15564) [arXiv 2501.15564](https://arxiv.org/abs/2501.15564) | checked loop 8 (html) | interface reference |
| K98 | prior fusion with log-confidence bias, presence gate and source dropout | UMPE (ICRA 2026; arXiv 2605.02762) [arXiv 2605.02762](https://arxiv.org/abs/2605.02762) | checked loop 8 (html) | E87, E104 (gate) |
| K99 | differentiable localisation of a BEV raster against a navigation map (pose volume, soft-argmax) | OrienterNet (Sarlin et al., CVPR 2023; arXiv 2304.02009) [arXiv 2304.02009](https://arxiv.org/abs/2304.02009) | checked loop 8 | E82, E103e |
| K100 | localisation to SD maps: regression, segmentation matching, homography | MapLocNet (2407.08561); BEV-Locator (2211.14927); SegLocNet (2502.20077); HOLO (2601.02730); U-ViLAR (2507.04503) [arXiv 2407.08561](https://arxiv.org/abs/2407.08561); [arXiv 2211.14927](https://arxiv.org/abs/2211.14927); [arXiv 2502.20077](https://arxiv.org/abs/2502.20077); [arXiv 2601.02730](https://arxiv.org/abs/2601.02730); [arXiv 2507.04503](https://arxiv.org/abs/2507.04503) | checked loop 8 (abstract / html) | E82, E103 (context) |
| K101 | global vector map with association and map NMS | GlobalMapNet (arXiv 2409.10063) [arXiv 2409.10063](https://arxiv.org/abs/2409.10063) | checked loop 8 (html) | E85 (context) |
| K102 | decayed, warped evidence; temporal consistency loss | PredMapNet (arXiv 2602.16669); MapTCL (arXiv 2608.05209) [arXiv 2602.16669](https://arxiv.org/abs/2602.16669); [arXiv 2608.05209](https://arxiv.org/abs/2608.05209) | checked loop 8 (html) | E79, E112 |
| K103 | lane graph from video: late graph merge against early BEV fusion | Can et al. (arXiv 2304.00930) [arXiv 2304.00930](https://arxiv.org/abs/2304.00930) | checked loop 8 (html) | E41, E105 |
| K104 | calibration and temperature scaling | Guo et al., ICML 2017 (On Calibration of Modern Neural Networks) [arXiv 1706.04599](https://arxiv.org/abs/1706.04599) | checked loop 12 (abstract) | E109 |
| K105 | calibration under dataset shift | Ovadia et al., NeurIPS 2019 [arXiv 1906.02530](https://arxiv.org/abs/1906.02530) | checked loop 12 (abstract) | E109 |
| K106 | BEV queries pulling from whatever sensors are present, trained for sensor failure | MetaBEV (ICCV 2023) [arXiv 2304.09801](https://arxiv.org/abs/2304.09801) | checked loop 12 (abstract) | E57, E108 |
| K107 | learning non-maximum suppression | Hosang et al., CVPR 2017 [arXiv 1705.02950](https://arxiv.org/abs/1705.02950) | checked loop 13 (abstract) | E116 |
| K108 | open-loop planning metrics and the ego-status caveat | AD-MLP (arXiv 2305.10430); BEV-Planner (arXiv 2312.03031); NAVSIM (arXiv 2406.15349) [arXiv 2305.10430](https://arxiv.org/abs/2305.10430); [arXiv 2312.03031](https://arxiv.org/abs/2312.03031); [arXiv 2406.15349](https://arxiv.org/abs/2406.15349) | checked loop 8 | E62, E66, E117 |
| K109 | map vector queries with hierarchical matching; per-layer refinement | MapTR (arXiv 2208.14437); MapTRv2 (arXiv 2308.05736) [arXiv 2208.14437](https://arxiv.org/abs/2208.14437); [arXiv 2308.05736](https://arxiv.org/abs/2308.05736) | checked loop 15 | E32, E120 |
| K110 | vectorised HD map as autoregressive polylines | VectorMapNet (Liu et al., ICML 2023; arXiv 2206.08920) [arXiv 2206.08920](https://arxiv.org/abs/2206.08920) | checked loops 10, 15 | E32 |
| K111 | DETR: set prediction with bipartite matching | Carion et al., ECCV 2020; arXiv 2005.12872 [arXiv 2005.12872](https://arxiv.org/abs/2005.12872) | checked loop 15 | E32 (context) |
| K112 | per-vertex offsets with circular convolution, iterated | Deep Snake (Peng et al., CVPR 2020; arXiv 2001.01629) [arXiv 2001.01629](https://arxiv.org/abs/2001.01629) | checked loop 15 | E120 (alternative) |
| K113 | numerical coordinate regression (DSNT) | Nibali et al., arXiv 1801.07372 [arXiv 1801.07372](https://arxiv.org/abs/1801.07372) | checked loop 15 | E103e (soft-argmax) |
| K114 | Gumbel-Softmax, Concrete, straight-through | Jang et al. 2017; Maddison et al. 2017; Bengio et al. 2013 [arXiv 1611.01144](https://arxiv.org/abs/1611.01144); [arXiv 1611.00712](https://arxiv.org/abs/1611.00712); [arXiv 1308.3432](https://arxiv.org/abs/1308.3432) | checked loop 15 | block BB21 (not used in training paths) |
| K115 | Chamfer distance as a point-set loss without assignment | Fan et al., CVPR 2017; arXiv 1612.00603 [arXiv 1612.00603](https://arxiv.org/abs/1612.00603) | checked loop 15 | E120 |
| K116 | differentiable rasterisation of vectorised maps | MapVR (Zhang et al., NeurIPS 2023; arXiv 2306.10502); Soft Rasterizer (arXiv 1904.01786) [arXiv 2306.10502](https://arxiv.org/abs/2306.10502); [arXiv 1904.01786](https://arxiv.org/abs/1904.01786) | checked loop 15 | E120 (alternative loss) |
| K117 | state-space memory for online maps | Mamba; MambaMap (arXiv 2507.20224) [arXiv 2312.00752](https://arxiv.org/abs/2312.00752); [arXiv 2507.20224](https://arxiv.org/abs/2507.20224) | checked loop 10 (search) | not tested |
| K118 | generated prior at BEV level before centre-line decoding; diffusion over vector maps | LaneDiffusion (arXiv 2511.06272); MapDiffusion (arXiv 2507.21423) [arXiv 2511.06272](https://arxiv.org/abs/2511.06272); [arXiv 2507.21423](https://arxiv.org/abs/2507.21423) | checked loops 8, 14, 15 | not tested |
| K119 | adaptive batch normalisation for test-time adaptation | AdaBN (Li et al.) | from memory, no link | E65 |
| K120 | ADASIS v2 horizon specification | ADASIS v2.0.1.0 (read in loop 7); v3 only vendor slides | checked loop 7 (specification) | E77 |
| K121 | dynamic number of pivot points per polyline | PivotNet (Ding et al., ICCV 2023) [arXiv 2308.16477](https://arxiv.org/abs/2308.16477) | link checked 2026-10-23 | sparsification reference |
| K122 | piecewise Bezier curves per map element | BeMapNet (Qiao et al., CVPR 2023) [arXiv 2306.09700](https://arxiv.org/abs/2306.09700) | link checked 2026-10-23 | sparsification reference |
| K123 | geometry-aware vectorised map construction | GeMap (Zhang et al., ECCV 2024) [arXiv 2312.03341](https://arxiv.org/abs/2312.03341) | link checked 2026-10-23 | reference |
| K124 | hybrid representation for vectorised maps | HIMap (Zhou et al., CVPR 2024) [arXiv 2403.08639](https://arxiv.org/abs/2403.08639) | link checked 2026-10-23 | reference |
| K125 | training and scaling practices of online vector map constructors | MapNeXt (arXiv 2401.07323) [arXiv 2401.07323](https://arxiv.org/abs/2401.07323) | link checked 2026-10-23 | reference |
| K126 | segmentation masks as queries for vectorised maps | Mask2Map (Choi et al., ECCV 2024) [arXiv 2407.13517](https://arxiv.org/abs/2407.13517) | link checked 2026-10-23 | reference |
| K127 | probabilistic planning over a vocabulary of trajectories; map tokens | VADv2 (arXiv 2402.13243) [arXiv 2402.13243](https://arxiv.org/abs/2402.13243) | link checked 2026-10-23 | interface reference |
| K128 | one-to-many auxiliary assignments in set-prediction detectors | H-DETR (Jia et al., CVPR 2023); Co-DETR (Zong et al., ICCV 2023) [arXiv 2207.13080](https://arxiv.org/abs/2207.13080); [arXiv 2211.12860](https://arxiv.org/abs/2211.12860) | link checked 2026-10-23 | reference for duplicates and one-to-many training |
| K129 | learned sparsity with hard-concrete gates and an expected L0 penalty | Louizos et al., ICLR 2018 [arXiv 1712.01312](https://arxiv.org/abs/1712.01312) | link checked 2026-10-23 | sparsity penalty candidate (SP4, SP6) |
| K130 | verified links for earlier from-memory ids: sparsemax (K20), Soft-NMS (K21), scheduled sampling and DAgger (K23), Hyperband (K18), NeRF (K02) | Martins and Astudillo 2016; Bodla et al. 2017; Bengio et al. 2015; Ross et al. 2011; Li et al. 2017; Mildenhall et al. 2020 [arXiv 1602.02068](https://arxiv.org/abs/1602.02068); [arXiv 1704.04503](https://arxiv.org/abs/1704.04503); [arXiv 1506.03099](https://arxiv.org/abs/1506.03099); [arXiv 1011.0686](https://arxiv.org/abs/1011.0686); [arXiv 1603.06560](https://arxiv.org/abs/1603.06560); [arXiv 2003.08934](https://arxiv.org/abs/2003.08934) | link checked 2026-10-23 | replaces the 'from memory' status of those ids |
| K131 | block references: spatial transformer, pointer networks, set transformer, FiLM, cost-volume networks, SetTransformer | Jaderberg 2015; Vinyals 2015; Lee 2019; Perez 2018; FlowNet; PWC-Net [arXiv 1506.02025](https://arxiv.org/abs/1506.02025); [arXiv 1506.03134](https://arxiv.org/abs/1506.03134); [arXiv 1810.00825](https://arxiv.org/abs/1810.00825); [arXiv 1709.07871](https://arxiv.org/abs/1709.07871); [arXiv 1504.06852](https://arxiv.org/abs/1504.06852); [arXiv 1709.02371](https://arxiv.org/abs/1709.02371) | link checked 2026-10-23 | building-block catalogue BB03, BB04, BB14, BB17 |
| K132 | multiple-choice learning, stochastic (diverse hypotheses) | Lee et al. 2016 (sMCL); see K38 for rMCL / aMCL [arXiv 1606.07839](https://arxiv.org/abs/1606.07839) | link checked 2026-10-23 | M3 children training |
| K133 | dataset-benchmark of lane topology | OpenLane-V2 (Wang et al., NeurIPS 2023 D&B) [arXiv 2304.10440](https://arxiv.org/abs/2304.10440) | link checked 2026-10-23 | external reference metrics |
| K134 | hard-concrete masks plus a Lagrangian pull to a target sparsity (prescribed density) | CoFi (Xia et al., ACL 2022; lit_p16_a SP02) [arXiv 2204.00408](https://arxiv.org/abs/2204.00408) | link checked 2026-10-23 | SPF1 / SPF2: hit 13 - 14 vertices per 100 m by a constraint, not one weight |
| K135 | learned edge keep gate with an expected-kept-edges penalty | PTDNet (Luo et al., WSDM 2021; SP03) [arXiv 2011.07057](https://arxiv.org/abs/2011.07057) | link checked 2026-10-23 | SPF2 keep gate, DAG-merge edges |
| K136 | input-dependent hierarchical token pruning trained through masking | DynamicViT (Rao et al., NeurIPS 2021; SP04) [arXiv 2106.02034](https://arxiv.org/abs/2106.02034) | checked (abstract) | keep decision trained through a mask |
| K137 | soft ranking and sparse structured selection: Gumbel top-k, differentiable sort, SparseMAP | Kool 2019; Blondel 2020; Grover 2019; Niculae 2018 (SP07 - SP10) [arXiv 1903.06059](https://arxiv.org/abs/1903.06059); [arXiv 2002.08871](https://arxiv.org/abs/2002.08871); [arXiv 1903.08850](https://arxiv.org/abs/1903.08850); [arXiv 1802.04223](https://arxiv.org/abs/1802.04223) | not verified (memory) | soft-selection ablations |
| K138 | learned encoder-token selection (top-rho tokens, scorer from decoder attention) | Sparse DETR (Roh et al., ICLR 2022; SP16) [arXiv 2111.14330](https://arxiv.org/abs/2111.14330) | link checked 2026-10-23 | selection scored by the consumer |
| K139 | supervised foreground selector with a cascaded keep ratio | Focus-DETR (Zheng et al., ICCV 2023; SP17) [arXiv 2307.12612](https://arxiv.org/abs/2307.12612) | link checked 2026-10-23 | over-complete first, prune late |
| K140 | dynamic number of queries from a predicted count | DQ-DETR (Huang et al., ECCV 2024; SP18) [arXiv 2404.03507](https://arxiv.org/abs/2404.03507) | checked (abstract) | count-driven hypothesis number |
| K141 | coarse-to-fine sparse pruning with k from data statistics | SparseOcc (Liu et al., ECCV 2024; SP19) [arXiv 2312.17118](https://arxiv.org/abs/2312.17118) | link checked 2026-10-23 | budget from oracle statistics |
| K142 | ground-truth compaction by Douglas-Peucker / Visvalingam (> 93 % fewer points, AP 97 - 100 % at 0.2 - 0.5 m) | MachMap report (arXiv 2306.10301; SP22) [arXiv 2306.10301](https://arxiv.org/abs/2306.10301) | link checked 2026-10-23 | simplification oracle as target |
| K143 | flexible-step road tracing | VecRoad (Tan et al., CVPR 2020; SP25) | not verified (page gave HTTP 403) | nearest adaptive-step tracer, unread |
| K144 | optimal polyline simplification: Imai-Iri valid-chord graph and shortest path versus Douglas-Peucker | van Kreveld et al., SoCG 2018 (SP26) [arXiv 1803.03550](https://arxiv.org/abs/1803.03550) | link checked 2026-10-23 | vertex floor: 13.2 per 100 m at 0.3 m (own count, 60 test frames) |
| K145 | sparse vertex set wired by Sinkhorn in training and exact assignment at inference; count-aware metrics (C-IoU, N ratio) | PolyWorld (Zorzi et al., CVPR 2022; SP29) [arXiv 2111.15491](https://arxiv.org/abs/2111.15491) | link checked 2026-10-23 | budget-aware metric template |
| K146 | learned task-driven edge removal with a per-node fan-in | NeuralSparse (Zheng et al., ICML 2020; SP28) | not verified (memory) | edge keep alternative |
| K180 | junction anchoring: a heat-map peak of the junction head forks (merges) the nearest edge and joins a free branch start within 8 m | own export step of loop 16 (scripts/p16_infer.py anchor_junctions); head after E44 key points, TopoPoint (K31) | own idea, tested in loop 16 (rung 3) | junction head + anchoring: J +0.03 (VAL250) |
| K181 | fork relocation: a fork vertex moves up its stem to the foot point of a strong junction peak if the straight branch edges stay within a tolerance | own export step of loop 16 (scripts/p16_infer.py relocate_forks) | own idea, tested in loop 16 (rung 3) | J 0.236 -> 0.278, S +0.007, fewer vertices (VAL250) |
| K182 | fork-coverage loss: the children of a tip before a true fork must cover the ground-truth points 10 - 16 m downstream (set coverage, min over the children) | own loss of loop 16 (scripts/p16_loss.py, weight fc) | own idea, tested in loop 16: negative | children end about 7 m ahead, the term stays at 1.87 of 2.07 |
| K183 | assembly-side fork recovery: admission of nodes down to a lower validity near junction peaks; sibling vertices kept apart near peaks (small reuse and snap radii) | own variants of loop 16 (p16_infer.assemble fork_tau, fork_vtx, fork_cov) | own idea, tested in loop 16: negative | J 0.15 - 0.23 against 0.236; more vertices |
| K184 | recall-weighted fine-tuning (soft-recall loss weight 3) of the champion | own variant of loop 16 (run rc1, validation only) | own idea, tested in loop 16 | lanes found against the control's 0.80 |
| K185 | route attribute along the edges: the route head is asked at points every <= 4 m of every edge (linear interpolation of the two vertex tokens, position on the edge, cross section) instead of at the vertices; a flagged piece of an edge is part of the route | own idea of loop 16b (scripts/p16_tok_resample.py, validation only) | own idea, tested in loop 16b: positive | route success of the sparse graph 0.798 -> 0.888 (control 0.885) |
| K186 | CUDA-graph capture of the builder forward (and of the time-recursion prior), one host transfer per frame, padded CUDA graphs of the route head | CUDA graphs (PyTorch documentation, from memory); own application in scripts/p16_runtime.py, p16_pipeline.py | own application, tested in loop 16b: positive | builder 69 -> 12 ms (time recursion), 6 ms (single frame); identical node arrays |
| K187 | vectorised union and anchoring in the assembly (growing vertex array, sorted edge list, shared maximum filters, vectorised fork relocation) with identical graphs | own optimisation of loop 16b (p16_infer.FastUnion, anchor_junctions_fast) | own, tested in loop 16b: identical graphs in 120 / 120 frames | assembly 50 -> 20 ms under load, 15 ms idle |

### 10.1 Ladder by round (one row per design, filled at the end of every rung)

Decisions: KEEP, CUT (reason), KILL (divergence or floor), PROTECTED (metric leader, rung 0), WIN; `fl` = fidelity-limited cut. S = composite of section 5 on VAL, rank in brackets. Basis = technique ids of 10.0 (factor levels of the L8 row plus the matching-free losses K15); refinements add their own ids in 10.2.

| design | short | basis | gate | rung 0: S trial / S ref 1 (rank) | decision | rung 1: S ref 1 / 2 / 3 / 4 (rank) | decision | rung 2: S ref 1 / 2 / 3 / 4 (rank) | decision | rung 3: S ref 1 - 4, joint, confirm | TEST F1@1 / J / Sp | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | G fixed lattice geo local free - | K04 K08 K11 K14 K15 K16 K21 | — | — | — | — | — | — | — | — | — | — |
| D2 | G fixed lattice learned global warm rev | K04 K08 K09 K11 K14 K15 K16 K23 | — | — | — | — | — | — | — | — | — | — |
| D3 | G teardrop anchors geo local warm rev | K01 K02 K03 K04 K06 K11 K14 K15 K16 K21 K23 | — | — | — | — | — | — | — | — | — | — |
| D4 | G teardrop anchors learned global free - | K01 K02 K03 K04 K06 K09 K11 K14 K15 K16 | — | — | — | — | — | — | — | — | — | — |
| D5 | L fixed anchors geo global free rev | K04 K06 K08 K10 K12 K13 K15 K16 K21 | — | — | — | — | — | — | — | — | — | — |
| D6 | L fixed anchors learned local warm - | K04 K06 K08 K09 K10 K12 K13 K15 K16 K23 | — | — | — | — | — | — | — | — | — | — |
| D7 | L teardrop lattice geo global warm - | K01 K02 K03 K04 K10 K12 K13 K15 K21 K23 | — | — | — | — | — | — | — | — | — | — |
| D8 | L teardrop lattice learned local free rev | K01 K02 K03 K04 K09 K10 K12 K13 K15 | — | — | — | — | — | — | — | — | — | — |
| H | best G x best L (challenge at the end of rung 1) | K01 K02 K03 K10 K13 K15 + parents | — | n/a | n/a | challenge: S ref 1 / 2 | — | — | — | — | — | — |
| R | control E94 + E120, seeds 0 / 1, never cut | K16 K15 | n/a | — | n/a | — | n/a | — | n/a | — | — | reference |

### 10.2 Run ledger (one row per run; the loop appends)

Run id `E123.<design>.r<rung>.<step>` (step 00 = trial, 01 ... = refinements), control `E122.s<seed>.<u>u`. Type: T trial, S stabilise, F factor flip, D diagnosis, H hyper-parameter. Basis = technique ids the change rests on (for a trial: the design's basis). dS is net of the control's gain for the same extra training.

| run id | design | rung | step | type | change and hypothesis | basis | parent | fidelity (u) | seed | wall (min) | F1@1 | F1@0.5 | J | Sp | FL | FL@R85 | F1@Sp17 | S | dS | flags | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E122.s0.1u | R | 0 | - | control | fidelity curve point | K16 K15 | - | 1 | 0 | — | — | — | — | — | — | — | — | — | - | — | planned |
| E122.s1.1u | R | 0 | - | control | fidelity curve point | K16 K15 | - | 1 | 1 | — | — | — | — | — | — | — | — | — | - | — | planned |
| E123.D1.r0.00 | D1 | 0 | 00 | T | trial as specified in the L8 row | K04 K08 K11 K14 K15 K16 K21 | - | 1 | 0 | — | — | — | — | — | — | — | — | — | - | — | planned |
| E123.D2.r0.00 | D2 | 0 | 00 | T | trial as specified in the L8 row | K04 K08 K09 K11 K14 K15 K16 K23 | - | 1 | 0 | — | — | — | — | — | — | — | — | — | - | — | planned |
| E123.D3.r0.00 | D3 | 0 | 00 | T | trial as specified in the L8 row | K01 K02 K03 K04 K06 K11 K14 K15 K16 K21 K23 | - | 1 | 0 | — | — | — | — | — | — | — | — | — | - | — | planned |
| E123.D4.r0.00 | D4 | 0 | 00 | T | trial as specified in the L8 row | K01 K02 K03 K04 K06 K09 K11 K14 K15 K16 | - | 1 | 0 | — | — | — | — | — | — | — | — | — | - | — | planned |
| E123.D5.r0.00 | D5 | 0 | 00 | T | trial as specified in the L8 row | K04 K06 K08 K10 K12 K13 K15 K16 K21 | - | 1 | 0 | — | — | — | — | — | — | — | — | — | - | — | planned |
| E123.D6.r0.00 | D6 | 0 | 00 | T | trial as specified in the L8 row | K04 K06 K08 K09 K10 K12 K13 K15 K16 K23 | - | 1 | 0 | — | — | — | — | — | — | — | — | — | - | — | planned |
| E123.D7.r0.00 | D7 | 0 | 00 | T | trial as specified in the L8 row | K01 K02 K03 K04 K10 K12 K13 K15 K21 K23 | - | 1 | 0 | — | — | — | — | — | — | — | — | — | - | — | planned |
| E123.D8.r0.00 | D8 | 0 | 00 | T | trial as specified in the L8 row | K01 K02 K03 K04 K09 K10 K12 K13 K15 | - | 1 | 0 | — | — | — | — | — | — | — | — | — | - | — | planned |

Refinement types and their usual basis: stage B refine K07 K22; stage C rescore K22 K04; b = 3 K03; sparsemax selection K20; tolerant survival K02; arc re-aim of far rays K01; time recursion K06 K24; path term K12; scheduled own states K23.

### 10.3 Cut log (one row per rung)

| rung | designs in | delta (noise band) | control S seed 0 / 1 | ranking by S | removed | rules used (kill, tie, protection) | fidelity-limited | finished at |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | — | — | — | — | — | — | — |
| 1 | 4 | — | — | — | — | — | — | — |
| 2 | 2 | — | — | — | — | — | — | — |
| 3 | 1 | — | — | — | — | — | — | — |

### 10.4 Main effects after rung 0 (L8 array)

| factor | level 1 (basis) | level 2 (basis) | effect on S (level 2 - level 1) | SE | supported (> 2 SE) | confirmed by flip run |
|---|---|---|---|---|---|---|
| A construction | G (K11 K14 K16) | L (K10 K13 K12) | — | — | — | — |
| B pattern package | fixed (K16 K08) | teardrop (K01 K02 K03) | — | — | — | — |
| C seeds | lattice (-) | anchors (K06) | — | — | — | — |
| D duplicates | geo (K21) | learned (K09) | — | — | — | — |
| E context | local (K04) | global (K04 K15) | — | — | — | — |
| F init | free (-) | warm (K23) | — | — | — | — |
| G revisit | - (-) | rev (K15) | — | — | — | — |

### 10.5 Control arm (fidelity curve and noise band)

| control | 1 u | 2 u | 4 u | 8 u | 15 u |
|---|---|---|---|---|---|
| S seed 0 | — | — | — | — | — |
| S seed 1 | — | — | — | — | — |
| spread | — | — | — | — | — |
| delta = 2 x spread | — | — | — | — | — |

### 10.6 Gate and spec tests per design

Spec tests from `Hypothesis_Tree_Spec_and_Tests_EN.md`; L designs run the applicable subset (pattern, verification, selection, seeds, warp: T01 - T09, T11, T12, T22), G designs all of T01 - T16, T21, T22.

| design | G0 spec tests | G1 op scan | G2 gradient reach | G3 planner reach | G4 finite differences | T17 - T20 | admitted |
|---|---|---|---|---|---|---|---|
| D1 | — | — | — | — | — | — | — |
| D2 | — | — | — | — | — | — | — |
| D3 | — | — | — | — | — | — | — |
| D4 | — | — | — | — | — | — | — |
| D5 | — | — | — | — | — | — | — |
| D6 | — | — | — | — | — | — | — |
| D7 | — | — | — | — | — | — | — |
| D8 | — | — | — | — | — | — | — |
| H | — | — | — | — | — | — | — |

### 10.7 Coverage ledger (which factor level was tested how often)

Planned runs follow from the arrays (L8 at rung 0: 4 per level; forked L4 at rung 1: 8 per level over four survivors; at rung 2: 4 per level over two survivors; time and joint at rung 3). The loop fills `done`, the designs and rungs, and the mean and best S of the level. At the end of the loop every row must have at least two finished runs or a written reason; the gaps are listed as limits in the report.

| factor level | planned runs | done | designs | rungs | mean S | best S | note |
|---|---|---|---|---|---|---|---|
| A1 G | 4 | — | — | — | — | — | — |
| A2 L | 4 | — | — | — | — | — | — |
| B1 fixed fan | 4 | — | — | — | — | — | — |
| B2 teardrop package | 4 | — | — | — | — | — | — |
| C1 lattice | 4 | — | — | — | — | — | — |
| C2 anchors | 4 | — | — | — | — | — | — |
| D1 geometry | 4 | — | — | — | — | — | — |
| D2 learned | 4 | — | — | — | — | — | — |
| E1 local | 4 | — | — | — | — | — | — |
| E2 global | 4 | — | — | — | — | — | — |
| F1 free | 4 | — | — | — | — | — | — |
| F2 warm | 4 | — | — | — | — | — | — |
| G1 none | 4 | — | — | — | — | — | — |
| G2 revisit | 4 | — | — | — | — | — | — |
| SPF1.1 survival only | 8 | — | — | — | — | — | — |
| SPF1.2 + vertex-count penalty | 8 | — | — | — | — | — | — |
| SPF2.1 survival x prior | 8 | — | — | — | — | — | — |
| SPF2.2 + learned keep (hard concrete) | 8 | — | — | — | — | — | — |
| SPF3.1 reach 24 m | 8 | — | — | — | — | — | — |
| SPF3.2 reach 12 m | 8 | — | — | — | — | — | — |
| HF1.1 stage B off | 4 | — | — | — | — | — | — |
| HF1.2 stage B on | 4 | — | — | — | — | — | — |
| HF2.1 stage C off | 4 | — | — | — | — | — | — |
| HF2.2 stage C on | 4 | — | — | — | — | — | — |
| HF3.1 b = 2 | 4 | — | — | — | — | — | — |
| HF3.2 b = 3 + MCL | 4 | — | — | — | — | — | — |
| T.1 time off | 1 | — | — | — | — | — | — |
| T.2 time on | 1 | — | — | — | — | — | — |
| J.1 staged training | 1 | — | — | — | — | — | — |
| J.2 joint fine-tune | 1 | — | — | — | — | — | — |

### 10.8 Pooled effects of the sparsification and hypothesis factors (rungs 1 and 2)

Effect = mean over the branches at level 2 minus level 1, pooled over the survivors; SE from the control spread (0.71 x sigma for differences of two means of equal size); supported if |effect| > 2 SE.

| rung | factor | level 1 | level 2 | effect on S | effect on vertices per 100 m | effect on FL@R85 | effect on F1@Sp17 | SE | supported | basis |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | SPF1 step-length rule | survival only | + vertex-count penalty | — | — | — | — | — | — | K02 K25 K129 |
| 1 | SPF2 gate | survival x prior | + learned keep (hard concrete) | — | — | — | — | — | — | K129 K15 K09 |
| 1 | SPF3 reach | 24 m | 12 m | — | — | — | — | — | — | K01 K25 |
| 2 | HF1 stage B | off | on | — | — | — | — | — | — | K07 K22 K36 |
| 2 | HF2 stage C | off | on | — | — | — | — | — | — | K22 K43 K44 K34 |
| 2 | HF3 hypotheses | b = 2 | b = 3 + MCL | — | — | — | — | — | — | K03 K38 K39 |

### 10.9 Frontier points per rung (sparsity against quality)

Points are (vertices per 100 m, F1@1) at the best validity threshold of each run; the oracle line is 14.3 vertices per 100 m (Douglas-Peucker, 0.3 m), E94 sits at 22 with F1@1 0.851 on TEST. A point is on the frontier if no run of the rung has fewer vertices and a higher F1@1.

| rung | run id | design | vertices per 100 m | F1@1 | F1@0.5 | FL@R85 | F1@Sp17 | on frontier | basis |
|---|---|---|---|---|---|---|---|---|---|
| ref | E94 (TEST) | R | 22.0 | 0.851 | 0.661 | — | — | — | K16 |
| ref | E120a (TEST) | R | 22.0 | 0.863 | 0.694 | — | — | — | K15 |
| ref | Douglas-Peucker oracle on ground truth | - | 14.3 | 1.000 | 1.000 | 0 | — | — | K25 |

## Appendix: sizing

Cost of rungs 0 - 2 = N x (2 + 4 + 4) = 10 N unit runs plus 14 u of control. Wall time = cost x 5 min / 1.5.

| pool N | rungs 0 - 2 (u) | wall | note |
|---|---|---|---|
| 4 | 54 | 3.0 h | halving 4 -> 2 -> 1; fewer factors (L4 array, 3 factors) |
| 8 | 94 | 5.2 h | this plan; rung 3 adds about 4.2 h |
| 16 | 174 | 9.7 h | two nights for rungs 0 - 2 alone; or u about 2.6 min (weaker ranking) |

Time compression for one night of about 7.5 h (2.25 h harness, 1 h triage overhead): u = 2.7 min would run rungs 0 - 3 without the 2-seed confirmation (140 u), with a higher risk of low-fidelity ranking; the recommended alternative is the two-night plan above. Alternative budget growth "4, then 16, then 64 refinements per rung": rung 2 alone would be 2 designs x 16 refinements x 4 u = 128 u, about 7 h.
