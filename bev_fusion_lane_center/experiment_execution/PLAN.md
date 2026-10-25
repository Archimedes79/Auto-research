# PLAN - experiment tree of the current loop (loop 18)

Written before the loop; a node's status changes, the nodes do not (nodes added during the loop are marked "added"). One row per run with prediction, rule, result and decision is in `LEDGER.md`; the details of an experiment are in `../experiments/E<id>_<slug>/`; the report is `../reports/weekly/Loop18_Fork_Pruning_Localisation_Report_EN.pdf`; the prompt is `history/prompts/PROGRAM_loop18.md`. Status of 2026-10-25 10:55: loop 18 CLOSED.

## Question of the loop
Can the failure-robust chain (champion D7 + hold + gate + map builder, 31 ms) keep its quality under sensor failure AND a wrong or shifted map, and get faster, with every module on the differentiable path?

## Tree (levels this loop owns: 2 = module idea, 3 = variant / ablation)

| node | idea (level 2) | hypothesis | gate (written before the run) | cost | status (result) | experiments |
|---|---|---|---|---|---|---|
| WP1 | finish loop 17: E209 unseen failure types | the hold / gate / map builder generalise to failure types not seen in training | numbers of E209 on VAL, 2 seeds, noise band 0.010 S / 0.07 route success | ~1 h GPU | done: the map benefit transfers (+0.04 ... +0.09 F1@1 over the no-map control) | E209.b2, E209.c2, E209.eval |
| R2.1 Loc | K188 pose-grid registration + K189 map trust in front of map builder B | with 1-3 m map error the corrected map restores the map benefit; a wrong map is switched off | pose RMS at sigma 2 < 1.0 m and < 1 deg; failure-half S >= no map + 0.015; wrong map within 0.01 of no map; exact map keeps >= 90 % of its gain | < 10 min GPU | done: pose error 2.5 -> 0.8 m, F1@1 +0.03 significant, S gate missed by 0.004, wrong map off at trust 0.75 (keeps 72 - 87 % of the exact map's gain) | E213.a, E213.b, E213.c |
| R2.2 J2 | joint fine-tune: builder + route head with failure rows (E214: 2400 steps, builder loss 0.05; E232: 4800 steps, builder loss 1.0) | route loss through the builder beats continued route-head training | route success failure half +0.03 over the control arm; graph S within 0.005 | ~2 h GPU | done, NEGATIVE three times (E202, E214, E232: a - c -0.07 ... +0.007); the joint E2E line stops; the failure-trained route head itself is the lever (+0.13 ... +0.37) | E214.a0 - c1, E232.a0, E232.c0 |
| R2.3 FastAsm | vectorised / compiled assembly (K191) | same graphs, fewer ms | identical graphs on 873 VAL frames; assembly < 8 ms | CPU | done: 873 / 873 identical, assembly 12.6 -> 7.0 ms, chain 26.1 -> 17.5 ms (j1 17.2 ms) | E216 |
| R2.4 GridE2E | unfreeze the dense head inside the builder training | graph loss into the dense head beats the modular pair | Q +0.01 at <= 14.3 vertices / 100 m | decide after R2.1-R2.3 | not run (the input stage is not the bottleneck, loop 17 E133a / E138) | |
| R2.5 Fork2 | second-successor head for forks | forks are missed | - | - | not run (38 % of the true forks are shallow, an annotation limit; the fork problem was precision) | |
| WP4 | deep evaluation of the weak points (J, lanes found) | missing lanes are range- or heat-limited | one test per weakness; honest negatives stay | CPU | done: far gap is partly a verifier trade (E217, E233), node validity calibration (E221), recall variants under failure (E231.a) | E217, E221, E231.a, E233 |
| R2.6 Prune (added) | K190 fork pruning with the junction-head heat | the champion's J deficit is false forks | J up to the control's level at the same F1@1 | CPU | done, KEEP: J 0.265 -> 0.409 (control 0.408), Q 0.701 -> 0.735 (control 0.737, tie), S 0.746 -> 0.769 | E215, E218 |
| R2.7 Recall (added) | recall-weighted fine-tunes under pruning and equal-budget controls | the weight closes the lanes-found gap | Q >= 0.745 with lanes found >= 0.78 (rc2 / rc4); j1rc: Q >= j1 + 0.005 and lanes found >= j1 + 0.04 | ~1 h GPU | done, MIXED: the weight buys lanes found (+0.06) and far recall, Q only partly (part of it is extra fine-tuning, saturates at 4 u); fast j1rc adopted on VAL, not robust with three cameras off (Q -0.038); rc1 stays the 288 / 144 recall variant | E219.rc1s1, rc2, rc3 (stopped), rc4, j1rc, j1rcs1, j1rc6, f1eq, j1eq, j1eqs1, j1eq8 |
| R2.8 Route (added) | failure-trained route head in the best chain; planner path error | route head trained on failure frames helps | - | - | done: +0.015 ... +0.06 in the best chain, no planner-level effect (floor 0.27 m) | E220, E222 |
| TEST (added) | the one TEST use of the loop | confirm the VAL picture | settings fixed on VAL | - | done once: pruned champion S 0.748 vs control 0.702, J 0.422, lanes found 0.653; rc1 + pruning Q 0.727; logged in `../evaluation/TEST_USAGE_LOG.md`; j1rc / j1eq NOT tested (needs the user's OK) | E230 |

## Controls and statistics of the loop
- matched reference on the same frames: control E94 and champion D7; null baseline = no map / no hold; seed twin for every route-head comparison (2 seeds); equal-budget controls for every fine-tune (E219)
- split: select on VAL (873, VAL250 for early rungs); TEST (21 scenes) once for the winner + control, logged in `../evaluation/TEST_USAGE_LOG.md`, only with the grant in the loop prompt
- report S together with Q (S without the sparsity term) and ms per frame; paired scene bootstrap for graph metrics

## End of loop
Update this table, `LEDGER.md`, `HANDOVER.md` (done 2026-10-25 10:55); next loop starts from a new plan (from `../../autoresearch/templates/experiment_plan.md`); this file then moves to the plans folder of history/ as PLAN_loop18.md.
