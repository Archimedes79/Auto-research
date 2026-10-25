# E239 (LATER card, blocked on the b456 loader) CONFIRM replication of the loop-18 fast-variant claims on 1,567 never-used frames

## 0. What the tools run (the fields of `experiment.json`)
| field | value |
|---|---|
| id / card / title | E239 / - / j1rc against its equal-budget control j1eq, and j1 against j1eq, on CONFIRM (39 new official-val scenes) |
| kind | variant (existing checkpoints, evaluation only; needs the step-7 loader and `evaluate.py --split confirm`) |
| deterministic | false (two training seeds per arm, already trained) |
| metric, direction, band | Q, higher, 0.010 |
| prediction | +0.004 (mean of two seed pairs), against arm c |
| arms | a: j1rc, j1rcs1 (recall weight 3); c: j1eq, j1eqs1 (weight 1, same budget); b: j1 (no fine-tune, report) ; p: f1 `champion` with pruning 0.3 / 0.4 against `champion_noprune` (report) |
| control | c |
| rule | lanes_found a - c >= +0.04; J a - c >= +0.02; Q a - c >= -0.005 (guard; a Q edge >= +0.005 is reported as a finding, not required) |
| budget_gpu_h, kill | 0.5 ; see card |
| evaluate | confirm (1,567 frames), compare c |

## 1. Card
| | |
|---|---|
| lens | past critic |
| group | A9 (evaluation, selection effects) |
| sources | D7 (split CONFIRM), `../../experiment_execution/plans/data_extension_b456.md` (steps 5 - 7, caveat on pipeline version), E219 (j1rc, j1rcs1, j1eq, j1eqs1, rc1, f1eq), E215, E231.a, D4 |
| status | proposed, blocked on step 7 (loader + `--split confirm`); do not schedule before |

- **Statement**: On CONFIRM (1,567 frames, 39 scenes, none used for any loop-18 decision), the fast recall variant j1rc keeps its lanes-found and J edge over the equal-budget control j1eq (lanes found >= +0.04, J >= +0.02) but not a Q edge (Q a - c < +0.005), as on VAL.
- **Mechanism / why**: about 35 runs and every threshold, weight and variant choice of loop 18 were made on the same 873 frames of 22 scenes; the noise band 0.010 is a spread between seeds, it does not cover selection on this set. The adopted variants (fast_recall, `recall`, pruning 0.3 / 0.4) stand on VAL only; the one TEST use (E230) did not cover j1rc / j1eq at all (D4). The protocol already reserves CONFIRM for this ("confirms a champion candidate before its TEST check"). Gains that were selected with the best of several variants should shrink on fresh data (winner's curse); gains that are structural (pruning removes false forks: J +0.144) should not. Telling the two apart is the point.
- **Prediction (VAL as reference, mean of two seed pairs)**: lanes found a - c +0.062 (VAL +0.062) stays >= +0.04; J a - c +0.030 stays >= +0.02; Q a - c about +0.004 (VAL +0.004 / +0.008 / +0.012 on the three pairings, TIE by decide.py: +0.0077). Report-only rows with their VAL values: j1eq - j1 Q +0.006 / +0.002; pruning on `champion` J +0.144 (expected >= +0.08), Q +0.034 (expected >= +0.02), F1@1 and lanes found unchanged.
- **Decision rule**: lanes found, J and the Q guard hold (KEEP): the claims transfer; the Q verdict stays TIE (|Q a - c| < 0.010 expected) and D4 is decided on lanes found / J / ms / robustness (E231.a: j1rc Q -0.038 / -0.026 with three cameras off in both seed pairs), not on Q. If lanes found a - c < +0.04 or J < +0.02 (CUT; also if Q a - c < -0.005): the recall-weight benefit was a selection effect, j1rc loses its place in D4. If Q a - c >= +0.005 with both pairs on the same side: the VAL TIE was too little power; say so. Noise band 0.010; INCONCLUSIVE if the two seed pairs differ by more than 0.010 in Q. Absolute CONFIRM numbers are NOT compared with VAL numbers (CONFIRM features come from today's pipeline, route share 89.7 vs 95.7 %, edges +26 %, plan caveat); only paired differences on CONFIRM frames are used.
- **Test design**: evaluate j1rc, j1rcs1, j1eq, j1eqs1, j1, f1, f1 without pruning on CONFIRM through new registry entries (a new model = a new registry entry; no change to champion / control entries) with fork pruning 0.4 (champion: also 0.3), standard assembly; `--compare` for paired scene bootstrap (39 scenes). Controls: matched reference and equal budget = j1eq / j1eqs1; seed twin = second pairs; null = j1; the control E94 has no graphs on CONFIRM and is not part of this card (its 628 ms per frame would be 16 min GPU; a later add-on only if needed). Fidelity rung: layer 4 on unseen frames. TEST is not touched (only the loop prompt can grant it).
- **Cost**: about 0.5 GPU h (raw network outputs of 7 models on 1,567 frames, then CPU assembly); coding 0.5 day for the b456 loader (step 7; a change of shared code with an ar-reviewer pass). Risk: the loader work, not the experiment.
- **Kill criterion**: stop if the golden check of the loader (a b123 VAL frame through the new path equals the stored path) fails; do not evaluate CONFIRM through a path that does not reproduce VAL numbers first (layer 1: `evaluate.py fast_recall` on VAL through the new loader = the stored card within 0.0005).
- **Inputs**: ready: checkpoints of all models, VAL cards, CONFIRM features and ground truth (E234.val, E235.gt, bit-reproduction checked in E235.chk3). To build: step 7 loader, `data/splits_b456.json` reader, `--split confirm` in `evaluate.py`, registry entries.
- **Referee** (2026-10-25, ar-referee): REVISE and LATER (blocked on the b456 loader, step 7; not tonight). Important and well argued (about 35 loop-18 decisions on the same 873 frames; CONFIRM is the intended pre-TEST check), but the machine rule does not implement the card: headline Q with only a max_loss clause means decide.py returns TIE whenever |dQ| <= 0.010 and never reaches the lanes_found / J clauses, so "KEEP = the claims transfer" cannot come out. Make lanes_found (band 0.02) the headline, keep J as a clause and Q as the guard, and write what happens if only one seed pair passes. Add: CONFIRM is a one-shot set (if it picks between variants it is spent as a selection set: say it is used only to confirm, with the single TEST check after); features are from a different pipeline version (route share 89.7 vs 95.7 %), so only paired differences count, as the card says; the first test is the layer-1 reproduction of fast_recall on VAL through the new loader.
