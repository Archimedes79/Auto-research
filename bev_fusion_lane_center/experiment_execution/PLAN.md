# PLAN - experiment tree of the current loop (loop 20)

The loop-19 plan with all statuses is in `history/plans/PLAN_loop19.md` (its section "What the system still needs", N1 - N12, is still the open list).

## Question of the loop
Mirko (2026-10-26, chat): when the other training is ready, do four iterations of automatic research here, test the methodology and the framework, improve the framework where necessary, report the week, then shut down. Prompt and reading: `history/prompts/PROGRAM_loop20.md`. The BEV-system chats' training (t1_r3) had finished, so the loop started at once.

## Tree
Four iterations, each: card -> referee (G2) -> run through `run.py` -> decision by the written rule -> summary. All numbers VAL 873, `evaluate.py`; the ids E252 ff. are not yet in `project.json` "loops" (D17), the loop table counts them under loop 19.

| node | experiment | hypothesis (short) | decision rule | cost | status (result) |
|---|---|---|---|---|---|
| I1 | E252 map offset of the ground truth by city | the 0.3 - 1 m map offset caps Q in Boston and shows as a lateral bias of correct predictions | CAP only if all Q gaps <= -0.03 and the |offset| ratio >= 1.5; NO EVIDENCE within +-0.010 | CPU 10 min | done (E252): cap NOT supported (Boston higher, +0.05; per-polyline offsets smaller in Boston); referee follow-up (null p 0.056, shift scale, per-polyline) done, verdict unchanged |
| I2 | E240 seed pair 3 (+ E241 pair 3) | completes the seed floor with a fourth pair | `decide.py --experiment E240` on four pairs | 2 x 20 min GPU | done (E240): TIE on four pairs, Q a - c +0.0062 (prediction +0.006 hit), lanes found +0.0596, seed SD Q 0.0030 / 0.0020, paired 0.0043; E241 KEEP as 'worth a 12-pairing replay' (cut test 4 of 4, 0 wrong) |
| I3 | E255 Poisson set loss, full schedule, matched vertex budget | the Poisson loss does not lift Q or lanes found at 14.3 vertices per 100 m | CUT <= -0.020 on one seed, KEEP >= +0.010 on both seeds and held-out frames | 20 min GPU + grids | done (E255): CUT, Q a - c -0.0726 at 14.3 vertices (prediction -0.045 missed by -0.028), topo F1 -0.127; K193 closed |
| I4 | E253 timing repeat with a machine guard | j1 is not a speed variant; recall costs +2.5 ms | session valid (5 of 6 SD <= 0.6 ms, stored values reproduced) then a - c <= -1.0 ms | 45 min exclusive | done (E253, 07:20 after the BEV chat's training ended): session valid, TIE, j1 - f1 -0.23 ms (prediction -0.3 hit), recall +2.65 ms, recursion +6.1 ms |
| I4b | E256 D4 candidates by city | the order of the five candidates holds in each city | flip = opposite order by > 0.010 in both cities | CPU 10 min | done (E256): no flip; fast_recall over fast is a Boston-only Q gain, lanes found +0.065 in both |

## System improvements (framework test, from the plan of loop 19 and new)
| item | what | state |
|---|---|---|
| N1 | no machine lease between chats | open (decision for the tools); guard in the timing driver done |
| N7 | timing protocol guard | done in `temp/loop20/run_e253.sh` (GPU, WSL load, Windows CPU, repair rounds) |
| N10 | exact map as loader default | done 2026-10-26 (D14) |
| N13 (new) | `project.json` "loops" (protected) must list each loop; stale planned rows of the previous loop remain in the schedule | D17 filed; schedule rows of loop 19 are still shown as planned |
| N14 (new) | matched-vertex evaluation had no code, a max over grid points favours the arm with more freedom | `temp/loop20/e255_matched.py` (interpolation at 14.3, stored-card reproduction, held-out frames): used in E255, works; promote to `evaluation/drivers/` when used a second time |
| N15 (new) | `decide.py` has no cross-pairing cut test (E241 was decided by hand) and `schedule.py add` accepts duplicate ids | open (tools chat) |
| D18 / D19 (new) | per-metric noise bands from four pairs; machine lease between chats | filed in DECISIONS |

## Controls and statistics
- the controls of E255 are the exact-map j1eq fine-tunes of E237 (same map, recipe and init); pair 3 of E240 is pinned to the stored map (`LCR_DIST_DIR`) so the four pairs stay comparable
- every claim names its vertex budget; noise band 0.010 until the SD table says otherwise (Q SD about 0.003)
- tuning of tau / vtx_r on val250 is declared, held-out frames are reported

## What the system still needs
See `history/plans/PLAN_loop19.md` (N1 - N12) plus N13 / N14 above and the report section 6 of `reports/weekly/Week3_Loops18-20_Report_EN.pdf`.

## End of loop (P4)
- per node: result and decision (one line each) in the summaries of E252, E240, E255, E253
- weekly report week 3 (loops 18 - 20): `reports/weekly/Week3_Loops18-20_Report_EN.pdf` built, numbers reviewer run; `check_structure.py` clean; full test suite; handover; E253 ran 07:20 - 07:38 on a quiet machine
