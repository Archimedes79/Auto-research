# PLAN - experiment tree of the current loop (loop 20)

The loop-19 plan with all statuses is in `history/plans/PLAN_loop19.md` (its section "What the system still needs", N1 - N12, is still the open list).

## Question of the loop
Mirko (2026-10-26, chat): when the other training is ready, do four iterations of automatic research here, test the methodology and the framework, improve the framework where necessary, report the week, then shut down. Prompt and reading: `history/prompts/PROGRAM_loop20.md`. The BEV-system chats' training (t1_r3) had finished, so the loop started at once.

## Tree
Four iterations, each: card -> referee (G2) -> run through `run.py` -> decision by the written rule -> summary. All numbers VAL 873, `evaluate.py`; the ids E252 ff. are not yet in `project.json` "loops" (D17), the loop table counts them under loop 19.

| node | experiment | hypothesis (short) | decision rule | cost | status (result) |
|---|---|---|---|---|---|
| I1 | E252 map offset of the ground truth by city | the 0.3 - 1 m map offset caps Q in Boston and shows as a lateral bias of correct predictions | CAP only if all Q gaps <= -0.03 and the |offset| ratio >= 1.5; NO EVIDENCE within +-0.010 | CPU 10 min | done: cap NOT supported (Boston higher, +0.05; per-polyline offsets smaller in Boston); referee follow-up (null p 0.056, shift scale, per-polyline) done, verdict unchanged |
| I2 | E240 seed pair 3 (+ E241 pair 3) | completes the seed floor with a fourth pair | `decide.py --experiment E240` on four pairs | 2 x 20 min GPU | running |
| I3 | E255 Poisson set loss, full schedule, matched vertex budget | the Poisson loss does not lift Q or lanes found at 14.3 vertices per 100 m | CUT <= -0.020 on one seed, KEEP >= +0.010 on both seeds and held-out frames | 20 min GPU + grids | queued behind E240.c3; controls' grids done |
| I4 | E253 timing repeat with a machine guard | j1 is not a speed variant; recall costs +2.5 ms | session valid (5 of 6 SD <= 0.6 ms, stored values reproduced) then a - c <= -1.0 ms | 45 min exclusive | last, after the trainings |

## System improvements (framework test, from the plan of loop 19 and new)
| item | what | state |
|---|---|---|
| N1 | no machine lease between chats | open (decision for the tools); guard in the timing driver done |
| N7 | timing protocol guard | done in `temp/loop20/run_e253.sh` (GPU, WSL load, Windows CPU, repair rounds) |
| N10 | exact map as loader default | done 2026-10-26 (D14) |
| N13 (new) | `project.json` "loops" (protected) must list each loop; stale planned rows of the previous loop remain in the schedule | D17 filed; schedule rows of loop 19 are still shown as planned |
| N14 (new) | matched-vertex evaluation had no code, a max over grid points favours the arm with more freedom | `temp/loop20/e255_matched.py` (interpolation at 14.3, stored-card reproduction, held-out frames) |

## Controls and statistics
- the controls of E255 are the exact-map j1eq fine-tunes of E237 (same map, recipe and init); pair 3 of E240 is pinned to the stored map (`LCR_DIST_DIR`) so the four pairs stay comparable
- every claim names its vertex budget; noise band 0.010 until the SD table says otherwise (Q SD about 0.003)
- tuning of tau / vtx_r on val250 is declared, held-out frames are reported

## What the system still needs
See `history/plans/PLAN_loop19.md` (N1 - N12) plus N13 / N14 above and the report section 6 of `reports/weekly/Week3_Loops18-20_Report_EN.pdf`.

## End of loop (P4)
- per node: result and decision (one line each) in the summaries of E252, E240, E255, E253
- weekly report week 3 (loops 18 - 20), fresh numbers reviewer, `check_structure.py`, full test suite at the end, handover, shutdown only if no other chat runs a job
