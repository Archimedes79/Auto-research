# PLAN - experiment tree of the current loop (loop 21)

The loop-20 plan with all statuses is in `history/plans/PLAN_loop20.md` (N1 - N15 open list); loop 19 in `history/plans/PLAN_loop19.md`.

## Question of the loop
Mirko (2026-10-30, chat): a general loop to improve performance (quality and speed), freedom to choose the questions; "create a card and run the loop". Prompt and reading: `history/prompts/PROGRAM_loop21.md`. The GPU is held by the BEV-system chat's t2 lane (12 GB, 100 %), so the iterations are CPU-only on cached raw outputs and timing is relative.

## Tree
Each node: card -> (referee when it can change a decision) -> run through `run.py` -> decision by the written rule -> summary. VAL 873 unless stated. Ids E257 ff. are not yet in `project.json` "loops" (D17).

| node | experiment | hypothesis (short) | decision rule | cost | status (result) |
|---|---|---|---|---|---|
| I1 | E257 Douglas-Peucker output simplification | 40 % of the assembled vertices are interior points of straight runs; removing them (0.3 m) keeps Q and cuts vert100 by >= 3 | KEEP if vert100 -3.0 on both models and Q loss <= 0.003, F1@0.5 <= 0.005, topo F1 <= 0.005, J <= 0.01 | CPU 15 min | done (E257): KEEP, vert100 15.0 -> 9.0 at eps 0.3 (Q -0.001), eps 0.2 holds all guards on both models (9.9 / 9.5); D20 filed |
| I2 | E258 assembly profile and speed-up | the assembly (7.5 - 8.9 ms) has a stage above 15 % whose change saves >= 1.0 ms with identical graphs | CUT if no stage > 15 %; KEEP >= 1.0 ms relative, identical on 873 | CPU 30 min | done (E258): profile (relocation 21 %, anchoring 16 %); relocation stays (J -0.05); anchoring off holds on 4 of 4 j1rc seeds (dQ -0.0011 .. +0.0005, dJ -0.007 .. 0) -> registry `fast_recall_na`, saving ~1.3 - 2 ms estimated, idle repeat pending |
| I3 | results video of the best system (Mirko's request) | - | no rule (presentation) | CPU | done: `reports/weekly/Lane_graph_recall_val.mp4` (recall, 2 median scenes, 80 frames) |

## Candidates for I3 and for the next GPU window (not started)
- re-spend the freed vertex budget: after E257, is Q at 8 - 10 vertices per 100 m at the level of 15 (the curve moves left)? (secondary output of E257)
- seed ensemble of cached raw outputs (teacher ceiling for a distillation: lanes found and R@1 far of the mean of four j1rc seeds)
- GPU, when a slot is free: calibration step for the validity scores before thinning (K193 follow-up), the loss added next to the hinge, cheaper time recursion (+6.1 ms), fp16 / compiled builder with identical graphs
- do not repeat: Poisson loss swap, clamp 8 m, per-range thresholds, joint end-to-end fine-tune, recall weight above 3 (`closed_questions.md`)

## Controls and statistics
- layer 1 in every script: the stored cards are reproduced first; layer 0: identity / geometry bound asserted
- noise band 0.010 for Q, seed SD 0.0025; E257 and E258 compare the same raw outputs, so the seed noise does not enter, only the guards
- every claim names its vertex budget and the way vertices are counted (node ids or Douglas-Peucker)

## System items (framework)
| item | what | state |
|---|---|---|
| N1 / D19 | machine lease between chats | open (decision); loop 21 works around by CPU-only iterations while the other lane runs |
| N15 | `decide.py` cross-pairing cut test; `schedule.py add` duplicate ids | open (tools chat) |
| N16 (new) | `experiment.py check` reports NOT READY for cards that the run itself writes (E257); a card path that does not exist yet is not an error before the run | open |
| D20 (new, with E257) | vertex budget 14.3: count node ids or after output simplification? | file with the result |

## End of loop (P4)
- per node: result and decision (one line each) in the summaries; HANDOVER; `run_all_tests.sh`; `check_structure.py`; commit by path
- a report only if Mirko asks (week 4 report later)
