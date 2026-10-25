# HANDOVER - state of 2026-10-24 (evening) and what to do next

Update this file at every milestone; a new chat reads it first. **Start a new chat in `C:\ML-Experiment` (or `...\bev_fusion_lane_center`) and say: "Read HANDOVER.md and continue."** `CLAUDE.md` is loaded automatically.

## Where the work stands
| loop | what | state |
|---|---|---|
| 1-15 | foundation, dense heads, graphs, map / route, failure, planner consumer, differentiable positions | done, reported (`reports/`), one folder per experiment in `experiments/` |
| 16 | eight differentiable builders by successive halving; champion **D7** (+ junction head, time recursion, fork relocation). VAL S 0.746 vs control 0.725; TEST S 0.728 vs 0.702 (the one TEST use, logged) | done, report |
| 16b | validation only: route head asked along the edges 0.889 (vertex head 0.798, control 0.885); chain 31 ms per frame vs 628 ms (CUDA graphs, vectorised assembly) | done, in the loop-16 report; new TEST evaluation waits for an OK |
| 17 | robustness of the champion (night, blur, cameras off ...), hold / gate memory, map-conditioned builder, consumer under failure; catalogue ids **E193-E212** (internal names E130-E144 in `runs/e17/`) | all experiments done **except E209**; report draft exists; **code and state file were deleted by a rollback** |

What today's checks say (`evaluation/examples/RESULT.md`, `METRIC_RESPONSE.md`): the champion's S gain over the control (+0.021 [+0.011, +0.031] paired) comes entirely from sparsity (13.5 vs 21.8 vertices per 100 m; Sp is capped at 14.3). At equal sparsity (Q) it is 0.036 *worse*: J 0.265 vs 0.408, lanes found 0.695 vs 0.800, F1@1 0.870 vs 0.883. **The direction for the next experiments is quality at <= 14.3 vertices per 100 m (J, lanes found, route success), not more sparsity.** S is blind to parallel duplicates and to travel direction (see `evaluation/sparse_lanes/README.md`).

## Open items, in order
1. **Restore the loop-17 code** (deleted by a rollback). The replay of the transcript is in `experiment_execution/loop17_recovery/`: `tree/scripts/` (50 files, replayed Write / Edit / shell-patch calls; unverified: 23 edits did not apply in an earlier run, probable cause a skipped node patch at transcript line 4291 and duplicate request blobs, both fixed in `_recover3.py` but not rerun), `tree/out/loop2/e17/LOOP17_STATE.md`, `tree/reports/Loop17_Plan_EN.md`, `pyc/` (compiled originals of 9 modules for comparison), the replay tools (`_recover3.py` ...). Steps: rerun `_recover3.py` (sandbox, touches nothing), `py_compile` all, compare with `pyc/`, then copy into place: `p17_*` -> `training/models`, `training/train`, `evaluation/drivers` (by role, `training/README.md`), `build_loop17_report.py`, `loop17_text*.py`, `fig_p17_*.py` -> `reports/build/`, `LOOP17_STATE.md` -> `experiment_execution/indexes/`. The launchers in `temp/launchers/loop17/` already name the right scripts.
2. **E209 rerun** (unseen failure types: builder trained without night / blur rows, tested on night / blur). The first run crashed at ~3500 / 4800 steps with a CUDA 'unknown error' (GPU memory 11.6 / 12 GB, two trainings in parallel), `runs/e17/logs/e142.done` was a false flag (renamed `.FALSE`). Add checkpoint + resume to `p17_mapft.py` (use `training/trainer.py` or its pattern), run through `tools/experiment/run.py --gpu-max-mb 9000` (retry on CUDA errors), at most 2 trainings; launcher `temp/launchers/loop17/_run_e142.sh` (trainings `e17_map_b2`, `e17_map_c2`, then three evals).
3. **Finish the loop-17 report**: `temp/launchers/loop17/_run_final17.sh` (master tables, extra, figures, report build), print the PDF (`reports/build/finish_report6.ps1`), read the pages, write the E209 result sentences (the summary bullet on unseen types has only a design sentence), copy the PDF to `reports/`, one fresh numbers reviewer, fix, final build; update the memory note with the review corrections (seed spread 0.05-0.09, map-only baseline, matched clean reference 0.94, consumer under failure).
4. **Next loop** = `experiment_execution/PROGRAM.md` (TEST allowed once, mail yes, shutdown only if no other chat runs): building-block list, tree search round 2 (joint E2E fine-tune, grid head + sparsification E2E, differentiable registration to the SD map, map trust / abstain, second-successor head for forks, speed), deep evaluation. Judge by Q, J, lanes found, route success, ms - and the paired bootstrap.

## Decisions waiting for the user
- TEST run of route head along the edges + fast configuration, and of route head + gate + map builder (PROGRAM.md says OK once for the next loop).
- Deployment configuration: j1 192/96 (25 ms single frame) or the 288/144 default.
- Purge of `_to_delete/` (35 GB, list in `_to_delete/README.md`); `git init` of this folder (`experiment_execution/PRACTICE_NOTES.md`).
- Literature: full text and code extraction of the ~25 key papers / ~8 repositories needs downloads (state files, source, size before; GPL repositories only as reading, not copying) - not started; the one-file literature overview is not done yet (`literature/README.md` is the summary, `papers/` the 404 notes).

## What changed in the folder today (so old notes make sense)
Everything moved from `C:\ML-Experiment\{scripts,out,reports,report_final,data,...}` into this folder; `experiment_execution/history/PATH_MAP.md` maps old to new, `MIGRATION_LOG.csv` lists every move. Code is grouped by role (`training/`, `evaluation/`, `tools/`, `reports/build/`) and short-lived code in `temp/`. New: `training/trainer.py`, `evaluation/{sparse_lanes/bootstrap.py, tests, examples}`, `tools/{experiment, viz, video}`, `experiment_execution/{LEDGER, PRACTICE_NOTES, templates}`, `literature/` (single place). Tested: `p16_fast_check.py` (identical graphs), all self-tests, 18 / 18 builder spec tests. Two other chats worked in parallel (`tools/env` tidy-up, CLAUDE.md pitfalls); a third (weekly reports) asked where report builders live: `reports/build/`.

## Facts a new chat needs and cannot derive
- GPU RTX 3060 12 GB: at most 2-3 jobs; run two trainings at once only with the GPU guard. WSL Ubuntu 22.04, env `gcar`; cached features in the WSL home (155 GB); no git here.
- Seed spread of the route head up to 0.09; noise band of S 0.010; seeds >= 2; TEST once.
- Memory notes of the user: `~/.claude/projects/D--Daten-Dropbox-Dokumente-Portfolio-Github/memory/` (`bev-lane-camera-radar-state.md`, `feedback_speed-and-quality.md`, `feedback_experiment-reports.md`).
