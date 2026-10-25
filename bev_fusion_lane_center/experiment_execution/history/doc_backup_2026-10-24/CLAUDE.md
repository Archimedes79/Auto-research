# CLAUDE.md - rules for a session in this project

Project: fully differentiable lane-centre-line graph from camera + radar BEV features (nuScenes), planner-ready, "very good AND very fast". The user (Mirko) writes German or English - answer in the user's language; **documents, plans, code comments are English**. Live results do not belong in this file.

## Read first, in this order
1. `HANDOVER.md` - state, open items, next actions (update it at every milestone).
2. `experiment_execution/PROGRAM.md` - the current loop description (STOP_AT, TEST / mail / shutdown permissions). Do not re-plan; follow the state files.
3. State: `experiment_execution/indexes/TRIAGE_STATE.md` (loops 16 / 16b), `.../LOOP17_STATE.md` (loop 17, not restored yet: see HANDOVER), `experiment_execution/STATE.md`, `experiment_execution/LEDGER.md`; memory note `bev-lane-camera-radar-state.md`.
4. `README.md` (short map with links), `CONVENTIONS.md` (names, ids, flags), `STRUCTURE.md` (generated folder tree).

## Where things go (code: three roles + temp)
| folder | what | rule |
|---|---|---|
| `training/` | models, data, losses, training loops, inference pipeline of the system; `trainer.py` = general trainer (checkpoint + resume) | long-lived; new model code goes here |
| `evaluation/` | metrics (`lane_grid`, `sparse_lanes`), eval `drivers/`, `tests/`, `examples/`, protocol | do not write a private metric - use / extend these; self-tests after any change |
| `tools/` | few reusable helpers: `env/` (launchers), `experiment/` (run.py, ledger.py), `viz/`, `video/`, `catalogue/`, `maintenance/` | a new tool gets a docstring, a selftest and a row in `tools/README.md`; **`tools/maintenance/run_all_tests.sh` runs every self-test (8 PASS on 2026-10-24) - run it after any change to code folders** |
| `reports/` | final documents; `build/` = report builders and figure scripts; `source/` = build workspace | |
| `temp/` | one-off analysis and launchers by loop (`analysis/loopNN`, `launchers/loopNN`), `archive/` | short-lived: delete a loop's folder when its experiments are filed; nothing in `training/` may import from `temp/` |
| `experiments/` | one folder per experiment: README, small results, images (+ `code/` of finished ones) | filled when a run is filed |
| `experiment_execution/` | plan + log in human-readable form: PROGRAM, states, ledger, logs | |
| `runs/`, `models/`, `data/`, `literature/` | raw outputs / checkpoint registry / nuScenes + splits / notes, register, repos | cached features are in the WSL home (`data/README.md`) |
| `_to_delete/` | staged, not purged | the user purges |

Never create files in the project root. Flat imports (`import graph_eval as GE`) work because the launcher puts every code folder on `PYTHONPATH`.

## How to run
- Windows host, WSL2 Ubuntu 22.04, conda env `gcar` (Python 3.12, torch 2.8, RTX 3060 12 GB). From PowerShell: `wsl bash /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/_py14.sh <script.py> args` - finds the script by name in the code folders (a folder prefix such as `sparse_lanes/selftest.py` disambiguates), cwd = its folder. `_pyr.sh` = same with retry on CUDA 'unknown error'.
- Long jobs: `python tools/experiment/run.py --id ... --hyp ... -- bash tools/env/_py14.sh <script.py> ...` (run record, log, ledger row, `done` flag only on success), in the background; wait with a monitor, do not poll. At most 2-3 GPU jobs, GPU memory < 11 GB (`nvidia-smi`), timing only on an idle GPU. Trainings checkpoint every 400-500 iterations (`training/trainer.py`).
- Windows: no `python3` in Git Bash (use `wsl`), PowerShell 5.1 has no `&&`, quoting through `wsl bash -c` is fragile - put commands into a script file.

## Rules (standing, from the user)
- **Quality AND speed**: every experiment reports ms per frame (idle GPU, batch 1, cached features); optimisations must give identical outputs. If quality ties inside the noise band, prefer the cheaper variant.
- **Fully differentiable path**: no argmax, no matching, no rule step on the training path; gradient reaches every module. Better short lane graphs than false ones. Say whether a head is pixel-based or graph-based.
- Every experiment gets: architecture drawing (no red / non-differentiable blocks), numbers table incl. baselines, result images (key images to the user with SendUserFile), ms per frame; honest negatives stay.
- Two route-head seeds for any comparison (seed spread up to 0.09); state the noise band (0.010 in S) next to every difference; use the paired scene bootstrap (`evaluation/sparse_lanes/bootstrap.py`); judge sparsity-capped designs with Q (S without the sparsity term). Select on VAL; **TEST (21 scenes) only with the user's explicit OK, once** per finalist, logged in `evaluation/TEST_USAGE_LOG.md`; index test rows with `data/splits.json['test']` (not contiguous).
- Every finished run: ledger row, catalogue row (E-id), K-id for each new technique, experiment folder. Before a report is final: one fresh numbers reviewer (agent checks numbers against the sources).
- No downloads / installs without asking (reading web pages is fine). Never delete user data: move to `_to_delete/` and list it in `_to_delete/README.md`; never delete raw data or checkpoints. Never kill another chat's jobs (check `ListAgents`). No mail and no shutdown unless the current `PROGRAM.md` says so (mail: mirko.meuter@aptiv.com via web.de in Chrome, confirm in Sent); the user withdrew the shutdown once - read it again each time.
- **Rollback hazard**: files a session creates with Write / Edit are deleted if that conversation is rolled back (this destroyed the loop-17 code). Copy new code / results with `cp` when they matter and update `HANDOVER.md`.
- Pronouns: they / them for people unless stated.

## Pitfalls (learned the hard way)
- GPU: at most 3 jobs (11.7 GB with five); more than 2 extraction streams with DataLoader workers cause OOM. WSL needs a background `sleep 36000` session to stay alive overnight.
- `pkill -f` inside `wsl bash -c` kills its own shell: use a script file or the `[p]attern` trick. PowerShell `Start-Process` splits quoted args: use a wrapper `.sh`. Git Bash rewrites `/mnt/...` in `wsl` calls: use PowerShell.
- Never name a script like a stdlib module (`numbers.py` broke every python start). Bash heredocs with non-ASCII, quotes or backslash-n corrupt files: use Write / Edit.
- Orphan `tail -F` processes lock directories on /mnt/c (renames fail).
- `model.pt` saved every N iterations is not a completion sign; wait for the process or the `.done` flag (a false flag cost loop 17 an experiment: `e142.done.FALSE`). A run on a subset of perturbations can overwrite `eval.json`: always run all.
- Training can derail suddenly (BN running_var blow-up): lr <= 6e-4, gradient clipping, roll-back guard (`trainer.fit` has all three). Add the builder's time recursion only after the seed heads have learned (else S collapses).
- Reports: HTML from `reports/build/build_*_report.py`, PDF by headless Chrome (`reports/build/finish_report6.ps1`; cannot overwrite a PDF that is open in a viewer).

## Style
Match the surrounding code (comment density, naming). Short sentences in documents. Update `HANDOVER.md`, not this file, for anything that changes from day to day.
