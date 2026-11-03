# records - the run records of bev_fusion under autoresearch (D23, 2026-10-31)

Since D23 a bev_fusion training starts through the launcher of the research system, `../../autoresearch/tools/run.py`, like the runs of the lane problem. The launcher writes the card (hypothesis, prediction, rule) and the facts of the run into the ledger, tags the code it ran (`bf/<run id>`: `training/`, `evaluation/`, `autogt/`, `tests/`, `temp/train/run_cfg.sh` and `autoresearch/`, uncommitted changes included), waits for a GPU slot shared with the lane problem (2 slots on this machine), retries a CUDA "unknown error" and writes a done flag only for exit code 0. The paths are in `../project.json`.

| file / folder | what |
|---|---|
| `LEDGER.csv` / `LEDGER.md` | one row per run (`ledger.py`); made by the first run |
| `runs/<run id>/` | run record of the launcher: `run.json` (command, commit, tag, status, duration), `run.log`, `heartbeat`, `done` |
| `schedule.csv` | optional work list for `schedule.py` (not used yet) |
| `champion.json` | the base chain every card is compared against (`base` = a run name with its seeds, `open` = the arms on the board, `options` = adopted options, `promoted` = null until Mirko's OK); read by `../evaluation/board.py` |
| `CKPT_SLIM_2026-11-01.md` | what the cleanup of 2026-11-01 did: 34 finished runs slimmed to one checkpoint each (per-run sizes), the outdated caches deleted, what was kept on purpose and what is open |
| `morning_2026-11-01.md` | summary of the overnight round of 2026-11-01 (D28): t10_dual seed twin, the velocity fix t11 (arms B' / B2), the prototype command, incidents and open items |
| `sensor_coverage_2026-11-01.md` | proposal: one coverage rule per output type for training, evaluation and execution (what the code does today, findings, open points; Mirko's question of 2026-11-01) |
| `cleanup_inventory_2026-11-02.md` | the repository-wide clean-up of D44 (5): a size and a rule per class, what is protected, what each class needs (owner, tag, archive); read-only list, nothing deleted by it |
| `lane_diagnostic_2026-11-03.md` | step 1 of `../plan/plan_2026-11-03.md`: lane divider IoU of `t16.full_s0` by range and train against val, the grid ceiling; why L4 is the lever (a diagnostic, no verdict) |
| `bn_over_dt_2026-11-03.md` | open finding #5 measured on `t16.full_s0` (EMA weights, 225 val frames): the dt 0 loss and tolerant IoU with the stored BatchNorm statistics (a mixture of dt -0.5 / 0 / +0.5), with batch statistics, and with statistics re-estimated from dt 0 only; the mixture costs at most 0.9 % of a group loss and 0.001 lane divider IoU@1, a per-dt BatchNorm is not worth a card now (a diagnostic, no verdict) |
| `open_findings_2026-11-02.md` | triage of the bug hunt's open findings (real? severity? fix size? what was checked and what is per report) for the module-iteration list after the baseline pair; nothing fixed |
| `speed_options_2026-11-03.md` | where the host-bound step spends its ~45 thousand operators (counted on CPU per section), the sync-like operations, what `torch.compile` can take, and the ranked speed levers with the GPU measurement that would confirm each; nothing changed |
| `doc_review_2026-11-03.md` (+ `_autoresearch.md`, `_bev_fusion.md`, `_lane.md`) | the documentation review of D49: are the md files, decisions, READMEs and drawings up to date, what breaks if each is deleted, how to re-answer a decision, what was deleted and what is proposed next; the three long files are the evidence for the proposals and can be deleted when those are done |
| `BOARD.md` | the board made by `python evaluation/board.py --write`: every open arm against the base, all heads, from the stored `metrics.json` files; regenerate it, do not edit it |

## Start a training
From `bev_fusion/` (the problem folder), in the WSL. `AR_PROJECT` names this problem (the tools default to the lane problem):

```
wsl bash -c 'AR_PROJECT=../bev_fusion nohup python3 ../autoresearch/tools/run.py --id t9.C --gpu --split train --hyp "..." --pred "..." --rule "..." -- bash temp/train/run_ar.sh training/configs/t9_C.json > runs/t9_C.launcher.log 2>&1 &'
```

- `temp/train/run_ar.sh` (not `run_cfg.sh`, which always exits 0) trains from a `git archive` of `training/` at `$AR_COMMIT`, the snapshot commit `run.py` made for this attempt (also on a resumed id, whose tag still names attempt 1), with the config copy inside that snapshot, and exits with `train.py`'s exit code, so a crash is a failed run and a CUDA "unknown error" is retried (`train.py` resumes from its last checkpoint).
- Not pinned: `autogt/` is imported from the live tree when the training starts. The tag records its state, and the run writes the GT signature it used to `runs/<name>/gt.json`; `evaluation/bev_suite.py compare` checks it.
- The training itself still writes `runs/<config name>/` (`train.log`, `log.jsonl`, checkpoints, `.done`) as before; the run record here holds the launcher's facts and the ledger row.
- Watch it with `python -m trainsuite --root ../bev_fusion/runs watch <config name>` (from `autoresearch/`).
- After the run: `ledger.py note <run id> --result "..." --decision "..."` (with the same `AR_PROJECT`).
- A planned row first (`ledger.py add <run id> --hyp ... --pred ... --rule ...`) lets the start line carry only `--id`; chains wait with `--after <run id>`.

Not yet under the launcher: the queue scripts in `temp/study/` (`lane*.sh`); they keep working, but their runs have no ledger row. The GT that a training reads is built from the live `autogt/` (the tag records its state; the GT signature in the run's config is item F3 of the repository review).

## The best three networks (keep3.json)



`keep3.json` (tool: `autoresearch/tools/keep3.py`, `AR_PROJECT=../bev_fusion`) holds the best three ARMS of this problem with all their seed checkpoints (standing instruction of Mirko, 2026-11-01). Ranking value `board7` = the mean of the six map IoUs and the mAP of `evaluation/board.py`, averaged over the seeds, on `val` and GT signature `0.5_4_15efbde2d1`, band 0.02; a convention for ordering, not a verdict (the card rule decides). Held today, chosen by hand: `t11_vel_hurdle_acc2` (best value), `t10_dual` (the new base, D39) and `t8_ctl` (the control, pinned). A new arm comes in with `keep3.py consider <arm> --apply --ledger <id>` (a KEEP row) or `--by-hand "reason"`; then commit `records/keep3.json` and push (`git fetch`, `check_git.py`). Other checkpoints may be slimmed or archived, never deleted.
