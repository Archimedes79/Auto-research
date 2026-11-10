# Open items and design rules of the research system

The long review and migration plans (2026-10-25, the target design of D32) are in git: `git show dad824d:autoresearch/history/ROADMAP_2026-10-25.md`, `history/TARGET_2026-11-01.md`. This file holds only what is still open and what was learned. What exists is described in `README.md`.

## Open
| item | what | needs |
|---|---|---|
| backups | the run records, one copy of the registered checkpoints and splits to a second place, and a tested restore; `main` and the run tags go to GitHub, nothing else is backed up | the person |
| CONFIRM split | a KEEP on VAL counts as "adopted" only when a CONFIRM split agrees (decided as an idea; the lane project lists a `confirm` split of 1,567 frames, but no tool requires it and `evaluate.py` does not read it) | first use |
| early kill | `trainsuite early` is advisory (gap to a control run, calibrated on finished pairs, stops nothing). The signal is weak: about 15 control / variant pairs from 8 or more independent runs of the same code and GT, evaluated about every 250 steps and logging a real metric (lanes found, Q), are needed before it may stop a run | more finished pairs |
| lane timing and route | route success, the failure series and timing are parts of the lane `evaluate.py` (`--route`, `--failure`, `--timing`); the stored-input paths are verified on the CPU, the paths that run the builder and the timing are not (commands in `bev_fusion_lane_center/evaluation/README.md`) | an idle GPU |
| claims | a file `state/claims.json` (chat, paths, reason, release) that `check_git.py` and `commit_paths.py` would warn about, for freezes that go stale; advice only, a second freeze that goes stale is the signal to build it | not built |
| the gate | D75: the hook reads the default problem for every path, the BEV problem has no protected list, a crash of the hook lets the edit through | the person |

## Design rules (what the reviews taught)
1. **One truth per fact.** A run's state lives in its run record; the ledger holds judgement (prediction, rule, result, decision) and a copy of the facts written once at the end; the schedule is a plan whose item states are read from the records. Every sync between stores produced stale or ghost rows.
2. **A feature earns its place by use.** The tick started 0 of 61 runs: the cycle ran on `--after` chains, so it went (D32). Keep only what is used.
3. **Waiting is not holding.** A launcher that waits for dependencies holds no slot; give-up times default to 12 hours so a chain does not die silently.
4. **Gates guard the truth, not the tools.** The expensive mistakes are a touched holdout, a changed golden number or champion entry, a changed problem definition. Tools are covered by git, tests and review.
5. **A test that runs the real tools.** A stub hid a swallowed error and the rule of a card was never registered (2026-11-10): the chain test run -> decide -> result with the real tools is the guard, a unit test per tool is not enough.
6. **Shared working tree, shared index.** Commit by explicit path (`commit_paths.py`) and in one command; never leave anything staged.
7. **Quiet machine for timing.** Training, tests and surveys on one GPU and CPU make load averages of 50 and invalidate timing; timing runs need an agreed quiet window (`machine_lock.py`).
8. **The hold marker.** A run that was stopped (exit 75) or put on hold gets a `HOLD` file; only the person's words clear it (`hold.py`), never a chat's own judgement. Relayed instructions are written as relayed.
