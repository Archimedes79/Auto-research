# Roadmap of the research system (short form, 2026-10-26)

The long review and migration plan of 2026-10-25 (weaknesses W1-W14, findings R1-R10, proposals P1-P10, steps M0-M7) is kept unchanged in `history/ROADMAP_2026-10-25.md`. This file is only what is still open and what was learned.

## Done (one line each; details in git and `DECISIONS.md` D8, D15, D16)
- Records: locked atomic ledger and schedule writes, run record with heartbeat, done flag only on exit 0, a code snapshot and tag per run (temporary git index, no branch moves), refusal of id reuse, card required before a run (W1, W3, W4, W9).
- Decisions as code: `experiment.json` with rule clauses, `decide.py --experiment` from metric cards; evaluator contract `schemas/card.schema.json` with `card.py` (P1, P8).
- Evaluation: one entry point with a registry, golden numbers (cached and one uncached) in `run_all_tests.sh` (M0).
- Gates: hook on the protected files with grants (P4), holdout split from `project.json`.
- Measurement of the improver: loop table at the head of `LEDGER.md` (P3).
- Method simplified: 7 agents, 5 templates, light cycle by default (P6); structure check over `experiments/` (P7).
- State derived from the run records instead of synced between three files (D16, deployed 2026-10-26): item states are computed on read, the ledger row is written once at the end, `ledger.py sync` adds rows of runs that ended or lost their launcher, a second launcher of one id is refused, one tick at a time, cpu items one at a time. A fresh `ar-reviewer` pass found 8 findings in the first version; 7 are fixed with tests (one is the open item L1 below).

## Open
| item | what | needs |
|---|---|---|
| M2 | backups and push: push `main` and the `lcr/*` tags, copy the records to a second place, one copy of registered checkpoints and splits, test a restore | Mirko's OK (the remote lags by dozens of commits and no run tag is pushed) |
| P9 | CONFIRM split and the "adopted" state: a KEEP on VAL counts as adopted only when CONFIRM agrees | first use on D4 |
| P10 | early kill from learning curves (E241 replay: lanes found separates at 25 %, Q does not) | after cycle 19 |
| W8 | evaluation cache key has no checkpoint hash or inference-code fingerprint | small change in `evaluate.py` |
| W13 | scouts have WebFetch and Write: fetched text is data, scouts write only under `literature/` | hook rule or agent definition |
| W11 | evaluation covers graphs only: route success, failure series, timing are still per-loop drivers | after loop 19 |
| L1 | a re-run of an id whose final ledger write fails keeps the old finished row (its result and decision are cleared at the start; `ledger.py sync` replaces only planned / running / stalled rows): compare the row's `started` with the record's | rare; small change in `ledger.py`/`run.py` |
| M3-M5 | package rename `ar` / facade `lcr` / `lcr.train` recipes | **deferred, probably not needed**: the loop runs without them; revisit only if a second problem is plugged in |

## What the last days taught (design rules for the tools)
1. **One truth per fact.** A run's state lives in its run record; the ledger holds judgement (prediction, rule, result, decision) and a copy of the facts written once at the end; the schedule is a plan whose item states are read from the records. Every sync between stores produced stale or ghost rows (D15 review: half of its findings were in the sync code).
2. **A feature earns its place by use.** The tick started 0 of 61 runs: the cycle ran on `--after` chains. Keep only what is used; the cheap queue is a chain of launchers, the tick is a view plus an optional starter.
3. **Waiting is not holding.** A launcher that waits for dependencies holds no slot; give-up times default to 12 hours so a chain does not die silently.
4. **Gates guard the truth, not the tools.** The expensive mistakes are a touched holdout, a changed golden number or champion entry, a changed problem definition. Tools are covered by git, tests and review.
5. **Shared working tree, shared index.** Commit by explicit path and in one command; never leave anything staged (another chat's `git commit` takes it).
6. **Quiet machine for timing.** Training, tests and surveys on one GPU and CPU make load averages of 50 and invalidate timing (E243); timing runs need an agreed quiet window.
