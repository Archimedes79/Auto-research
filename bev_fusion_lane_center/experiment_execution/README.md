# Experiment execution - the state of the method for this problem

How loops are planned, run and logged follows `../../autoresearch/METHOD.md` (P1 research -> P2 hypotheses + plan -> P3 loop -> P4 close). This folder holds the problem's state and records; the tools are in `../../autoresearch/tools/` and read the paths from `../project.json`.

| file / folder | what it holds |
|---|---|
| (no plan, no schedule) | the loop-22 `PLAN.md` and `schedule.csv` were stale and were removed on 2026-11-10 (git history keeps them). The experiments are the cards in `../experiments/`; what happens next is in `../HANDOVER.md`. A queue can come back as `schedule.csv` (`autoresearch/tools/schedule.py add`), then `plan.py status` lists its experiments with runs and verdicts |
| [LEDGER.md](LEDGER.md) / `LEDGER.csv` | one row per run started with `autoresearch/tools/run.py`: hypothesis, prediction, rule, status, result, decision, commit |
| [decisions/](decisions/) (one file D<n>.md per decision) | the only channel for Mirko's decisions (champion promotion with its TEST check, problem changes, tool / gate changes, permissions): open entries with options, recommendation and evidence; decided entries with date |
| [closed_questions.md](closed_questions.md) | questions answered the same way twice; re-opening needs a new mechanism |
| `runs/` | run records of the launcher: `run.json` (with commit and tag), `run.log`, `done` flag only on success |
| [plans/](plans/) | longer design documents written before loops: loop 16 triage plan, hypothesis tree, concept, growth / link architectures, literature compatibility space, experiment table (generated), `PRACTICE_NOTES.md`, `System_Review_2026-10-25.md` |
| [history/](history/) | everything that was state once: STATE of loops 2-11, loop indexes 8-15, loop-16 triage state, LOOP18_STATE, the loop prompts (`prompts/PROGRAM_loopNN.md`, one per loop from 18 on; written from the start checklist of `autoresearch/prompts/loop.md`), old handovers, loop-17 recovery and loop-18 code backup, logs of loop 17, old templates, launcher scripts of loops 15-16 and the loop-wide tools of loop 18 (`history/launchers/loop18/`: backup, ledger notes, status, final VAL table; the experiment scripts of loop 18 are in `../experiments/E2xx_*/code/`, D13), path map, migration log, doc backups, `SYSTEM_2026-10-25.md` (first design note, now METHOD.md + PROBLEM.md) |

## The cycle in this folder
1. **P1 / P2** (before a loop, the steps of `autoresearch/prompts/research_steps.md`, started by hand): research in `../literature/`, experiment cards (`experiment.md` per experiment folder), optionally a queue (`schedule.py add`).
2. **P3** (the loop, prompt `autoresearch/prompts/loop.md`): every wake-up reads HANDOVER and the ledger (`ledger.py tail`, `schedule.py show` if a queue exists), advances the next items (detail plan -> code -> `run.py` -> `evaluate.py` -> `decide.py` -> summary) and writes them back.
3. **P4**: loop summary / report, `closed_questions.md`, card statuses, `clean_temp.py loopNN --move`, `check_structure.py`, HANDOVER rewritten.
