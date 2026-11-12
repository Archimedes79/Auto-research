# tools - the method as code

Standard-library Python (any interpreter: WSL, Windows, a Linux server). Most tools read the problem folder named by `AR_PROJECT` (else `default_project` of `../config.json`) and its `project.json` (not `archdraw.py`, `check_git.py`, `check_numbers.py`). The head docstring of every file is that tool's reference; `--help` lists the options of most of them (`testgate.py` prints its docstring, `check_structure.py` has no options). `selftest.py` runs all of them in a throw-away problem (about 5 minutes); the real records are never touched.

| file | what it does |
|---|---|
| `run.py` | the launcher and START GATE: card (hypothesis, prediction, rule), the TEST row, the hold marker, GPU slot and line, code snapshot and tag, run record, ledger row, retries |
| `testgate.py` | the one rule for the TEST log (also used by the lane's evaluator) |
| `guard_hook.py` | the WRITE GATE: a Claude Code hook that refuses edits of protected files without a grant under a decided D<n> (`grant`, `grants`, `check`) |
| `hold.py` | the hold marker of a run id: only the person's words clear it |
| `machine_lock.py` | one machine lock between chats (an exclusive lock stops other owners' runs) |
| `ledger.py` | the run log: `LEDGER.csv` -> `LEDGER.md`, `add`, `note`, `tail`, `sync` |
| `schedule.py`, `plan.py` | the list of experiments as work items (states read from the run records); `plan.py status`: each experiment with its runs and verdict |
| `experiment.py` | the experiment card (the `json card` block of `experiment.md`): `check`, `show`, `kill`; the rule hash and the arms lock that run.py stores at launch |
| `card.py` | metric cards: the evaluator contract (`../schemas/card.schema.json`) |
| `decide.py` | the RULE GATE: the card's rule on the metric cards -> KEEP / CUT / TIE / INCONCLUSIVE; refuses a rule or arms changed after the launch |
| `result.py` | writes a decided experiment where people look: `results.md`, the schedule link, the research notes |
| `metrics.py` | the loop table: runs, GPU hours, verdicts, prediction hits and errors, TEST uses |
| `keep3.py` | the best three networks of a problem and their checkpoints |
| `provenance.py`, `lineage.py`, `package.py` | what a result card records, the verified chain network -> run -> code -> data, the result package and the clean-up of derived data |
| `commit_paths.py`, `check_git.py` | commit exactly the named paths in the shared checkout; what may go into git and be pushed |
| `check_numbers.py`, `check_structure.py` | numbers of a report against their sources; folder structure and dead paths of the living documents |
| `archdraw.py` | architecture drawings from a JSON spec, checked against the code |
| `topic_table.py` | the table of a research topic, made from its notes |
| `run_report.py` | one page per training run for a person |
| `install_agents.py` | lints `../agents/` and copies them to the workspace's `.claude/agents/` |
| `clean_temp.py` | tags the scratch of a finished loop, then moves or deletes it |
| `project.py` | finds the problem folder and reads its `project.json` (imported by most tools) |
| `selftest.py`, `selftest_launch.py`, `selftest_verdict.py` | the tests: all tools; the launcher and ledger; the verdict path |

The commands in one line each are in `../README.md` ("Tools in one line each").
