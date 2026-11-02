# Fewer files for experiment ideas (built 2026-11-01)

Mirko, looking at `overview.svg`: for research results a summary plus detailed extractions are clear; but to carry an experiment idea, is more needed than one file for the planner, one file per experiment and one list of the experiments? Mirko: "Yes, build". 

## Built (2026-11-01)
- **Card block:** the card is a fenced `json card` block in section 0 of `experiment.md` (`experiment.py` reads it; an old `experiment.json` is still read; both present and different is refused; `experiment.py migrate` moves a json into the md and the json to `<problem>/_to_delete/experiment_json_<date>/`). The 26 lane cards that had both files are migrated, 3 lane cards without an `experiment.md` (E215, E219, E232) keep their json. `decide.py` writes `decision.json` next to the md. Template, schema text and the docs that named `experiment.json` follow.
- **Status from the ledger:** `plan.py status` prints each plan node with its schedule items, ledger rows and the last `decide.py` verdict. The status column is gone from `templates/plan.md`; the live lane `PLAN.md` of loop 22 keeps its column until the loop is filed.
- **Not built, on purpose:** `INDEX.csv` was not generated but removed on 2026-11-02 (Mirko: nobody writes it; it was a copy of the table in `experiments/README.md`, written once at the migration of 2026-10-24); `schedule.csv` and `LEDGER.csv` stay two files with one id (a queue is edited, a ledger is appended to) and a finished item stays in the schedule (its state is derived from the run records); `plans/` stays (loop-level detail plans, not per-experiment).
- Reviewed by a fresh critic (mode code); its findings (broken or doubled blocks, a sample inside a longer fence, CRLF, same-day move, id matching, verdict only from the `decide.py` token) are fixed and tested.

## The proposal as drafted

## Today (per problem)
| file | job | read by code |
|---|---|---|
| `experiment_execution/PLAN.md` | the experiment tree of the loop, with a status column | no |
| `experiments/E<id>_<slug>/experiment.md` | the idea: card, detail plan, summary (one growing file) | no |
| `experiments/E<id>_<slug>/experiment.json` | the same card as data (arms, prediction, rule, kill) | `decide.py`, `experiment.py`, `run.py --require-card` |
| `experiment_execution/schedule.csv` | the queue: order, dependencies, resource, command | `schedule.py` |
| `experiment_execution/LEDGER.csv` / `LEDGER.md` | one row per run: prediction, rule, result, decision (`.md` is generated) | `ledger.py`, `decide.py` |
| `experiments/INDEX.csv` | catalogue of the experiment folders, written after the fact | no tool (a path lookup in `project.py`) |
| `experiment_execution/plans/` | detail plans of a loop | no |

So an idea lives in up to five places: PLAN status, card (md + json), schedule row, ledger row, index row.

## Proposal
1. **Planner file = `PLAN.md`.** It keeps the question of the loop and the tree with the reasoning. The status column goes: a status is read from the ledger (`plan.py status` prints it), so it cannot go stale.
2. **One file per experiment = `experiment.md`.** The card becomes a fenced json block at the top of this file; `experiment.py` / `decide.py` read the block. `experiment.json` and `plans/` go away (the detail plan is section 2 of the same file, as the template already says). Old json files stay readable until converted by one migration script.
3. **One list.** `schedule.csv` is the list of planned and running experiments (order, `after`, resource); `LEDGER.csv` is the same list after the run (result, decision). One id runs through both, and a finished item leaves the schedule. `INDEX.csv` is generated from the headers of the `experiment.md` files, not kept by hand.

Result: planner file, one file per experiment, one list (to do, then done), and the rest is derived.

## What stays
Research notes (summary plus detailed extractions), `DECISIONS.md`, `closed_questions.md`, the handover. They have other jobs.

## Cost and risk
- Tools to touch: `experiment.py`, `decide.py`, `selftest.py`, one new `plan.py status`, one migration script; the card schema stays the same, so hard rule 3 (`run.py --require-card`) is not weakened.
- Risk: a json block inside markdown is easier to break by hand than a pure json file; `experiment.py check` must run on every write (it does for json today).
- Not decided: whether `schedule.csv` and `LEDGER.csv` should become one physical file. They differ in life cycle (a queue is edited, a ledger is append-only), so this proposal keeps two files with one id.
