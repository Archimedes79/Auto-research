# The autoresearch method

Improving a technical system by research: agents do most of the work, a person (the owner) sets the direction and decides. A problem plugs
in through one folder `problems/<name>/` (`research/README.md`: what a problem writes, what the core does; `EXTENDING.md`: a new one). Code
measures and refuses; the files are the memory; git keeps what ran.

## The loop
| phase | what happens | it ends when |
|---|---|---|
| Problem | `PROBLEM.md` (goal, chain, metrics, gaps) and `DESIGN.md` (the modules and the rules they obey); changes of `PROBLEM.md` are the owner's | the owner's answer is in `DECISIONS.md` |
| Research (on a trigger) | scans and deep notes on one topic (`ar-researcher`), a check by a fresh agent, a summary for the planner | the notes are checked |
| Cards | `research new <p> <slug> "<title>"`, the card (question, hypothesis, prediction, arms as settings of the base config, seeds, rule, kill rule, Design delta); `ar-critic` (cards) challenges it; `research check` | the critic accepts and check is clean |
| Runs | code behind a default-off setting (`ar-coder`, spec test, `ar-critic` on shared code); `research start` freezes card and code in one snapshot and tag; workers run the steps from it | every run is finished, stopped or killed |
| Verdict | `research judge --write`: the snapshot's rule on the stored counts, one commit of the records; the conclusion into `experiment.md` | the verdict is committed |
| Close | the planner reads the verdicts and proposes; the owner decides an adoption (`research adopt`, with an anchor test) and the next round; reports are built from the records (`research report`) | `HANDOVER.md` says where things stand |

Several experiments run at once (the machine's GPU slots decide). An arm of a card can reuse stored runs (`from`) or build on one
(`needs`). INCONCLUSIVE means: add a seed; TIE: take the cheaper variant; INVALID: fix the control or the comparison first.

## Hard rules (the code enforces them; everything else is a default)
1. **TEST** opens only for a run named in the problem's `TEST_LOG.md` under a decided decision (`research test`; the evaluation step and every read through `Evaluation` check the log again). A design sees TEST once; what was chosen on TEST is in-sample and labelled so.
2. **Protected files** change only under a grant of a decided decision (`research grant`): `.claude/settings.json` and `.claude/settings.local.json`; the gate code `research/__init__.py`, `research/gates.py`, `research/paths.py`, `research/hook.py` and `Evaluation/defaults.py`; `champions/`; every problem's `TEST_LOG.md`, `problem.json` and what its "protected" list names. The hook `research/hook.py` refuses every other edit of them, and every edit of the machine's state folder, where the grants are.
3. **A number that feeds a verdict** comes from a run of `research start`: it executes its snapshot, and only its worker writes records. A diagnostic is any other call and never a record.
4. **A verdict** comes from the rule the snapshot holds (`research judge`), or is adopted by hand with a written reason (`research adopt --by-hand`).
5. **Data**: raw data and the weights of champions and comparison networks are never deleted; other checkpoints, caches and run outputs only after a list names them and the owner confirmed that list in a decided decision (D154). For run folders that is `research clean`: the decided row names the list's hash, and `--delete <list> --decision D<n>` deletes only runs of that list that are still not needed (`research/README.md`).
6. **Other people**: the owner's decisions are acted on only when recorded; mail, shutdown, downloads, installs and remote actions only when allowed; another chat's jobs, running scripts and uncommitted files are not touched.

When a default does not fit, do the sensible thing and say in the record what was done differently and why; keep it away from where a
number comes from, the TEST split, data, other chats' work and the owner's decisions. When a hard rule is in the way, stop and ask.

## Files
| file | in | writer |
|---|---|---|
| `PROBLEM.md`, `DESIGN.md`, `HANDOVER.md` (<= 3 KB, rewritten, not appended) | the problem | the lead; `PROBLEM.md` only with the owner |
| `DECISIONS.md` (decided rows `\| date \| D<n>: ... \|`, open questions) | the problem; the root for what concerns several | agents record, the owner decides |
| `experiments/E###_slug/` (card, run records, verdict) and `ledger.csv` (generated) | the problem | see `research/README.md` |
| `champions/<problem>/` | the root | `research adopt` / `register` |

## Roles
| role | modes |
|---|---|
| lead (the main chat) | briefs, decisions with the owner, small work itself |
| `ar-planner` | problem, branches (one lens per run), plan, detail, tracking |
| `ar-researcher` | scan, deep, check, summary |
| `ar-coder` | the coding step of a card |
| `ar-critic` | cards, code, numbers - never what it wrote |

A role is spawned when it protects context, gives fresh eyes or runs in parallel. `agents/install.py` copies the roles into `.claude/agents/`.

## Git
`main` holds code and text records, never data, weights, rendered pictures or files over 1 MB (`research push-check`). Commit own paths only
(`research commit <paths> -m ...`); push only when asked. A run's code is its experiment's tag; nothing committed is rewritten.
