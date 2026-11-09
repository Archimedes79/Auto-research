# decisions/ - the decisions of the person, one file each (D69, 2026-11-10)

Replaces `DECISIONS.md` (it is in git history). This folder is the only channel for decisions that are the person's: promoting a champion (with its one TEST check), changes of the problem (`PROBLEM.md`), changes of the gates (the protected files of `project.json`, `guard_hook.py`, the hook settings), permissions (downloads, installs, mail, shutdown, backups, remote). An agent writes the question with options, recommendation and evidence and acts only on the recorded answer; an answer given in chat is recorded by the agent with the date and the person's words.

| file | what |
|---|---|
| `D<n>.md` | one decision. The next free id is the highest `D<n>` plus one (`ls | sort -V | tail`). |
| `no_id.md` | the early rows of 2026-10-25 that had no D id; kept word for word, they decide nothing for the hook. |

**A file** has one block per step, newest decided block first:
```
# D70

## Open

Id: D70
Question: ...
Options: ...
Recommendation: ...
Evidence: ...

## Decided

Decided: 2026-11-10
Decision: D70 (the person, where, his words verbatim). My reading, which he can correct: ...
Consequence: what changes, which grants, which files
```
A question is open while the file has no line that starts with `Decided: <yyyy-mm-dd>`; the answer is appended as the `## Decided` block. An open question can also be a schedule item of kind `decision` with owner Mirko (`schedule.py add`); its `results.md` and this file say the same thing once.

**The gate.** `autoresearch/tools/guard_hook.py` reads this folder (project.json `decisions`): a grant for a protected file is valid only while its decision id has a file `D<n>.md` with a `Decided:` line. Rows that were migrated from the old table are unchanged word for word (escaped pipes stay escaped); the three old Open rows (D4, D31, D66) carry an `## Open` block and, where the table had one, a `## Decided` block too.
