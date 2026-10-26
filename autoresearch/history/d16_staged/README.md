# D16 staged tools (remove after the deploy)

The simplified improver tools of decision D16 (state is derived from the run records, not synced between ledger, schedule and tick), built and self-tested in a scratch copy. The real `tools/` folder and `METHOD.md` are under the gate hook, so the files wait here until Mirko grants the edit or deploys by copy.

| file | becomes |
|---|---|
| `run.py.staged`, `tick.py.staged`, `schedule.py.staged`, `ledger.py.staged`, `selftest.py.staged` | `../../tools/<name>.py` (five files, 1623 -> 1426 lines) |
| `METHOD.patched.md` | `../../METHOD.md` (two lines: wake-up routine, read the records first) |
| `deploy.sh` | copies all six; run `wsl bash autoresearch/history/d16_staged/deploy.sh` from the repository root, then `wsl python3 autoresearch/tools/selftest.py` |

Revert: `git checkout -- autoresearch/tools autoresearch/METHOD.md`. After the deploy this folder is deleted in the same commit (git keeps it).
