# _to_delete - files moved out of the live tree, kept until Mirko empties the folder

Not in git (scratch rule). What left the tip of main is still in git history: restore with `git show <commit>:<path>` (the commit before the cleanup commit).

| folder | what, why |
|---|---|
| `root_docs_2026-11-01/` | `plan_draft2.md`, the previous plan draft (front camera, virtual upright camera, sigma points), superseded by `plan/plan.md` (draft 3); moved out of the root of `bev_fusion/` by the tidy of 2026-11-01 (Mirko: "Yes, tidy it") |
| `cleanup_2026-11-01/tests/` | `box_optimality_fresh.txt` (a generated log of one test run, no reference) and `run_when_gpu_free.sh` (a waiter that `run.py --gpu --wait-gpu` and `machine_lock.py` replace); found by the inventory of 2026-11-01 |
