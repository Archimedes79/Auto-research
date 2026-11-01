# HANDOVER - 2026-10-31 23:45

New chat: start in the repository folder (parent of autoresearch/ and this folder), say "Read HANDOVER.md and continue". Rewrite this file at every milestone (<= 3 KB).

## Who does what
- **Lead of loop 22 and of the overnight bev_fusion round, owner of the code of both: the repository-review chat (session 6e9638e0)**, prompt `experiment_execution/history/prompts/PROGRAM_loop22.md` (D28: prototype by 06:30, +2-3 h allowed, then shutdown; one TEST check; no mail). The tools chat owns `autoresearch/tools` (it has uncommitted work there: commit by path). BEV chats: idle tonight.

## Where we are
- Loop 21 filed in its state (E257-E260 summaries, status report `reports/weekly/Status_after_Loop21_EN.pdf`, review); its scratch `temp/loop21/` stays because the loop-21 records point to it (it was lost from disk once on 2026-10-31 and written back from git; tag it with `clean_temp.py` only together with the record links). `project.json` loops now 19-22 (grant D28).
- Loop 22 (E261 ff.): light research cycle on gap 1 (coverage at range), three hypothesis lenses (E261-E266), then referee + planner; GPU arms run after the bev_fusion trainings (one training fills the GPU).
- Lane prototype: `tools/prototype/lane_prototype.py` (being built: `recall_dp`, planner-ready graph JSON, numbers, video).
- bev_fusion: `t10.dual_s1` (seed twin, its rule E1) trains 23:24 - ~01:20; next a t11 card on the weakest head (motion term flat at 1.094 in every run) and / or t10 at 6000 steps.

## Next
1. Cards E261-E266 -> `ar-referee` -> `ar-planner` (PLAN.md loop 22, schedule).
2. bev: decide t10 by its rule after s1 (`evaluation/board.py`), start t11.
3. Lane GPU arms after bev; CPU arms now. 06:30: summaries, one TEST check of the lane prototype, check_numbers, reviewer, shutdown.

Open decisions for Mirko: D25, D4, D18, D19 (DECISIONS.md).

## Pointers
PROBLEM.md; METHOD.md; closed_questions.md; literature/OVERVIEW.md; git: tag `lcr/<id>` per run (bev: `bf/<id>`), remote origin (pushes only on request).
