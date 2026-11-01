# HANDOVER - 2026-11-01 06:35

New chat: start in the repository folder (parent of autoresearch/ and this folder), say "Read HANDOVER.md and continue". Rewrite this file at every milestone (<= 3 KB).

## Who does what
- Lead of loop 22 and of the overnight bev_fusion round (D28): the repository-review chat (session 6e9638e0); the round is finished (P4). Next: the Architecture chat (local_79b594f7) merges `ar/layered-design` in the quiet window it has been told about; after that the 6000-step bev run `t10.dual_long_s0` (smoke test first). The tools chat owns `autoresearch/tools` (commit by path).

## Where we are
- Loop 22 filed in its state: PLAN.md "End of loop" section, experiment files E261 - E268 (section 3 each), ledger rows; morning page `reports/source/weekly/morning_2026-11-01_loop22.md`.
- Lane prototype of record: `recall_sp0_dp` (rc1 + fork pruning 0.4 + spurs kept + output DP 0.2). VAL lanes found 0.781, Q 0.742; TEST (E268.p, the one TEST use) lanes found 0.740, Q 0.727, R@1 far 0.749 vs control 0.762 / 0.712 / 0.776. Run: `wsl bash tools/env/_py14.sh lane_prototype.py recall_sp0_dp [--scenes 23 7 --video]`. `champion` / `control` registry entries unchanged.
- KEEP but not in the prototype: E266 seed ring (R@1 far +0.015). Coverage-positive but blocked by precision guards: E267 ring + spurs, E264 two-seed union.
- bev_fusion: B2 (`runs/t11_vel_hurdle_acc2`) is the bev prototype (velocity revived, see `../bev_fusion/records/morning_2026-11-01.md`); its t11 code lives in tag `bf/t11.vel_hurdle_acc2` and is not on main yet (needs an ar-reviewer code pass and the backbone chat's train.py hunks kept apart).

## Next
1. Mirko: read both morning pages; decide whether spurs kept becomes the recall-family default (D-entry), D4, D17.
2. Next lane loop: precision at range (range-dependent threshold for edge seeds; union with a wider vertex merge), idle timing of ring and union, route success on spur-kept graphs.
3. bev: code review + commit of the t11 code; seed twin of B2; the 6000-step run with the hurdle + accumulation 2.

Decided 2026-11-01 and applied: D25 (PROBLEM.md vertex sentence), D18 (per-card bands), D19 (`autoresearch/tools/machine_lock.py`, run.py waits on another owner's exclusive lock), D33 (gate-hook gaps), D17 (loop ranges by hand, closed). Open for Mirko: D4 (to discuss), D29 - D31 (bev); Mirko's "D1." is asked back (DECISIONS.md).

Method change 2026-11-01 (Mirko in chat, no D entry): research (P1) only on a trigger; planner / coder / researcher work inline unless context or fresh eyes need a spawn. A round starts with `autoresearch/prompts/research_steps.md` step 0 (ten lines, Mirko says OK); rules in `autoresearch/METHOD.md`, "Planning a round".

## Pointers
PROBLEM.md; PLAN.md; METHOD.md; closed_questions.md; git: tag `lcr/<id>` per run (bev: `bf/<id>`), remote origin (pushes only on request).
