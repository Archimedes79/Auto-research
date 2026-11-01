# HANDOVER - 2026-11-01 14:35

New chat: start in the repository folder (parent of autoresearch/ and this folder), say "Read HANDOVER.md and continue". Rewrite this file at every milestone (<= 3 KB). Mirko is often interrupted and reads little: one small step per turn, say where we are.

## Who does what
- Lead of loop 22 (filed): the repository-review chat (session 6e9638e0). The layered design is merged (8292f4a). The long run `t10.dual_long_s0` was stopped at step 5030 (14:38, exit 75) on Mirko's decision: do not restart or resume it; it resumes only on his word. Until then the freeze of bev_fusion/training/*.py and the autogt signature files has no end date; the GPU is idle. The GT Discussion chat owns autogt (D29), the tools chat `autoresearch/tools`.

## Where we are
- Lane prototype of record: `recall_sp0_dp` (VAL lanes found 0.781, Q 0.742; TEST E268.p, the one TEST use: lanes found 0.740, Q 0.727, R@1 far 0.749 vs control 0.762 / 0.712 / 0.776). Run: `wsl bash tools/env/_py14.sh lane_prototype.py recall_sp0_dp`. `champion` / `control` unchanged. KEEP but not in it: E266 seed ring; blocked by precision: E267, E264.
- bev: B2 (`runs/t11_vel_hurdle_acc2`) is the bev prototype (`../bev_fusion/records/morning_2026-11-01.md`); its t11 code is in tag `bf/t11.vel_hurdle_acc2`, not on main (needs a critic pass).
- Method change today (Mirko, no D entry): research only on a trigger, roles inline; `autoresearch/METHOD.md` "Planning a round", step 0 of `research_steps.md`.

## Decided today
D17 loop ranges by hand; D18 bands per card; D19 machine lock; D25 vertex sentence; D29 one GT version + autogt tags (v1-v5 done; evaluation default, run records and training start check after the freeze); D30 one image region for all pipelines; D33 hook gaps; D35 key-level protection; D36 coverage rule; D37 fusion numbers on the fusion area only; D38 front-only rig: outside the fusion area don't care on both sides (settles D31).

## Open for Mirko
- D4: promote `recall_sp0_dp`? after route success + idle ms (next lane loop). "D1." (unclear, asked back).
- D36: camera loss only where the camera sees, radar loss only where the radar sees, lanes camera-only, one rule for training, evaluation and execution; note `../bev_fusion/records/sensor_coverage_2026-11-01.md`; BEV + GT chats start when the freeze ends (no date).
- D36 details, one at a time: radar rule open (Mirko: no per-box gate; radar should see moving objects in line of sight; maybe weights), cells hidden behind a box, object counts by its centre cell (D38 b); aux heads: own area, debug tool, off in final E2E. GT chat plans v6 after the freeze.

## Next
1. Next lane loop: precision at range, idle timing of ring / union / prototype, route success on spur-kept graphs.
2. bev (BEV system chat): t11 code review + commit, seed twin of B2, the long run with hurdle + accumulation 2, then D29 part 2 and D36.

## Pointers
PROBLEM.md; PLAN.md; METHOD.md; closed_questions.md; DECISIONS.md; git: tag `lcr/<id>` per run (bev `bf/<id>`), pushes only on request.
