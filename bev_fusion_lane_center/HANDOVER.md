# HANDOVER - 2026-11-01 14:35

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko is often interrupted and reads little: one small step per turn, say where we are.

## Who does what
- Lead of loop 22 (filed): the repository-review chat (session 6e9638e0). The long run `t10.dual_long_s0` was stopped at step 5030 (14:38, exit 75) on Mirko's decision: do not restart or resume it; it resumes only on his word. Until then the freeze of bev_fusion/training/*.py and the autogt signature files has no end date; the GPU is idle. The GT Discussion chat owns autogt (D29), the tools chat `autoresearch/tools`.

## Where we are
- Lane prototype of record: `recall_sp0_dp` (VAL lanes found 0.781, Q 0.742; TEST E268.p, the one TEST use: lanes found 0.740, Q 0.727, R@1 far 0.749 vs control 0.762 / 0.712 / 0.776). Run: `wsl bash tools/env/_py14.sh lane_prototype.py recall_sp0_dp`. `champion` / `control` unchanged. KEEP but not in it: E266 seed ring; blocked by precision: E267, E264.
- bev: B2 (`runs/t11_vel_hurdle_acc2`) is the bev prototype (`../bev_fusion/records/morning_2026-11-01.md`); its t11 code is in tag `bf/t11.vel_hurdle_acc2`, not on main (needs a critic pass).
- Method change today (no D entry): research only on a trigger, roles inline; `autoresearch/METHOD.md` "Planning a round".

## Decided today
D17 loop ranges by hand; D18 bands per card; D19 machine lock; D25 vertex sentence; D29 one GT version + autogt tags (v1-v5 done; evaluation default, run records and training start check after the freeze); D30 one image region for all pipelines; D33 hook gaps; D35 key-level protection; D36 coverage rule; D37 fusion numbers on the fusion area only; D38 front-only rig: outside the fusion area don't care on both sides (settles D31); object visibility = footprint share; aux heads: own area, debug tool, off in final E2E.

## Open for Mirko
- D4: promote `recall_sp0_dp`? after route success + idle ms (next lane loop). "D1." (unclear, asked back).
- D36: camera loss only where the camera sees, radar loss only where the radar sees, lanes camera-only, one rule for training, evaluation and execution; note `../bev_fusion/records/sensor_coverage_2026-11-01.md`; BEV + GT chats start when the freeze ends (no date).
- D36/D38 open: radar rule (Mirko: no per-box gate; radar should see moving objects in line of sight; soft weights later, BEV chat); thresholds of the footprint share come from the data. GT chat plans v6 after the freeze.

## Next
1. Next lane loop: precision at range, idle timing of ring / union / prototype, route success on spur-kept graphs.
2. bev (BEV system chat): t11 code review + commit, seed twin of B2, the long run with hurdle + accumulation 2, then D29 part 2 and D36.
3. Core (tools chat): reproducibility block (`autoresearch/reproducibility.md`), hold marker for stopped runs, hygiene list.

## Pointers
PROBLEM.md; PLAN.md; METHOD.md; closed_questions.md; DECISIONS.md; git: tag `lcr/<id>` per run (bev `bf/<id>`), pushes only on request.
