# HANDOVER - 2026-11-01 04:55

New chat: start in the repository folder (parent of autoresearch/ and this folder), say "Read HANDOVER.md and continue". Rewrite this file at every milestone (<= 3 KB).

## Who does what
- **Lead of loop 22 and of the overnight bev_fusion round, owner of the code of both: the repository-review chat (session 6e9638e0)**, prompt `experiment_execution/history/prompts/PROGRAM_loop22.md` (D28: prototype by 06:30, +2-3 h allowed; shutdown CANCELLED; one TEST check; no mail). After the round the Architecture chat (local_79b594f7) merges `ar/layered-design`: tell it when the quiet window begins. The tools chat owns `autoresearch/tools` (commit by path).

## Where we are (loop 22, light cycle on gap 1 = coverage at range)
- E265 spurs kept (`spur_len 0`): **KEEP** (lanes found +0.0115 on 4 checkpoints, Q +0.004, guards hold). E261 rc2 at its own point: **CUT**; E262 cancelled.
- Lane prototype of record: `recall_sp0_dp` (candidate S; VAL 873: lanes found 0.7809, R@1 far 0.7795, Q 0.7424 vs `recall_dp` 0.7672 / 0.7713 / 0.7399). Not yet packaged with `tools/prototype/lane_prototype.py`; no TEST yet (E268).
- E264 union s2 + s3: u23 card lanes found +0.047 but false length 0.064 (> 0.062); CPU arms u / v / mfl running; timing E264.t needs the idle GPU.
- E266 (seed ring) / E267 (ring + spurs): need the GPU (E266.b0 first); E266.gold (golden + all tests) PASS.
- bev_fusion: t10_dual E1 holds on both seeds (BEV chat). t11 velocity: B' trained (1000 steps), B2 (accum 2) training until ~05:40, then evals of both (scratchpad queue `bev_queue6.sh`).

## Next
1. After B2 + evals: lane GPU window (E264.t timing on an idle box, E266.b0, g01, b1, then E267, E266.c / v).
2. Decide E264, E266, E267 (`decide.py` + lead checks); prototype by the PLAN rule; package it; E268 card + TEST log row (grant D28) + one TEST run.
3. bev: decide t11 by its section 2 rule; package the bev prototype (`evaluation/bev_prototype.py`).
4. P4: summaries, morning pages, check_numbers, ar-reviewer (numbers), check_structure; message the Architecture chat.

Open decisions for Mirko: D25, D4, D18, D19 (DECISIONS.md).

## Pointers
PROBLEM.md; PLAN.md; METHOD.md; closed_questions.md; git: tag `lcr/<id>` per run (bev: `bf/<id>`), remote origin (pushes only on request).
