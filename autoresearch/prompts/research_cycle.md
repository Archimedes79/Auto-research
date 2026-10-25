# Research cycle prompt (P1 + P2) - paste and fill the <...>

Problem folder: <path>. Cycle <NN>. Budget: <hours>, <n> scouts at most in parallel. Downloads: no (abstract and project pages only) unless I say so.

Follow `autoresearch/METHOD.md` phases P1 and P2; read `PROBLEM.md`, `HANDOVER.md`, the last loop summary and `closed_questions.md` first.

P1 research
1. Write one topic brief per scout (`templates/topic_brief.md`): for every problem group with a gap, plus the foundation topics: <list or "all of PROBLEM.md">, plus training and losses.
2. Spawn `ar-scout` per brief (parallel), then `ar-deep-reader` for the top notes (at most <n>), then `ar-group-summarizer` per group.
3. Write the research digest (one page). Gate G1.

P2 hypotheses and plan
4. Spawn in parallel: `ar-hypothesis` with lens state of the art, foundations, training and losses, free; `ar-past-critic`; `ar-system-reviewer`.
5. `ar-referee` on all cards; revise or drop; gate G2.
6. `ar-planner` -> `PLAN.md` for loop <NN>; `ar-scheduler` -> `schedule.csv`.
7. Rewrite `HANDOVER.md`; show me the plan (tree, budget, what needs my OK: TEST, long runs) before the loop starts.
