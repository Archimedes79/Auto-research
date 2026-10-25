# HANDOVER - 2026-10-25 17:05

New chat: start in the repository folder (parent of autoresearch/ and this folder) or in this folder, say "Read HANDOVER.md and continue". Rewrite this file at every milestone (<= 3 KB).

## Where we are
- Loop 18 is CLOSED (10:55). Report: reports/weekly/Loop18_Fork_Pruning_Localisation_Report_EN.pdf. Results per node: experiment_execution/PLAN.md; every run: LEDGER.md; folders experiments/E209-E233. Headline: fork pruning closes the J gap (0.265 -> 0.409), chain 17.5 ms, joint E2E negative three times, TEST once (E230, S 0.748 vs 0.702).
- D8 (a), lead: loop-18 chat ("BEV Fusion Spurmitte Experiments", owner of autoresearch/tools and evaluation/). Done: M0 golden numbers (evaluation/golden.json, 7 entries incl. the uncached champion on VAL250), P5 launcher hygiene, P1 experiment.json + `decide.py --experiment` (decision.json for E215 / E219 / E232), run_all_tests 13 / 13 at 16:05.
- Data extension D7 (b): features (E234, 15:08) and steps 5-6 ground truth, route, SD map (E235, 16:56, CPU) of the 255 new scenes in `~/exp_gcar_b456/`; reproduction on b123 bit-identical except the exact-distance map, whose stored b123 version has TF32 errors (D10). Step 7 (loader) open: experiment_execution/plans/data_extension_b456.md.
- The research system was simplified today (7 agents, 5 templates, prompts/research_steps.md; roadmap and reviews in ../autoresearch/ROADMAP.md). No experiment of loop 19 has started (the user: system refinement first).

## Next step
1. Split agreed 16:00. Lead: P2 DONE (tick.py starts ready schedule items within the GPU / CPU slots, `--handover` status block; run.py takes its card from the planned ledger row, `--no-card` for smoke tests; schedule.py `command` column), review pending; next P8 (evaluator contract). Done by the data chat (8a18b25, bfd318e): P4 gate hook ON (`.claude/settings.json`; protected list in project.json; grant for ../autoresearch/tools/ under D8 until 2026-10-26 03:56, renew with `guard_hook.py grant`), P3 (`metrics.py`, loop table at the head of LEDGER.md), P7, the small items. Then P9 + b456 step 7 (with `lcr.paths`, ar-reviewer mode code), P10 after P2.
3. Research cycle 19 by hand from `../autoresearch/prompts/research_steps.md`; its GPU runs wait for the safety work and the user's go.

## Decisions waiting for the user
In experiment_execution/DECISIONS.md: D3 folder rename, D4 fast deployment variant, D5 purge of `_to_delete/` (also the radar archives in `~/downloads/nuscenes_radar_456/`, 632 MB, no longer needed), D9 launcher identity check (done in P5, to be ticked), D10 rebuild of the b123 exact-distance map (TF32 errors in a training target).

## Pointers
Problem: PROBLEM.md; method: ../autoresearch/METHOD.md; closed questions: experiment_execution/closed_questions.md; literature: literature/OVERVIEW.md; old state: experiment_execution/history/; git: one repository (parent of this folder), code and text records, a tag per run (`lcr/<id>`), remote `origin` = github.com/Archimedes79/Auto-research (pushes only on request).
