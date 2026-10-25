# HANDOVER - 2026-10-25 15:20

New chat: start in the repository folder (parent of autoresearch/ and this folder) or in this folder, say "Read HANDOVER.md and continue". Rewrite this file at every milestone (<= 3 KB).

## Where we are
- Loop 18 is CLOSED (10:55). Report: reports/weekly/Loop18_Fork_Pruning_Localisation_Report_EN.pdf. Results per node: experiment_execution/PLAN.md; every run: LEDGER.md; folders experiments/E209-E233. Headline: fork pruning closes the J gap (0.265 -> 0.409), chain 17.5 ms, joint E2E negative three times, TEST once (E230, S 0.748 vs 0.702).
- D8 (a) is carried out by the loop-18 lead chat ("BEV lane loop 18", owner of autoresearch/tools and evaluation/): M0 golden numbers (evaluation/golden.json), P5 launcher hygiene (locked atomic records, snapshot via a temporary index, git identity check, heartbeat, timeout, GPU slots), P1 experiment.json + `decide.py --experiment`; committed 4435cab / 9a3723e / ef5fe86, self-test passes.
- Data extension D7 (b) is DONE (15:08): GaussianCaR features of the 255 new scenes in `~/exp_gcar_b456/feats/` (E234.tr0 / tr1 / val, exit 0, post-check PASS); split file data/splits_b456.json (TRAIN2 new 7,961, UNSEEN 770, CONFIRM 1,567 frames). Ledger is free and the GPU idle. Steps 5-7 (ground truth, route / map, loader) are open: plan experiment_execution/plans/data_extension_b456.md.
- The research system was simplified today (7 agents, 5 templates, prompts/research_steps.md; roadmap and reviews in ../autoresearch/ROADMAP.md). No experiment of loop 19 has started (the user: system refinement first).

## Next step
1. Loop-18 lead: `evaluate.py --golden write champion --subset val250 --fresh --tol <measured>` (two runs for the tolerance), `decide.py --experiment ... --write` for the five re-encoded experiments, `tools/maintenance/run_all_tests.sh`.
2. Then P2 (`ar tick`), P4 (hooks on the protected files), P3 (`ar metrics`, loop 18 as baseline); order in DECISIONS D8. b456 steps 5-7 come with M1 / `lcr.paths` (shared data code, ar-reviewer mode code).
3. Research cycle 19 by hand from `../autoresearch/prompts/research_steps.md`; its GPU runs wait for the safety work and the user's go.

## Decisions waiting for the user
In experiment_execution/DECISIONS.md: D3 folder rename, D4 fast deployment variant, D5 purge of `_to_delete/` (also the radar archives in `~/downloads/nuscenes_radar_456/`, 632 MB, after b456 step 5), D9 launcher identity check (done in P5, to be ticked).

## Pointers
Problem: PROBLEM.md; method: ../autoresearch/METHOD.md; closed questions: experiment_execution/closed_questions.md; literature: literature/OVERVIEW.md; old state: experiment_execution/history/; git: one repository (parent of this folder), code and text records, a tag per run (`lcr/<id>`), remote `origin` = github.com/Archimedes79/Auto-research-lmc (pushes only on request).
