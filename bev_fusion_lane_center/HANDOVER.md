# HANDOVER - 2026-10-31 14:00

New chat: start in the repository folder (parent of autoresearch/ and this folder), say "Read HANDOVER.md and continue". Rewrite this file at every milestone (<= 3 KB).

## Who does what
- **Lead of loop 21 and owner of the lane-project experiments, code, PLAN, schedule, HANDOVER: this chat** (prompt: experiment_execution/history/prompts/PROGRAM_loop21.md; TEST closed, no mail, no shutdown). The tools chat owns `autoresearch/tools`. The "BEV system" chat (session local_ef1c1a32-671e-4989-9e6a-a1d4fe88f515) runs a ~20 h training lane (runs/t2_*, 12 GB GPU): GPU-free iterations only, no idle-GPU timing.

## Where we are
- Loop 21 (CPU only, cached raw outputs): E257 Douglas-Peucker output stage (D20 adopted: default-off `assemble(dp_eps)`, registry `fast_recall_dp` / `champion_dp` / `recall_dp`), E258 profile (`fast_recall_na`), E259 clean-up stays, E260 dead options removed (graph hashes identical). Q stays 0.735 - 0.740.
- **Review 2026-10-31 (fresh ar-reviewer, "correct with caveats"):** (1) the stage lowers the node-id count (15.0 -> 9.9) but NOT the count on the 14.3 budget's basis (per-piece DP 0.3 m: fast_recall 12.24 -> 12.23, champion 11.32 -> 11.32, recall 12.82; `experiments/E257_*/vertex_basis.txt`): it is a re-count, Q cannot change, S rises only through its sparsity term; (2) E257 KEEP was by the pooled mean, the champion misses the F1@0.5 guard at eps 0.3 and eps 0.2 was chosen after the sweep; (3) loop-21 timings are above the load rule (E258 step 2 load up to 9, E259 up to 15): relative and provisional only; (4) E258 as carded never ran its rule, `fast_recall_na` is a post-hoc per-family entry (J -0.004 .. -0.007). Fixed in the records: E258 header, E259 slips, registry text. Open: PROBLEM.md line 18 compares a node-id count with the budget (protected: needs a grant naming a decision, ask Mirko).
- Status report with images and intermediate network outputs: `reports/weekly/Status_after_Loop21_EN.pdf` (builder `reports/build/reports/build_status21_report.py`, figures `temp/loop21/fig_status_*.py`).
- Open decisions for Mirko: D4 fast variant, D5 purge, D17 (`project.json` loops: loop 21 not in it), D18 (per-metric bands), D19 (machine lease).

## Next
1. Idle-window timing (E253 driver) with `fast_recall_na`, `*_dp`, E259 stage ranking, once the BEV lane pauses; only then identical-graph speed-ups of `relocate_forks` / `anchor_junctions_fast`.
2. Route success and map assignment on simplified graphs (needs route scores carried to the simplified polylines); decide whether the stage becomes the default.
3. Coverage gap (lanes found 0.77 vs 0.80, R@1 far 0.77 vs 0.81, route success on TEST 0.85 vs 0.89): candidates in PLAN.md (seed-ensemble ceiling on cached outputs, GPU items when a slot is free). Commit by path (the BEV chat's files are not mine).

## Pointers
PROBLEM.md; METHOD.md; closed_questions.md; literature/OVERVIEW.md; old state in experiment_execution/history/; git: one repository, tag `lcr/<id>` per run, remote origin = github.com/Archimedes79/Auto-research (pushes only on request).
