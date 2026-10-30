# HANDOVER - 2026-10-30 12:00

New chat: start in the repository folder (parent of autoresearch/ and this folder), say "Read HANDOVER.md and continue". Rewrite this file at every milestone (<= 3 KB).

## Who does what
- **Lead of loop 21 and owner of the lane-project experiments, code, PLAN, schedule, HANDOVER: this chat** (prompt: experiment_execution/history/prompts/PROGRAM_loop21.md; TEST closed, no mail, no shutdown). The tools chat owns `autoresearch/tools`. The "BEV system" chat (session local_ef1c1a32-671e-4989-9e6a-a1d4fe88f515) runs a ~20 h training lane (runs/t2_*, 12 GB GPU at 100 % on 2026-10-30): GPU-free iterations only, no idle-GPU timing; it announces its GPU rounds.

## Where we are
- Loop 21 (CPU only, on cached raw outputs): E257 Douglas-Peucker output simplification (KEEP: 15.0 -> 9.0 vertices per 100 m at Q -0.001, eps 0.2 safer: 9.9 / 9.5, Q -0.0003 / -0.0005; D20 decided: adopted 2026-10-30 as default-off `assemble(dp_eps)` + `simplify_dp`, registry `fast_recall_dp` / `champion_dp` reproduced by evaluate.py, 9 spec tests, code review done, PROBLEM.md / PROTOCOL.md sentences; route / map pipelines (`aux=True`) stay on unsimplified vertices); E258 assembly profile (relocation 21 %, anchoring 16 %): relocation stays, anchoring off is free for the j1 family on four seeds -> registry entry `fast_recall_na` (saving ~1.3 - 2 ms estimated from relative timing on a loaded machine).
- Results video of `recall`: `reports/weekly/Lane_graph_recall_val.mp4` (made by `temp/loop21/make_results_video.py <registry name>`).
- Loop 20 results and the week-3 report: `reports/weekly/Week3_Loops18-20_Report_EN.pdf`; loop-20 plan in `experiment_execution/history/plans/PLAN_loop20.md`.
- Open decisions for Mirko: D4 fast variant, D5 purge, D17 (`project.json` loops), D18 (per-metric bands), D19 (machine lease).

## Next
1. Idle-window repeat of the timing with `fast_recall_na` (E253 driver, new entry) once the BEV lane pauses.
2. D20 follow-ups: route success and map assignment on simplified graphs (not possible with `aux=True`: needs a way to carry the route scores to the simplified polylines), a `recall_dp` entry, the ms of the stage on an idle machine.
3. Candidates in PLAN.md (seed ensemble of cached outputs, GPU items when a slot is free). Commit by path (the BEV chat's files are not mine).

## Pointers
PROBLEM.md; METHOD.md; closed_questions.md; literature/OVERVIEW.md; old state in experiment_execution/history/; git: one repository, tag `lcr/<id>` per run, remote origin = github.com/Archimedes79/Auto-research (pushes only on request).
