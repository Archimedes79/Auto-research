# HANDOVER - 2026-10-25 11:00

New chat: start in the repository folder (parent of autoresearch/ and this folder) or in this folder, say "Read HANDOVER.md and continue". Rewrite this file at every milestone (<= 3 KB).

## Where we are
- Loop 18 is CLOSED (stopped at the user's request 10:55, before STOP_AT 12:00). Nothing runs, no GPU job, no shutdown was done.
- Report: reports/weekly/Loop18_Fork_Pruning_Localisation_Report_EN.pdf (12 pages, 3 numbers reviews); mailed to mirko.meuter@aptiv.com 03:59 and 10:13 (v2); v3 on disk adds one sentence (j1eq8), not mailed.
- Results (VAL 873 frames; tree and status per node: experiment_execution/PLAN.md; every run: LEDGER.md, folders experiments/E209-E233):
  - fork pruning (K190) closes the J gap: J 0.265 -> 0.409 (control 0.408), Q 0.701 -> 0.735 (control 0.737)
  - registration + trust (K188 / K189): map error 2.5 -> 0.8 m, wrong map off at trust 0.75
  - numba assembly (K191): chain 26.1 -> 17.5 ms (j1 17.2 ms), control 628 ms
  - joint E2E fine-tune: negative three times (E202, E214, E232); the failure-trained route head is the lever (+0.13 ... +0.37), behind the ORIGINAL builder
  - recall weight buys lanes found (+0.06) and far recall, not Q (equal-budget controls, E219); fast j1rc adopted on VAL only; with three cameras off its Q is -0.038 (E231.a): j1eq is the safer fast variant
  - TEST used once (E230): pruned champion S 0.748 vs control 0.702

## Next step
Research cycle 19 with the new system (`../autoresearch/`, prompt `autoresearch/prompts/research_cycle.md`): P1 research on the gaps of PROBLEM.md (coverage at range, connectivity, planner-level metric, map error 1-3 m) and the foundation topics F1-F3, then P2 hypothesis cards -> PLAN.md + schedule.csv for loop 19. Evaluate with `tools/env/_py14.sh evaluate.py <name>` (models/registry.json; new metrics topo F1, R@1 far: champion vs control -0.022 / -0.102).

## Decisions waiting for the user
All in `experiment_execution/DECISIONS.md` (the only channel): D1 radar parts 4-6 (doubles the data), D3 folder rename to C:\Auto-research-lmc (script prepared: `temp/rename_repo_folder.ps1`), D4 fast deployment variant (champion decision with its TEST check), D5 purge of `_to_delete/`, D6 start of roadmap M0 + M1. Owner of the code (lead chat): none while no loop runs.

## Pointers
Problem: PROBLEM.md; method: ../autoresearch/METHOD.md; closed questions: experiment_execution/closed_questions.md; literature: literature/OVERVIEW.md; old prompt, handover and state: experiment_execution/history/ (prompts/PROGRAM_loop18.md, HANDOVER_2026-10-25_loop18.md, indexes/LOOP18_STATE.md, loop18_backup/); launchers temp/launchers/loop18/; git: one repository (parent of this folder) with code and all text records, no binaries or data, a tag per run (`lcr/<id>`); remote `origin` = github.com/Archimedes79/Auto-research-lmc (private, main pushed 2026-10-25, tip af34624; later pushes only on request); roadmap of the system: ../autoresearch/ROADMAP.md.
