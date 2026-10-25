# HANDOVER - 2026-10-25 11:00

New chat: start in C:\ML-Experiment, say "Read HANDOVER.md and continue". Rewrite this file at every milestone (<= 3 KB).

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
Deployment choice, then the open gaps: lanes found beyond 40 m (partly a verifier trade, E233), registration on the perceived graph, validity calibration. Evaluate with `tools/env/_py14.sh evaluate.py <name>` (models/registry.json).

## Decisions waiting for the user
- deployment config (idle RTX 3060): j1eq 17-19 ms (Q 0.735) / j1rc 20-22 ms (Q 0.739, lanes found 0.736) / f1 + time recursion 24.2 ms (Q 0.735) / rc1 (Q 0.740, lanes found 0.767)
- OK for a TEST run of j1rc or j1eq (this loop's TEST grant is spent)
- purge of _to_delete/ and CLEANUP_LIST.md items

## Pointers
Prompt PROGRAM.md; literature: literature/OVERVIEW.md (the literature block of the old handover is in the history copy); memory note bev-lane-loop18-results; old handover and state: experiment_execution/history/ (HANDOVER_2026-10-25_loop18.md, later LOOP18_STATE.md); code backup of the loop: experiment_execution/loop18_backup/; launchers temp/launchers/loop18/; git is code-only (CLAUDE.md).
