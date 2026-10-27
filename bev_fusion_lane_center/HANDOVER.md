# HANDOVER - 2026-10-27 08:00

New chat: start in the repository folder (parent of autoresearch/ and this folder), say "Read HANDOVER.md and continue". Rewrite this file at every milestone (<= 3 KB).

## Who does what
- **Lead of loop 20 and owner of the lane-project experiments, code, PLAN, schedule, HANDOVER: this chat** (prompt: experiment_execution/history/prompts/PROGRAM_loop20.md; TEST closed, no mail). The tools chat owns `autoresearch/tools`. The "BEV system" chat (session local_ef1c1a32-671e-4989-9e6a-a1d4fe88f515) finished t1_r4_soa at 07:19 and does CPU work only; it announces any further GPU round first. Mirko asked for a shutdown at the end of the loop: not done, see the last report.

## Where we are
- Loop 20 done: E252 (map-offset cap not supported), E240 four pairs (TIE, Q +0.0062, seed SD Q 0.0030 / 0.0020), E241 pair 3 (cut test 4 of 4, KEEP = worth a 12-pairing replay), E255 (Poisson set loss CUT, Q -0.073 at 14.3 vertices), E256 (D4 candidates by city: no flip, order not established). E253 timing ran 07:20 after the BEV chat's training ended: valid session, TIE (j1 - f1 -0.23 ms), j1rc +2.65 ms over j1eq, recursion +6.1 ms.
- Week 3 report built and numbers-reviewed: `reports/weekly/Week3_Loops18-20_Report_EN.pdf` (builder `reports/build/reports/build_week3_report.py`, figures `fig_w3_results.py`; PDF needs absolute paths in `print_weekly.ps1`; delete the html).
- Open decisions for Mirko: D4 fast variant, D5 purge, D17 (`project.json` loops), D18 (per-metric noise bands), D19 (machine lease). Loader default is the exact distance map (D14).

## Next
1. The 12-pairing E241 replay only if D4 needs it; nothing else is queued.
2. Tools chat items: N15 (cross-pairing cut test in decide.py, duplicate ids in schedule.py add), N1 / D19 lease.
3. Commit by path (the BEV chat's files are not mine).

## Pointers
PROBLEM.md; METHOD.md; closed_questions.md; literature/OVERVIEW.md; old state in experiment_execution/history/ (plans/PLAN_loop19.md); git: one repository, tag `lcr/<id>` per run, remote origin = github.com/Archimedes79/Auto-research (pushes only on request).
