# HANDOVER - 2026-11-12 10:10

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: plain terms. D57 + D61: lane first. GPU only in his windows (nights 22-07, Mon-Fri 08:30-17:30). D60: Mirko away -> work alone. No TEST, push, mail. Lane project is mine (D80). **Stay differentiable (D55); GaussianCaR features are the foundation (D50).**
**Mirko 2026-11-12 ~08:50: continue, be ambitious; SHUT DOWN THE COMPUTER around 17:30 after a round** (allowed in chat; first check no other chat / job runs, save handover + memory, `shutdown /s /t 120`).

## Where we are
Report (private page, republish same file path; v4 published 13:30): https://claude.ai/artifact/KpXQ2mrXkDjUtwBdJxFXeA (source results/Junction.report/, build `temp/jr/build_junction_report.py --media`, publish with root + files list).
**System of the day (E312 KEEP): E307 maps (heat sigma 1.0 m: kp F1 0.72 vs 0.63) + pair head `--e2e --scales` (E305, E303) + connector ends snapped to the delivered pieces at p >= 0.1 (E311): topo F1 +0.057 / +0.062 over the plain graph (the pair head alone from the keypoints: +0.005), lanes found +0.11 / +0.13, junction F1 +0.05 / +0.07; maneuver coverage 52.5 % -> 70.7 % (`evaluation/jmaneuver.py`); per junction frame 0.52 of the GT pairs (before 0.19).** E313 (sigma 0.7) KEEP on the maps (kp F1 +0.02..+0.03, isolated recall -0.03). Cards E285-E314 in experiments/.

## Running unattended (WSL, logs in temp/)
`temp/day9/day9.sh` (log temp/day9/day9.log, flag DAY9_DONE): E314 = the E312 pair head on the E313 maps (run ids E314.p_s0/1, started 14:12, ends ~15:30), post: dump (flock), snapped p>=0.1 score first, record scorer, PR vs E312, maneuver. After it, if time (start <= 16:30): E298 on the E312 system (driven weight; `temp/e298/start_driven_e312.sh`, card amended; day8 chain script is the pattern). Host RAM: dumps one at a time (flock). Never edit a script a chain runs. CPU side jobs max 3.

## Next
1. Read E314 by its card (`temp/gt2/e312_table.py` pattern; control E312 snapped p>=0.1: +0.0566 / +0.0615), rebuild + publish the report (`temp/jr/media/errors_snap_e312` is the after-gallery; make a new one if E314 wins: `temp/gt2/run_gallery4.sh` pattern, GAL_SNAP=1).
2. ~17:15 final report, memory, this file; 17:30: no other chat / job running (ListAgents, `pgrep`), then `shutdown /s /t 120` (Mirko's order of 2026-11-12).
Open for Mirko: D4, D5, soft step E279, TEST once for a finished design, papers.csv merge, the full `GtGenerator selftest`, scratch `temp/gt_l6*` (delete only via `package.py`), jj, E306 / E302 (cards kept, not run), E298 (amended, may not have run).
