# HANDOVER - 2026-11-12 10:10

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: plain terms. D57 + D61: lane first. GPU only in his windows (nights 22-07, Mon-Fri 08:30-17:30). D60: Mirko away -> work alone. No TEST, push, mail. Lane project is mine (D80). **Stay differentiable (D55); GaussianCaR features are the foundation (D50).**
**Mirko 2026-11-12 ~08:50: continue, be ambitious; SHUT DOWN THE COMPUTER around 17:30 after a round** (allowed in chat; first check no other chat / job runs, save handover + memory, `shutdown /s /t 120`).

## Where we are
Report (private page, republish same file path): https://claude.ai/artifact/KpXQ2mrXkDjUtwBdJxFXeA (source results/Junction.report/, build `temp/jr/build_junction_report.py --media`, publish with root + files list of img/ vid/).
Results: E286 maps KEEP (kp_f1 0.64 vs 0.39); pair heads E287 / E308 / E305 TIE when the connectors are drawn from the keypoints. **E311 KEEP: connector ends snapped to the delivered pieces, p >= 0.1: topo F1 +0.023..+0.036 over the plain graph, lanes found +0.06..+0.09, junction F1 +0.02..+0.06, duplicates -0.04** (E287 p2, E305, E303; both seeds; cards E309-E311). E310: maneuver coverage (`evaluation/jmaneuver.py`): plain 52.5 %, p>=0.1 61.6 %, perfect pairs 64.3 %; the pair decision is saturated, the ENDS are the cap (52 % of GT pairs have both ends matched) -> E307 / E304 / E312.

## Running unattended (WSL, logs in temp/)
`temp/day4/day4.sh` (log temp/day4/day4.log): E303 done (TIE on its rule; learned handle scales +0.006..+0.011 topo F1 on the snapped read-out: E303 snapped p>=0.1 +0.0320 / +0.0356 over the plain graph, the best so far), E307 narrow end heat training since 10:57 (ends ~12:00), then E304 far range (~13:05); the E306 / E298 / E302 slots are stubs (originals `.orig`). Then `temp/day5/day5.sh` (log temp/day5/day5.log, waits for DAY4_DONE): card E312 per lane: maps m2 (sigma 1.0 + far-w 2) -> E305-recipe pair head with `--e2e --scales` on them (BFLC_SIG=1.0) -> dump, scorer, snapped scores p>=0.1/0.5, PR vs E305, maneuver; ends ~15:30. Host RAM: two dumps beside two trainings die with "Cannot allocate memory": redo one at a time (`temp/day4/redo_e303_s1.sh` is the pattern). Never edit a script a chain runs. CPU side jobs max 3.

## Next
1. Read E307 / E304 by their cards (jmaps_eval, `kpf1_maps.py`), then E312 when it lands (rule in the card; reference points E303 +0.0320 / +0.0356, E305 +0.0304 / +0.0285 on the snapped p>=0.1 read-out).
2. Update the report (builder `temp/jr/build_junction_report.py`; galleries: `temp/jr/media/errors_snap/` E305, `errors_snap_e303/` E303 snapped with scales; publish with root + files list, same file path) and send the key images.
3. ~16:45: final report, memory, this file; ~17:30: check ListAgents / no running jobs of other chats, `shutdown /s /t 120` (Mirko's order of 2026-11-12).
Open for Mirko: D4, D5, soft step E279, TEST once for a finished design, papers.csv merge, the full `GtGenerator selftest`, scratch `temp/gt_l6*` (delete only via `package.py`), joint arm jj, E306 / E298 / E302 (cards kept, not run).
