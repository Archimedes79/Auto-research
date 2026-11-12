p = "HANDOVER.md"
s = open(p, encoding="utf-8").read()
i = s.index("## Running unattended")
new = '''## Running unattended (WSL, logs in temp/)
`temp/day4/day4.sh` (log temp/day4/day4.log): E303 done (TIE on its rule; learned handle scales +0.006..+0.011 topo F1 on the snapped read-out: E303 snapped p>=0.1 +0.0320 / +0.0356 over the plain graph, the best so far), E307 narrow end heat training since 10:57 (ends ~12:00), then E304 far range (~13:05); the E306 / E298 / E302 slots are stubs (originals `.orig`). Then `temp/day5/day5.sh` (log temp/day5/day5.log, waits for DAY4_DONE): card E312 per lane: maps m2 (sigma 1.0 + far-w 2) -> E305-recipe pair head with `--e2e --scales` on them (BFLC_SIG=1.0) -> dump, scorer, snapped scores p>=0.1/0.5, PR vs E305, maneuver; ends ~15:30. Host RAM: two dumps beside two trainings die with "Cannot allocate memory": redo one at a time (`temp/day4/redo_e303_s1.sh` is the pattern). Never edit a script a chain runs. CPU side jobs max 3.

## Next
1. Read E307 / E304 by their cards (jmaps_eval, `kpf1_maps.py`), then E312 when it lands (rule in the card; reference points E303 +0.0320 / +0.0356, E305 +0.0304 / +0.0285 on the snapped p>=0.1 read-out).
2. Update the report (builder `temp/jr/build_junction_report.py`; galleries: `temp/jr/media/errors_snap/` E305, `errors_snap_e303/` E303 snapped with scales; publish with root + files list, same file path) and send the key images.
3. ~16:45: final report, memory, this file; ~17:30: check ListAgents / no running jobs of other chats, `shutdown /s /t 120` (Mirko's order of 2026-11-12).
Open for Mirko: D4, D5, soft step E279, TEST once for a finished design, papers.csv merge, the full `GtGenerator selftest`, scratch `temp/gt_l6*` (delete only via `package.py`), joint arm jj, E306 / E298 / E302 (cards kept, not run).
'''
s = s[:i] + new
open(p, "w", encoding="utf-8").write(s)
print(len(s))
