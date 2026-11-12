# HANDOVER - 2026-11-12 10:10

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: plain terms. D57 + D61: lane first. GPU only in his windows (nights 22-07, Mon-Fri 08:30-17:30). D60: Mirko away -> work alone. No TEST, push, mail. Lane project is mine (D80). **Stay differentiable (D55); GaussianCaR features are the foundation (D50).**
**Mirko 2026-11-12 ~08:50: continue, be ambitious; SHUT DOWN THE COMPUTER around 17:30 after a round** (allowed in chat; first check no other chat / job runs, save handover + memory, `shutdown /s /t 120`).

## Where we are
Report (private page, republish same file path): https://claude.ai/artifact/KpXQ2mrXkDjUtwBdJxFXeA (source results/Junction.report/, build `temp/jr/build_junction_report.py --media`, publish with root + files list of img/ vid/).
Results: E286 maps KEEP (kp_f1 0.64 vs 0.39); E287 pair head, E308 node tokens, E305 maps+pairs trained together: all TIE on the written topo-F1 rule when the connectors are drawn from the keypoints. **E311 (found today): snap the connector ends to the delivered lane pieces: topo F1 over the plain graph +0.015..0.02 at p>=0.5 and +0.026..0.030 at p>=0.1 (E308, E287 p2, E305 seed 0; seed 1 of p2 / E305 running), lanes found +0.06..0.07, junction F1 +0.03..0.04, duplicates -0.05, no guard broken** (card E311, rule written before the confirmation). E309: threshold 0.1 gives lanes found +0.06. E310: maneuver coverage (`evaluation/jmaneuver.py`, reviewed): plain 52.5 %, connectors p>=0.5 56.6 %, p>=0.1 61.6 %, perfect pairs 64.3 %, + perfect scales 67.1 %; keypoint ends are 1.44 m off, strict both-ends-found only 52 % of pairs, so scales (E303) and ends (E307, E304) beat the pair decision.

## Running unattended (WSL, logs in temp/)
`temp/day4/day4.sh` (log temp/day4/day4.log, flag DAY4_DONE): E303 handle scales (training since 09:25, ends ~11:00) -> E307 narrow end heat -> E304 far range -> E306 exact raster -> E298 driven weight -> E302 capacity; each run scored + dumped in the background. GPU starts only 08:30-16:30. Never edit a script a chain runs. CPU side jobs 1 thread each, max 3. Snap confirmation seed 1: `temp/gt2/snap_confirm2.sh` -> `temp/gt2/snap/*.json`; wiring sweep (radius / angle / thr 0.05) `temp/gt2/wire_run.sh` was stopped to free CPU, restart after.

## Next
1. Read E311 confirmation (p2, E305, seeds 0/1, thr 0.1/0.2/0.5) by the card's rule; ledger-free replay (CPU) -> E311 card result + decision; then promote the snapped read-out (critic pass on the diff of qends_eval.py / jends_eval.py running; code uncommitted: qends_eval.py, jends_eval.py, jmaneuver.py, README row, run_gates row).
2. Score E303 / E307 / E304 / E306 / E298 / E302 when they land (card rules; `jmaneuver.py` + `jends_eval.py score --snap`).
3. Update the report (new sections exist in the builder: failures, maneuver, snap, thr) and republish; new gallery with the snapped system `temp/jr/media/errors_snap/`.
4. Before 17:30: wait for the round, update memory + this file, check ListAgents / no running jobs, shutdown.
Open for Mirko: D4, D5, soft step E279, TEST once for a finished design, papers.csv merge, the full `GtGenerator selftest`, scratch `temp/gt_l6*` (delete only via `package.py`), joint arm jj.
