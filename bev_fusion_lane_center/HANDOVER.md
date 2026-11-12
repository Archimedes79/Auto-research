# HANDOVER - 2026-11-12 00:50

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: plain terms. D57 + D61: lane first. GPU only in his windows (nights 22-07, Mon-Fri 08:30-17:30). D60: Mirko away -> work alone. No TEST, push, mail, shutdown. Lane project is mine (D80). **Stay differentiable (D55); GaussianCaR features are the foundation (D50).**

## Mirko's brief (D82 11-11 evening, D83) in keywords
GT first (looked at) -> 1 m pixel maps (entry / exit / zone / split / merge) -> 2 m nodes -> pruning with protected special types -> pairs keep ORDER (entry 1 -> exit 1) -> probability per pair + splines, driven weighted higher -> time -> map link / road-level SD map with lane counts. TIE = keep and refine, never worse. Final: report with images + videos. D83: l6 merged.

## Done and committed (main, last a61ef288)
E285 GT layer + gates, E286 maps head, E287 pair head v2, E288-E290, E293, E296, E300, E301 (closed by data); code ready, default off: E298 driven weight, E303 handle scales, E304 far-range loss, E302 capacity (Sinkhorn + degree heads), E305 `--e2e` (maps + pair head together), E306 `--raster exact` (the legacy polygon fill is half a cell off: 36 % wrong cells, exact 2.3 %); `jpair_pr.py` precision at a fixed recall. Critic pass 1 fixed (painted exits, rank features from real peaks, strict one-to-one matching: rule control recall 0.36 / precision 0.34 vs head 0.29 / 0.74, ceiling 0.43). The critic once loaded TEST lane GT (counts only, discarded, nothing selected): disclosed, TEST log is Mirko's. Critic pass 2 (code of 11-12) running; E293: chord curve good 5 m deep only.

## Running unattended (WSL, logs in temp/)
`e286/night.sh` (E286 j2, then scores) -> `e287/night2.sh` (E287 p0/p1/p2 x2 seeds) -> `night3.sh` (joint jj + q2) -> `day1/day.sh` (E304, E298, E303) -> `day2/day2b.sh` (per lane E307 narrow end heat, E306 exact raster, E305 e2e, E302 capacity, then scores); `e289/night4.sh` line probe. Flags: `temp/e286/SCORED`, `e287/JJ_DONE`, `day1/DAY1_DONE`, `day2/DAY2_DONE`. GPU smokes of E302/E305/E306/E307 passed. Never edit a script a chain runs. E307 (found in the fixed E286 scorer): crowded ends are found 0.54 vs 0.66 isolated.

## Next
Read E286 / E287 results by the cards' rules (ledger notes, commit); `jsplits` / `jrule_control` / `jpair_pr` on the E287 dumps; E290 flips on a dump; report: `temp/jr/make_media.sh <maps run> <pair run>` then `build_junction_report.py --media`, view, send key images; later E292 wiring, E295, E297, E299, E291, E294. Open for Mirko: D4, D5, soft step E279, TEST once for a finished design, papers.csv merge (`literature/` digests), the full `GtGenerator selftest` after the chains, scratch `temp/gt_l6*` (delete only via `package.py`).
