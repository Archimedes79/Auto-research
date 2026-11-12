# HANDOVER - 2026-11-12 08:30

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: plain terms. D57 + D61: lane first. GPU only in his windows (nights 22-07, Mon-Fri 08:30-17:30). D60: Mirko away -> work alone. No TEST, push, mail, shutdown. Lane project is mine (D80). **Stay differentiable (D55); GaussianCaR features are the foundation (D50).**

## Mirko's brief (D82 11-11 evening, D83) in keywords
GT first (looked at) -> 1 m pixel maps (entry / exit / zone / split / merge) -> 2 m nodes -> pruning with protected special types -> pairs keep ORDER (entry 1 -> exit 1) -> probability per pair + splines, driven weighted higher -> time -> map link / road-level SD map with lane counts. TIE = keep and refine, never worse. Standing order: be ambitious, try a lot, final presentation (approach, architecture, numbers, qualitative results, images, videos).

## Results so far (cards have the tables)
E286 maps head KEEP (kp_f1 +0.18..+0.27 over B's event heat; base arm j1); jj (joint pyramid) stopped at step 1000 (11 s/step, not judged, ckpt kept). E287 pair head p0/p1/p2: TIE (topo F1 +0.00..+0.006 over the plain graph, lanes found +0.02, junction recall +0.04..+0.11; the written rule picks p1, follow-ups run on p2). E308 node tokens: TIE (AUC +0.01). **Finding:** on the SAME ends (both systems found them) the E284 head is much better than the E287 head (AUC 0.975 vs 0.94, recall at equal precision 0.67 vs 0.34): the E287 head ranks worse and says yes too rarely. Candidates: pair head not trained with the maps (E305), degree budget (E302), threshold (`temp/gt2/thr_sweep.py` diagnostic, VAL-tuned).

## Running unattended (WSL, logs in temp/)
night3b done. `day1/day.sh` (E304 far-range maps, E298 driven weight, E303 handle scales; two lanes) starts 08:30, then `day2/day2b.sh` (E307 narrow end heat, E306 exact raster, E305 e2e, E302 capacity + scores). Flags DAY1_DONE / DAY2_DONE. Never edit a script a chain runs. CPU side jobs: 1 thread (`temp/gt2/thr_run.sh` threshold sweep on E308 dumps -> `temp/gt2/thr/`).

## Next
Read E304 / E298 / E303 / E307 / E306 / E305 / E302 by the cards' rules (ledger notes, commit by path); if E305 or E302 closes the pair-head gap make it the base; report: `temp/jr/make_media.sh <maps run> <pair run>` then `build_junction_report.py --media`, view, send key images (GT figures exist in temp/jr/media/gt); later E292 wiring, E295, E297, E299, E291, E294. Open for Mirko: D4, D5, soft step E279, TEST once for a finished design, papers.csv merge, the full `GtGenerator selftest` after the chains, scratch `temp/gt_l6*` (delete only via `package.py`), joint arm jj (one lane alone, resume).
