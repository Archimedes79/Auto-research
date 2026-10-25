# Loop 11 (2026-10-18): temporal blocks around the learned chain, route threshold selected on VAL

Report: `reports/LaneGraph_Condensed_Report_EN.pdf` sections 3.17, 7.25 - 7.30, cards E105 - E107, 8.7. Bundle: `bundles/E2E_LaneGraph_Bundle_v4_light_2026-10-18(.zip)`. Reviews: `report_final/review11/`.

| experiment | scripts | results (out/loop2/) |
|---|---|---|
| E104 revisited (route threshold) | `p10_route_eval2.py --kind x --taus ...` | `e104b/route2_val_sweep.json` (VAL, thresholds 0.3 - 0.7), `e104b/route2_test_tau.json` (TEST at 0.5 / 0.55 / 0.6) |
| E105 temporal memory of the input | `p11_temporal.py` (variants single, ema_all:a, ema_maps:a, hold_mon, ema_hold_mon:a; `--odo`, `--odo_t`, `--mon heat|fwd|quad`, `--tokens`), `p11_eval.py` (F1, scene bootstrap) | `e105_a` (graphs_<pert>_<variant>.pkl, comps_*.npy, eval.json), `e105_odo1`, `e105_odo3` (night, odometry error 1 deg/sqrt(s) + 2 %, 3 deg + 5 %), `e105_monfwd`, `e105_monquad` (exploratory monitor signals), `e105_dbg` (one-scene debug) |
| E106 memory of the route attribute | `p11_route_time.py` | `e106_a/result.json`, `e106_dbg` (kernel-width trial: s = 1.5 m, threshold 0.6, cell s2) |
| E107 whole chain under failure | `p11_temporal.py --tokens`, `p11_route_fail.py` | `e105_tok` (graphs with tokens), `e107_a/result.json` |
| figures, tables, cards, report | `fig_p11.py`, `make_loop11_tables.py`, `_mkcards11.py`, `build_final_report11.py`, `_p11_buildrep.sh`, `_figs11.sh`, `_reeval11.sh` | `report_final/figures/p11_*.png`, `report_final/tables/tab_e104_tau, e105, e105_head, e105_odo, e106, e107`, `report_final/cards/cards_loop11.json` |
| bundle v4 | `_bundle_temporal.py` -> `learned_extractor/temporal.py`, `_gen_ref11.py`, `_patch_bundle11*.py`, `_build_bundle11.sh` | `bundles/E2E_LaneGraph_Bundle_v4_light_2026-10-18` |

Key numbers (TEST): route head E104b at the VAL-selected threshold 0.5: route success 0.853 / 0.843 / 0.832 / 0.820 at map error 0 / 1 / 2 / 3 m (loop 10 at 0.6: 0.813 / 0.815 / 0.795 / 0.792); failure-phase F1 at 1 m, single frame -> EMA 0.7 -> EMA 0.5 + hold: night 0.337 -> 0.428 -> 0.728 (rule pipeline 0.293), blur 0.524 -> 0.640 -> 0.699 (0.474), radar off 0.813 -> 0.849 -> 0.838 (0.829), three front cameras off 0.653 -> 0.675 -> 0.666 (0.666); route success of the chain in the night 0.093 -> 0.146 -> 0.346 (clean phase 0.847); route memory (E106) no gain.
Protocol exceptions: failure experiments have no VAL run, all variants evaluated on TEST; kernel width of E106 and the two exploratory monitors tried on TEST.
