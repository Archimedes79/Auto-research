# Loop 8 (2026-10-16/17): index of the stored experiments

Perception-first lane graph with assignment to a navigation (SD) map after the perception. Scripts in `C:\ML-Experiment\scripts` (prefix `p8_`), results in `out/loop2/<run>/result.json` (+ console logs in `out/loop2/p8logs/`), caches in WSL `~/exp_gcar_b123/p8/`. The first versions of the results, computed with the frames that have an empty perceived graph left out, are in `out/loop2/_subset_v1` (superseded; kept for traceability). The exact stage commands (thresholds, tags) are in `scripts/_p8_chainA.sh` and `scripts/_p8_chainB2.sh` (the sample rebuild in `scripts/_p8_rebuild.sh`).

| experiment | what | scripts | result runs |
|---|---|---|---|
| data | perceived loose graphs per split / condition; edge samples with SD draws, localisation, features, labels | `p8_build.py`, `p8_dataset.py` (`--plan train|dev|test|testq|testw`, `--srcprefix r` robust front end), `p8_lib.py`, `p8_seg.py` | `~/exp_gcar_b123/p8/*.pkl` |
| E81 | sparse graph, precision control (tracer, probability filter, learned keep) | `p8_sparse.py` | `e81_sparse`, `e81_sparse_ens` |
| E82 | rough localisation: lane-count-aware template, error tails, lane width, front ends | `p8_geo.py`, `p8_eval_loc.py`, `p8_lanewidth.py`, `p8_locstats.py` | `e82_geo_clean`, `e82_geo_clean3`, `e82_geo_pert`, `e82_loc`, `e82_lanewidth`, `e82_locstats` |
| E82b | held world-frame correction through a failure (constant map bias) | `p8_trackloc.py` | `e82_trackloc` |
| E83 | oracle unit analysis; assignment network, DEV study (23 runs), ensemble | `p8_oracle_unit.py`, `p8_assign.py`, `queue_e83.txt` (+ `queue_runner7.py`), `p8_ens.py`, `p8_eval_front.py` | `e83_oracle_unit`, `e83_*` (single runs), `e83_ens`, `e83_ensR` (robust front end), `e83_nosdT` |
| E83b-d | lane count / index; gate, abstention, calibration (also under failures); map quality (per-road bias); changed goal | `p8_attr.py`, `p8_misc.py`, `p8_eval_front.py --plan testq|testw`, `p8_alt.py` | `e83_attr`, `e83_misc`, `e83_ensQ`, `e83_ensW`, `e83_alt` |
| E84 | map extension (SD route + perceived offset) | `p8_pipe.py` (`extend_points`), `p8_stages.py --stage ext` | `e84_ens`, `e84_ens_tk9`, `e84_ensR`, `e84_ensQ` |
| E85 | temporal fusion on the graph level; raster against graph; sweep on VAL | `p8_pipe.py` (`Tracks`), `p8_stages.py --stage temporal`, `p8_early.py` | `e85_dev`, `e85_ens`, `e85_ens_t5`, `e85_ensR`, `e85_early*` |
| E86 | reference path against the driven trajectory | `p8_pipe.py` (`reference_path`), `p8_stages.py --stage plan` | `e86_ens`, `e86_tk9` |
| E86b / E86c | learned consumer; joint training | `p8_consumer.py`, `p8_joint.py` | `e86b_consumer`, `e86c_joint` |
| timing | run time per stage | `p8_timing.py` | `e86_timing` |
| report | tables, figures, builder | `make_loop8_tables.py`, `fig_p8.py`, `fig_p8b.py`, `arch_figs8.py`, `build_final_report8.py` | `report_final/` |
| literature | scans of loop 8 | (agents) | `report_final/lit8/`, `report_final/text/lit_extra8.md` |
| reviews | numbers review and fresh-eyes review (two rounds) | (agents) | `report_final/review8/` |
| bundle | code bundle of the best method | (agent) | `bundles/E2E_LaneGraph_Bundle_v1_2026-10-17` (+ light) |
