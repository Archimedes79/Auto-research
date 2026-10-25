# tools - the few reusable helpers

Python tools run through `tools/env/_py14.sh <script>` (conda env `gcar`; the script is found by name; `PYTHONPATH` covers all code folders). Code of the system is in `training/`, metrics in `evaluation/`, report builders in `reports/build/` - not here.

| tool | use |
|---|---|
| `env/_py14.sh`, `_pyr.sh`, `bflc_env.sh` | the launchers: run a script in `gcar` (`_pyr.sh` retries on CUDA 'unknown error'); `bflc_env.sh` sets the module search path (sourced by both). `export_env.sh` -> `environment_gcar.txt` / `.yml` (package list of 2026-10-24) |
| `env/setup_history/` | one-time scripts that built the machine, kept to rebuild it: `setup_gcar_env.sh`, `gcar_nvcc.sh`, `gcar_build_rast.sh`, `gcar_patch_rast.sh` (GaussianCaR env + rasteriser), `crn_env.sh`, `crn_fetch.sh`, `setup_env.sh` (untested BEVFusion draft), `patch_bevfusion.sh`, `setup_wsl.ps1`, `fix_dns.sh`, `fix_ipv4.sh`, `download_nuscenes.ps1`, `extract_blob.sh`. Not run day to day. `env/legacy/`: old wrappers of the former `bev` env, nothing calls them. |
| `experiment/run.py`, `ledger.py` | launcher with run record, log, ledger row, `done` flag only on success, GPU guard, retry on CUDA errors; ledger render / note / add. `experiment/selftest.py` tests both. See `../experiment_execution/README.md`. |
| `viz/render.py` | result images of lane graphs: `render_frame`, `render_compare` (several methods, same frame), `render_grid`, `frame_array`, `montage`; built on `viz_graph.py` (arrows, split triangles, merge diamonds, raster background). `viz/selftest.py` |
| `video/make_video.py` | result videos: `graphs_to_video`, `frames_to_video`, `side_by_side`; mp4 (OpenCV) or gif by file extension; no ffmpeg needed |
| `catalogue/` | experiment catalogue and table: `e2e_catalogue.py` (what / result / tag per experiment), `e2e_basis.py` (building blocks, K ids), `build_experiment_table.py` (-> `experiments/EXPERIMENT_TABLE.*`, `experiment_execution/plans/Experiment_Table_EN.md`), `p16_catalogue.py`, `dump_catalogue_json.py`, `lit16_merge.py` (literature cards) |
| `maintenance/` | `make_structure.py` regenerates `../STRUCTURE.md` and `../models/REGISTRY_ALL.md`; `resolve_links.py` turns `[[GR01]]` tokens in literature notes into links |

Add a new reusable helper here with a docstring that says how to call it, a selftest if it computes something, and a row in this table. One-off scripts go to `../temp/`. The scripts of the folder migration of 2026-10-24 (`mig_*.py`, move log) are in `C:\ML-Experiment\restructure\` and move to `../experiment_execution/history/migration_tools/` once nothing needs them.
