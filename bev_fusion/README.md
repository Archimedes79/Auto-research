# bev_fusion - discussion folder: camera + radar BEV system

For now this folder only holds the discussion of a new BEV system (camera pixel-group cones, a learned radar density per detection, temporal fusion, semantic head) with Mirko. It is not yet a problem folder of `../autoresearch/` (no ledger, no runs); that comes when the plan is agreed.

| file / folder | content |
|---|---|
| `plan.md` | the current plan (draft 3) with the paper and code basis per block, the GT generation and viewer, and the tests |
| `radar_encoding.md` | radar grid encoding ideas (measured nuScenes radar statistics, Mirko's per-detection decoder, the symmetric polar radar image + table pull, comparison, tests); discussed separately until one is chosen |
| `plan_draft2.md` | the previous draft (front camera, virtual upright camera, sigma points), kept for the record |
| `training/` | code (plain PyTorch, a self-test per module): the blocks (camera lift, camera encoder with pre-trained DualBEV heads, radar block, splat, temporal fusion, fusion, BEV trunk, time-decoded heads, object head), the paper fallbacks, the losses, the nuScenes loader and the whole-chain assembly `model.py` - see `training/README.md` |
| `evaluation/` | metric code (segmentation / line IoU with tolerances, nuScenes-style detection AP and TP errors, latency) with known-answer self-tests - see `evaluation/README.md` |
| `tests/` | synthetic gates per block (`tests/run_gates.sh`) - see `tests/README.md` |
| `models/` | weights (not in git): `pretrained/dualbev-r50-cbgs.pth` with source and checks - see `models/README.md` |
| `literature/papers.csv`, `literature/papers/` | the papers of the scans of 2026-10-25 (lift, radar, temporal, heads, SDF, polar radar; 68 notes), moved here from `../bev_fusion_lane_center/literature/` |
| `literature/code/front_bev_code_sources.md` | repository, licence and module per block (read on the repository pages) |
| `temp/scout_tables/` | the raw tables of the scouts |
| `temp/setup/` | one-off scripts of this discussion (calibration check, table merges, the move out of the lane project) |

Shared data (nuScenes) is described in `../datasets/NuScenes/README.md`.
