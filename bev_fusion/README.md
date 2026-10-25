# bev_fusion - discussion folder: camera + radar BEV system

For now this folder only holds the discussion of a new BEV system (camera pixel-group cones, a learned radar density per detection, temporal fusion, semantic head) with Mirko. It is not yet a problem folder of `../autoresearch/` (no ledger, no runs); that comes when the plan is agreed.

| file / folder | content |
|---|---|
| `plan.md` | the current plan (draft 3) with the paper and code basis per block, the GT generation and viewer, and the tests |
| `plan_draft2.md` | the previous draft (front camera, virtual upright camera, sigma points), kept for the record |
| `training/` | code (plain PyTorch, self-tests): `lift_pd.py` (camera lift) - see `training/README.md` |
| `literature/papers.csv`, `literature/papers/` | the papers of the scans of 2026-10-25 (lift, radar, temporal, heads, SDF, polar radar; 68 notes), moved here from `../bev_fusion_lane_center/literature/` |
| `literature/code/front_bev_code_sources.md` | repository, licence and module per block (read on the repository pages) |
| `temp/scout_tables/` | the raw tables of the scouts |
| `temp/setup/` | one-off scripts of this discussion (calibration check, table merges, the move out of the lane project) |

Shared data (nuScenes) is described in `../datasets/NuScenes/README.md`.
