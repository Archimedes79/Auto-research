# The restructure of 2026-11-26 (D154 - D156): status

Carried out and merged into `main` (D156). Done: S0 structure, S1 the core `research/` with the toy as its chain test, S2 the tag `v1`
(the old system with its uncommitted records), S3 data and runs on D:, S4 BEV and S5 lane (the champions' code moved, everything else left
`main`), S6 the old folders and tools retired from `main`. Open: the row on C: of section 13 and the BEV champion
(`../problems/bev_fusion/DECISIONS.md`). The system as it is: `README.md`, `METHOD.md`, `research/README.md`. The plan with the reviews it
answered: this file in git before it was shortened (`git show b06e616f:autoresearch/INTEGRATION_2026-11-26.md`).

## 13. Measured, kept, deleted (2026-11-27, D155)
**Measured through the research system** (VAL; the records are in the problems' `experiments/`):
| what | result |
|---|---|
| lane E396 (measure): the champion E372 S4, seeds 0 and 1, from its snapshot (dump step + evaluation step) | topo F1 0.668172 / 0.681281, junction F1 0.2538 / 0.2481, Q 0.6796 / 0.6808: the numbers of its v1 cards to the last digit; champion under D120 |
| lane E397 (compare): keep threshold 0.5 against 0.6, the control reused from E396 | TIE: gain -0.0026, paired scene bootstrap [-0.0039, -0.0017], band 0.01; the guard lanes_found breaks (-0.0123); the decode stays |
| BEV E27 / E28 (measure; imported weights, warm evaluation on ground truth v13, `spread:1000:val`) | `t21_ctl` board7 0.4631 / 0.4635 (seed 0 / 1), `t20_base_s0` 0.4569, inside the band 0.02; the older 0.4549 / 0.4465 are cold reads on ground truth v8, not comparable |

**Kept.** Raw nuScenes (`D:\Auto-research\NuScenes`, 121 GB, never deleted). In the data root `D:\Auto-research\data`: `lane_center/b123`
(30.1 GB: the B' features of train and val, the val rows hold the 833 TEST rows too, graph ground truth, poses, PCA),
`groundtruth/lane/{9ad1d4ce30,8a04143278}`, `groundtruth/a107e0b4e8` (front v13, copied from C:: 34,150 files, bytes equal),
`groundtruth/cache` (the reader's tables), `bev_fusion/` (the caches the BEV champion reads: ground truth v13, camera features, BEVFusion
features, radar, front images, the annotation table; the pretrained DualBEV and BEVFusion weights), `nuscenes/` (the view: links to the raw
data and the CRN info files). In the runs root `D:\Auto-research\runs`: `lane_center/` (the champion chain E292 a, E273 p, E271 a, E373,
E372 S4 with seeds 0 and 1, the caches E365 and E357, the refiner statistics, 6.9 GB; the runs of E396 and E397) and `bev_fusion/`
(`t21_ctl_s0`, `t21_ctl_s1`, `t20_base_s0`). In WSL: `~/work` (GaussianCaR code and weights, the BEV comparison network) and the conda
environments. In `D:\Auto-research\archive`: `lcr_local_tags_2026-11-24.bundle` (25 MB, the code of the old run tags),
`temp_scratch_2026-11-27.tar.gz` (35 MB, 1,250 files of `temp/`).
The best three networks of both problems that were only on C: are copied, hashes checked, to `D:\Auto-research\archive\v1_best_networks`.

**Deleted** (listed here before; nothing of it is read by the code), done for WSL and D: (D: free 271 -> 311 GB):
| where | what | size | why |
|---|---|---|---|
| WSL `~/exp_gcar_b123` | the old lane pipeline's features, datasets and failure series; the 30 GB the chain reads were copied to `data/lane_center/b123` first | 164.8 GB | old experiment data |
| WSL `~/exp_gcar_b456`, `~/exp_crn_b123`, `~/exp_crn_mini`, `~/exp_gcar_mini`, `~/exp_gcar_b123x` | old feature extractions (GaussianCaR, CRN) | 80.4 GB | old experiment data |
| WSL `~/mutk`, `~/mutk2`, `~/head_check`, `~/exp_gcar_gtchk`, `~/_verify_v4` - `_v7`, `~/_review_v6_b`, `_c`, `~/critic_fixes_tmp` | scratch of old checks | 6.9 GB | scratch |
| D: `lane_archive_2026-11-01` | the heavy files of the lane runs E19 - E72 | 19.9 GB | old experiment data |
| D: `NuScenes_checkpoints/bev_fusion` | superseded BEV checkpoints | 6.7 GB | no champion chain |
| D: `NuScenes_cache/bev_fusion/cam_feats_parts7_10` | feature shards of the 340 scenes of parts 7 - 10 (only 700-scene runs read them) | 10.4 GB | a cache nothing kept reads |
| D: `backup_best_network_2026-11-10/t21_ctl_s0` | a copy of `runs/bev_fusion/t21_ctl_s0` (shared files checked identical; code snapshot and notes moved into the kept run as `backup_2026-11-10/`) | 0.4 GB | duplicate |
| D: `NuScenesAdditionalGt/c643a7539d`, `lane/{0702a065c6,3576fa1f7e,db288acca5}`, `examples`, `reports` | front ground truth v8, older lane ground truth, generated pictures | 3.8 GB | older versions (D40) |
| C: `bev_fusion/`, `bev_fusion_lane_center/` (untracked rest), `data/autogt` (links; v13 moved first), `temp/` | the old problems' data, runs, results and scratch; `temp/` (except the worktree) is packed and removed | about 178 GB | the folders left the tree with the merge; their untracked rest is still to do |

The WSL disk image does not shrink by itself after a deletion: compacting it (`wsl --shutdown`, then `Optimize-VHD` or
`diskpart compact vdisk` on `wsl\ext4.vhdx`) is the owner's step.
