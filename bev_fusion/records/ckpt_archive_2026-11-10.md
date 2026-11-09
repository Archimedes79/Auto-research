# Archive of old checkpoints, 2026-11-10

Mirko ("Klar, räume auf", 2026-11-10; hard rule 5: checkpoints are moved, never deleted). 11 checkpoint files (1.8 GB) of old runs moved from `bev_fusion/runs/<run>/` to `D:\Auto-research\NuScenes_checkpoints\bev_fusion\<run>\` (the folder `bev_fusion/data/checkpoints/` links there). Each file was copied, compared by SHA-256 and size, and only then removed from C:. The hash list is `D:\Auto-research\NuScenes_checkpoints\bev_fusion\ARCHIVE_2026-11-10.csv`; each run folder keeps a `MOVED.txt` with the file, hash and target.

| run | file | MB |
|---|---|---|
| t1_r3_combined | ckpt_00006000.pt | 63 |
| t1_r0 | ckpt_00005000.pt | 62 |
| t4_fpn_order | ckpt_00003000.pt | 212 |
| t6_maps_only | ckpt_00003000.pt | 212 |
| t7_res4_s0 / t7_res4_s1 | ckpt_00003000.pt | 214 each |
| t7_rgb_s0 / t7_rgb_s1 | ckpt_00003000.pt | 213 each |
| t8_bevfeat_s1 | ckpt_00003000.pt | 212 |
| distill_resnet18_smoke | ckpt.pt, student.pt | 136 + 45 |

Kept in place on purpose: `t1_r4_soa` (golden test of the evaluation), `t8_ctl_s0` and `t8_bevfeat_s0` (selftest of `board.py`), the best three (`t8_ctl_s1`, `t10_dual_s0`, `t10_dual_s1`, `t11_vel_hurdle_acc2`) and every run of the t16 - t23 chain (`t20_base_s0` ckpt 6000 is the warm start of t21 - t24). The runs t2_*, t3_* had no checkpoint files left (slimmed or archived earlier). No code, test or config reads the archived files; the metric cards (`eval/`), logs and configs of these runs stay in `runs/`.

Restore one file: `copy D:\Auto-research\NuScenes_checkpoints\bev_fusion\<run>\<file> C:\Auto-research\bev_fusion\runs\<run>\`, then compare its hash with the csv.
