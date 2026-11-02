import io
p = "experiments/README.md"
s = open(p, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(s) if l.startswith("| `t11_vel_hurdle/`"))
s.insert(i, "| `t12_motion_block/` | does a two-frame + radar-Doppler velocity field (the fused flow pyramid of plan 7.4, `training/motion_block.py`), trained ALONE on the frozen `t10_dual_s0`, beat the base's own velocity (time mover head, box velocity) on the same VAL frames? replaces `TimeMover.motion`; controls v = 0, base, `t11_vel_hurdle` box velocity; 2 seeds | card 2026-11-01; code and self-test built; runs t12.mb_s0 / s1 |")
open(p, "w", encoding="utf-8").write("\n".join(s))
p = "training/README.md"
s = open(p, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(s) if l.startswith("| `train.py`"))
s.insert(i, "| `train_motion_block.py` | t12: trains ONLY the motion block on a frozen finished run (config + checkpoint, EMA weights): wraps `camera_bevs` (block inputs: current camera BEV, an older window frame ego-aligned by `splat_window`, radar raster, age) and replaces `TimeMover.motion` by the block's field; the base's frozen loss + the coarse levels' motion terms; evaluates base and block on the same `val_subset` (velocity EPE on moving / static / vehicle cells, at box centres vs the object head, layer-0 warp check with GT v, loss groups, block ms) -> metrics.json | dry run on `t10_dual_s0` (4 steps, 6-30 val samples) |")
i = next(k for k, l in enumerate(s) if l.startswith("| `temporal_fusion.py`"))
s.insert(i, "| `motion_block.py` | velocity field at time k from two camera BEVs (current + an older frame ego-aligned to k) + the fixed radar raster (count, linear Doppler, ray, age) in a small pyramid (2 / 1 / 0.5 m): backward warp with the coarser v, Doppler residual u . v - vr, ray / range channels, a zero-init update per level; returns every level (deep supervision); the fused flow model of plan 7.4 without correlation, on model features | warp sign with a known v, radar raster means and pooling, zero start, gradient to all 42 tensors after one step, learns a blob velocity (0.53 m/s after 150 steps); 294 k parameters at 64 input channels, 120 x 104 |")
open(p, "w", encoding="utf-8").write("\n".join(s))
