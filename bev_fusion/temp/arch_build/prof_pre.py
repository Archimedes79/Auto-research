"""one-off: where does the time of the input pipeline go? mini split, CPU, no GT, 12 items; per stage the median ms per item."""
import os
import statistics as st
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.join(ROOT, ".."))
torch.set_num_threads(2)
from lift_pd import Grid  # noqa: E402
from nuscenes_dataset import FrontRadarDataset, collate  # noqa: E402
from precompute import Precompute, radar_hits, radar_static  # noqa: E402
from camera_encoder import depth_centres  # noqa: E402


def timed(fn, n=1):
    t = time.perf_counter()
    for _ in range(n):
        out = fn()
    return (time.perf_counter() - t) / n * 1e3, out


grid = Grid()
t0 = time.perf_counter()
ds = FrontRadarDataset(os.path.expanduser("~/data/nuscenes"), "v1.0-mini", grid=grid, n_cam=3, cam_stride=2, with_gt=False)
print(f"dataset open {time.perf_counter() - t0:.1f} s ({len(ds)} samples)")
idx = list(range(10, 22))
rows = {k: [] for k in ("item (all)", "  cam frames (3 images + calibration)", "  one image decode+resize", "  radar (5 radars, 0.5 s)", "  gap steps + radar poses",
                        "collate", "precompute, first call", "precompute, cached call", "  radar_hits", "  radar_static")}
pre = Precompute(grid, grid.z_levels, depth_centres(), oversample=1)
for k, i in enumerate(idx):
    tok = ds.samples[i]
    sample = ds.nusc.get("sample", tok)
    ms, item = timed(lambda: ds[i])
    rows["item (all)"].append(ms)
    rows["  cam frames (3 images + calibration)"].append(timed(lambda: ds._cam_frames(tok))[0])
    sd = ds.gt.cam(tok)
    rows["  one image decode+resize"].append(timed(lambda: ds._image(sd))[0])
    rows["  radar (5 radars, 0.5 s)"].append(timed(lambda: ds._radar(sample, ds.gt.cam(tok)))[0])
    rows["  gap steps + radar poses"].append(timed(lambda: ds._radar_poses(sample))[0])
    ms, batch = timed(lambda: collate([item]))
    rows["collate"].append(ms)
    ms, _ = timed(lambda: pre(dict(batch)))
    rows["precompute, first call" if k == 0 else "precompute, cached call"].append(ms)
    if k:
        pass
    rows["  radar_hits"].append(timed(lambda: radar_hits(batch["radar"], 1, grid), 5)[0])
    rows["  radar_static"].append(timed(lambda: radar_static(batch["radar"], 1, grid), 5)[0])
print(f"{'stage':44s} {'median ms':>10s}  n")
for k, v in rows.items():
    if v:
        print(f"{k:44s} {st.median(v):10.2f}  {len(v)}")
