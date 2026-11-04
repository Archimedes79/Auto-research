"""one-off: cProfile of FrontRadarDataset._radar (warm file cache), mini split, 12 samples; prints the top entries by cumulative and own time"""
import cProfile
import io
import os
import pstats
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.join(ROOT, ".."))
torch.set_num_threads(2)
from lift_pd import Grid  # noqa: E402
from nuscenes_dataset import FrontRadarDataset  # noqa: E402

ds = FrontRadarDataset(os.path.expanduser("~/data/nuscenes"), "v1.0-mini", grid=Grid(), n_cam=3, cam_stride=2, with_gt=False)
items = [(ds.nusc.get("sample", ds.samples[i]), ds.gt.cam(ds.samples[i])) for i in range(10, 22)]
for s, c in items:                                   # warm the file cache
    ds._radar(s, c)
t = time.perf_counter()
for s, c in items:
    ds._radar(s, c)
print(f"warm _radar: {(time.perf_counter() - t) / len(items) * 1e3:.1f} ms per sample, "
      f"{sum(len(ds._radar(s, c)['x']) for s, c in items) / len(items):.0f} detections")
pr = cProfile.Profile()
pr.enable()
for s, c in items:
    ds._radar(s, c)
pr.disable()
for key in ("cumulative", "tottime"):
    out = io.StringIO()
    pstats.Stats(pr, stream=out).sort_stats(key).print_stats(14)
    print("\n".join(l[:150] for l in out.getvalue().splitlines() if l.strip())[:3200])
    print("-----")
