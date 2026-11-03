"""one-off: the vectorised radar reader against the devkit-based _radar of HEAD: identical outputs, and the time (cold and warm file cache)"""
import importlib.util
import os
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.join(ROOT, ".."))
torch.set_num_threads(2)
old_src = subprocess.run(["git", "-C", ROOT, "show", "HEAD:bev_fusion/training/nuscenes_dataset.py"], capture_output=True, text=True, check=True).stdout
old_path = os.path.join(ROOT, "temp", "arch_build", "nuscenes_dataset_head.py")
open(old_path, "w", encoding="utf-8").write(old_src)
import nuscenes_dataset as NEW  # noqa: E402
spec = importlib.util.spec_from_file_location("nuscenes_dataset_head", old_path)
OLD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OLD)
from lift_pd import Grid  # noqa: E402

root = os.path.expanduser("~/data/nuscenes")
new = NEW.FrontRadarDataset(root, "v1.0-mini", grid=Grid(), n_cam=3, cam_stride=2, with_gt=False)
old = OLD.FrontRadarDataset(root, "v1.0-mini", grid=Grid(), n_cam=3, cam_stride=2, with_gt=False)
idx = list(range(10, 22)) + [100, 200, 300]
n_det = 0
for label in ("cold-ish", "warm"):
    t_old = t_new = 0.0
    for i in idx:
        tok = new.samples[i]
        s, c = new.nusc.get("sample", tok), new.gt.cam(tok)
        t = time.perf_counter(); a = old._radar(s, c); t_old += time.perf_counter() - t
        t = time.perf_counter(); b = new._radar(s, c); t_new += time.perf_counter() - t
        assert a.keys() == b.keys()
        for k in a:
            assert torch.equal(a[k], b[k]), (i, k, a[k].shape, b[k].shape)
        n_det += len(a["x"])
    print(f"{label}: old {t_old / len(idx) * 1e3:.1f} ms, new {t_new / len(idx) * 1e3:.1f} ms per sample ({len(idx)} samples)")
print(f"identical outputs (torch.equal on all 8 fields), {n_det // 2} detections compared per pass")
os.remove(old_path)
