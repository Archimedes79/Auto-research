"""t21 scratch: the standard (non-cached) chain with the real image encoder, nuScenes mini: cam_buffer 'window' (the t20 path), 'pair' (n_cam 2)
and 'recurrent' run, are finite, and the second key frame continues from the first one's buffer. Random weights, CPU, one process at a time (RAM)."""
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))
os.chdir(ROOT)
from model import BEVFusionModel, ModelConfig  # noqa: E402
from nuscenes_dataset import FrontRadarDataset, collate  # noqa: E402

torch.set_num_threads(2)
torch.manual_seed(0)
kind, n_cam = sys.argv[1], int(sys.argv[2])
ds = FrontRadarDataset(os.path.expanduser("~/data/nuscenes"), "v1.0-mini", n_cam=n_cam, cam_stride=2, with_gt=False)
b1, b2 = collate([ds[10]]), collate([ds[11]])
m = BEVFusionModel(ModelConfig(pretrained="", trunk_pretrained="", grid_kwargs={"x_max": 16.0, "y_min": -8.0, "y_max": 8.0}, c_cam=16, c_rad=8, c_fuse=24,
                               trunk_width=0.25, cam_buffer=kind, motion_block=True)).eval()
with torch.no_grad():
    r1 = m(b1, dts=(0.0, 0.5))
    step = (torch.tensor([0.5]), torch.tensor([float(b1["v"][0])]), torch.tensor([float(b1["omega"][0])]))
    r2 = m(b2, dts=(0.0, 0.5), state=r1["state"], step=step)
ok = all(torch.isfinite(t).all() for r in (r1, r2) for t in (r["out"][0.0]["sem"], r["out"][0.5]["sem"], r["out"][0.0]["v_hat"]))
print(kind, "n_cam", n_cam, "finite:", bool(ok), "state keys:", sorted(r2["state"]))
assert ok
