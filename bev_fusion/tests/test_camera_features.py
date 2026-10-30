"""Spec test of ModelConfig.camera_features (feature swap: cached BEVFusion features at stride 8 instead of the DualBEV cache at stride 16).
(a) default "dualbev": the lift is exactly what it was (DualBEV depth centres, cell-centre convention) and the cached model output on a
    16 x 44 batch is the same as with the field set explicitly (default path untouched);
(b) "bevfusion": frustum cells, depth centres 1.0 + 0.5 i, 32 x 88 features give the same camera BEV shape; lift 's' is refused; the
    cached dataset carries bevfusion_ida (scale 0.48), the default dataset the DualBEV transform.
Run: wsl bash tools/env/_py14.sh tests/test_camera_features.py   (CPU is enough, ~20 s)"""
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "training")]
from camera_encoder import depth_centres  # noqa: E402
from feature_cache import CachedBEVFusionModel, CachedFrontRadarDataset  # noqa: E402
from model import ModelConfig  # noqa: E402

torch.manual_seed(0)
K = torch.tensor([[1260.0, 0, 800.0], [0, 1260.0, 450.0], [0, 0, 1]])
T = torch.eye(4)
T[:3, :3] = torch.tensor([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]])                 # car -> camera (x forward = camera z)
T[:3, 3] = torch.tensor([0.0, 1.5, -1.5])


def batch(h, w, ida):
    prob = torch.softmax(torch.randn(2, 1, 118, h, w), dim=2)
    return {"prob": prob, "ctx": torch.randn(2, 1, 80, h, w), "K": K.expand(2, 1, 3, 3), "T_cam_car": T.expand(2, 1, 4, 4),
            "hw": (256, 704), "ida": ida}


def build(**kw):
    m = CachedBEVFusionModel(ModelConfig(pretrained="", **kw)).eval()
    return m


# (a) default untouched
m0, m1 = build(), build(camera_features="dualbev")
assert not m0.lift.frustum_cells and torch.equal(m0.lift.depth_centres, depth_centres())
from nuscenes_dataset import nuscenes_front_ida  # noqa: E402
ida0 = nuscenes_front_ida(dst_hw=(256, 704))
m1.load_state_dict(m0.state_dict())
b0 = batch(16, 44, ida0)
with torch.no_grad():
    o0, o1 = m0.camera_bevs(b0), m1.camera_bevs(b0)
assert torch.equal(o0, o1), "explicit camera_features='dualbev' changed the output"

# (b) bevfusion
from bevfusion_camera import bevfusion_ida  # noqa: E402
mb = build(camera_features="bevfusion")
assert mb.lift.frustum_cells and float(mb.lift.depth_centres[0]) == 1.0 and mb.lift.depth_centres.numel() == 118
assert abs(float(mb.lift.depth_centres[1] - mb.lift.depth_centres[0]) - 0.5) < 1e-6
bb = batch(32, 88, bevfusion_ida())
with torch.no_grad():
    ob = mb.camera_bevs(bb)
assert ob.shape == o0.shape == (2, 1, mb.cfg.c_cam, mb.grid.nx, mb.grid.ny), (ob.shape, o0.shape)
assert torch.isfinite(ob).all() and float(ob.abs().sum()) > 0
try:
    build(camera_features="bevfusion", lift="s")
    raise AssertionError("lift 's' with bevfusion features must be refused")
except ValueError:
    pass
try:
    build(camera_features="nonsense")
    raise AssertionError("unknown camera_features must be refused")
except ValueError:
    pass
# dataset: ida attribute (no data read: object built without __init__)
for feats, scale in (("bevfusion", 0.48), ("dualbev", float(ida0[0, 0]))):
    try:
        from nuscenes_dataset import FrontRadarDataset
        orig = FrontRadarDataset.__init__
        FrontRadarDataset.__init__ = lambda self, *a, **k: setattr(self, "ida", ida0)
        ds = CachedFrontRadarDataset(None, None, store=None, camera_features=feats)
    finally:
        FrontRadarDataset.__init__ = orig
    assert abs(float(ds.ida[0, 0]) - scale) < 1e-6, (feats, float(ds.ida[0, 0]))
    if feats == "bevfusion":
        assert float(ds.ida[0, 2]) == -32 and float(ds.ida[1, 2]) == -176
print("test_camera_features ok: default path identical, bevfusion lift (frustum cells, centres 1.0 + 0.5 i) and ida, refusals")
