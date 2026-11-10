"""Spec test of ModelConfig.camera_features (feature swap: cached BEVFusion features at stride 8 instead of the DualBEV cache at stride 16).
(a) default "dualbev": the lift is exactly what it was (DualBEV depth centres, cell-centre convention) and the cached model output on a
    16 x 44 batch is the same as with the field set explicitly (default path untouched);
(b) "bevfusion": frustum cells, depth centres 1.0 + 0.5 i, 32 x 88 features give the same camera BEV shape; lift 's' is refused; the
    cached dataset carries bevfusion_ida (scale 0.48), the default dataset the DualBEV transform.
Run (from the bev_fusion folder): wsl ~/miniconda3/envs/gcar/bin/python tests/test_camera_features.py   (CPU is enough, ~20 s)"""
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

# (c) "dual" (t10): both cached feature sets, merged back to c_cam; branches use their own depth centres / frustum convention / ida
md = build(camera_features="dual")
assert md.lift.frustum_cells is False and md.lift2.frustum_cells is True
assert torch.equal(md.lift.depth_centres, depth_centres()) and float(md.lift2.depth_centres[0]) == 1.0
bd = {**batch(16, 44, ida0), "ida2": bevfusion_ida()}
b2 = batch(32, 88, bevfusion_ida())
bd["prob2"], bd["ctx2"] = b2["prob"], b2["ctx"]
with torch.no_grad():
    od = md.camera_bevs(bd)
assert od.shape == o0.shape and torch.isfinite(od).all() and float(od.abs().sum()) > 0, od.shape
# merge [I, 0] / [0, I] reproduces each single-branch camera BEV (eval-mode BN set to identity, ReLU on non-negative maps)
c = md.cfg.c_cam
conv, bn = md.cam_merge[0], md.cam_merge[1]
bn.weight.data.fill_(1.0), bn.bias.data.zero_(), bn.running_mean.zero_(), bn.running_var.fill_(1.0 - bn.eps)
with torch.no_grad():
    bev1 = torch.stack([md.cam_reduce(md.lift(bd["ctx"].float()[b], bd["prob"].float()[b], K, T, (256, 704), ida0)) for b in range(2)])
    bev2 = torch.stack([md.cam_reduce2(md.lift2(bd["ctx2"].float()[b], bd["prob2"].float()[b], K, T, (256, 704), bd["ida2"])) for b in range(2)])
    for sel, ref in ((0, bev1), (1, bev2)):
        w = torch.zeros(c, 2 * c, 1, 1)
        w[torch.arange(c), torch.arange(c) + sel * c] = 1.0
        conv.weight.data.copy_(w)
        got = md.camera_bevs(bd)
        assert torch.allclose(got, torch.relu(ref), atol=1e-5), ("branch select", sel, float((got - torch.relu(ref)).abs().max()))
# gradients reach both reductions and the merge (random merge: the selector above kills one branch on purpose)
conv.weight.data.normal_(0, 0.3)
md.train()
md.camera_bevs(bd).square().mean().backward()
for nm in ("cam_reduce", "cam_reduce2", "cam_merge"):
    g = sum(float(p.grad.abs().sum()) for p in getattr(md, nm).parameters() if p.grad is not None)
    assert g > 0, f"no gradient in {nm}"
# refusals and the dataset pairing
for kw in (dict(camera_features="dual", lift="s"), dict(camera_features="dual", cam_adapter=True)):
    try:
        build(**kw)
        raise AssertionError(f"{kw} must be refused")
    except ValueError:
        pass
try:
    FrontRadarDataset.__init__ = lambda self, *a, **k: setattr(self, "ida", ida0)
    CachedFrontRadarDataset(None, None, store=None, camera_features="dual")
    raise AssertionError("dual without store2 must be refused")
except ValueError:
    pass
finally:
    FrontRadarDataset.__init__ = orig
print("test_camera_features dual ok: shapes, branch select [I,0] / [0,I], gradients to both reductions and the merge, refusals")

# (d) the camera slot table (model.CAMERAS): the one place that names the feature sources of every kind. The dataset, collate_cached, the cached
# model and train.build read it; the checks below pin what a checkpoint and a batch depend on (names, order), not the kinds' strings.
import dataclasses  # noqa: E402

import feature_cache  # noqa: E402
from interface import INPUTS  # noqa: E402
from model import CAMERAS  # noqa: E402
import train as _train  # noqa: E402

assert tuple(CAMERAS) == ("dualbev", "bevfusion", "dual"), tuple(CAMERAS)
tcfg = {f.name for f in dataclasses.fields(_train.TrainConfig)}
for kind, srcs in CAMERAS.items():
    assert srcs[0].suffix == "" and all(s.suffix for s in srcs[1:]) and len({s.suffix for s in srcs}) == len(srcs), (kind, srcs)
    for s in srcs:                                              # every batch field a source reads is in the input contract
        assert s.cache in tcfg, (kind, s.cache)
        assert "prob" + s.suffix in INPUTS and "ctx" + s.suffix in INPUTS, (kind, s)
        assert s.suffix == "" or "ida" + s.suffix in INPUTS, (kind, s)
assert [(s.features, s.cache, s.suffix) for s in CAMERAS["dual"]] == [("dualbev", "feature_cache", ""), ("bevfusion", "feature_cache2", "2")]
assert len(CAMERAS["dualbev"]) == len(CAMERAS["bevfusion"]) == 1
# what an existing checkpoint depends on: key names of the second source and the merge, and the order the parameters are registered in
keys = set(md.state_dict())
for k in ("lift2.depth_centres", "lift2.z_levels", "cam_reduce2.0.weight", "cam_reduce2.1.weight", "cam_merge.0.weight", "cam_merge.1.running_mean"):
    assert k in keys, f"checkpoint key {k} missing from the dual model"
assert "lift3.depth_centres" not in keys and "cam_merge.0.weight" not in set(m0.state_dict()), "single-source kinds must not carry the merge"
first = {}
for i, (n, _) in enumerate(md.named_parameters()):
    first.setdefault(n.split(".")[0], i)
assert first["cam_reduce"] < first["cam_reduce2"] < first["cam_merge"] < first["fusion"], first      # optimizer states are positional

# the dataset and the collate follow the table (fake stores, fake base item: no data is read)
class _Store:
    def __init__(self, n_prob, hw):
        self.n_prob, self.hw, self.root = n_prob, hw, "fake_cache"

    def feature_hw(self):
        return self.hw

    def get(self, token):
        return torch.rand(self.n_prob, *self.hw), torch.rand(80, *self.hw)


try:
    orig_init, orig_get = FrontRadarDataset.__init__, FrontRadarDataset.__getitem__
    orig_collate = feature_cache.collate
    FrontRadarDataset.__init__ = lambda self, *a, **k: setattr(self, "ida", ida0)
    FrontRadarDataset.__getitem__ = lambda self, i: {"token": "t", "frames": [{"token": "a"}, {"token": "b"}], "ida": self.ida}
    feature_cache.collate = lambda items: {"ida": items[0]["ida"]}
    d1 = CachedFrontRadarDataset(None, None, store=_Store(118, (16, 44)), camera_features="dualbev")
    d2 = CachedFrontRadarDataset(None, None, store=_Store(118, (16, 44)), store2=_Store(118, (32, 88)), camera_features="dual")
    db = CachedFrontRadarDataset(None, None, store=_Store(118, (32, 88)), camera_features="bevfusion")
    assert [s.suffix for s in d2.sources] == ["", "2"] and d2.stores[1] is not None
    assert torch.equal(d2.ida, ida0) and torch.equal(d2.ida2, bevfusion_ida()) and torch.equal(db.ida, bevfusion_ida()) and torch.equal(d1.ida, ida0)
    it1, it2, itb = d1[0], d2[0], db[0]
    assert "ida2" not in it1 and "prob2" not in it1["frames"][0] and "ida2" not in itb
    assert torch.equal(it2["ida2"], bevfusion_ida()) and torch.equal(it2["ida"], ida0)
    assert it2["frames"][0]["prob2"].shape == (118, 32, 88) and it2["frames"][0]["ctx2"].shape == (80, 32, 88) and it2["frames"][1]["prob"].shape == (118, 16, 44)
    b1, b2 = feature_cache.collate_cached([it1, it1]), feature_cache.collate_cached([it2, it2])
    assert b1["prob"].shape == (2, 2, 118, 16, 44) and b1["ctx"].shape == (2, 2, 80, 16, 44) and not any(k.endswith("2") for k in b1), sorted(b1)
    assert b2["prob2"].shape == (2, 2, 118, 32, 88) and b2["ctx2"].shape == (2, 2, 80, 32, 88) and torch.equal(b2["ida2"], bevfusion_ida())
    assert torch.equal(b2["ida"], ida0) and b2["hw"] == feature_cache.DST_HW
    for bad in (dict(store=_Store(118, (16, 44)), store2=_Store(118, (32, 88)), camera_features="dualbev"),     # store2 for a one-source kind
                dict(store=_Store(118, (16, 44)), camera_features="dual"),                                       # a second source without store2
                dict(store=_Store(118, (32, 88)), camera_features="dualbev"),                                    # caches swapped: feature size
                dict(store=_Store(118, (16, 44)), store2=_Store(118, (16, 44)), camera_features="dual"),
                dict(store=_Store(118, (16, 44)), camera_features="nonsense")):
        try:
            CachedFrontRadarDataset(None, None, **bad)
            raise AssertionError(f"{sorted(bad)} {bad['camera_features']} must be refused")
        except ValueError:
            pass
finally:
    FrontRadarDataset.__init__, FrontRadarDataset.__getitem__, feature_cache.collate = orig_init, orig_get, orig_collate
print("test_camera_features table ok: CAMERAS names, checkpoint keys and registration order, dataset items and collate per source, refusals")
