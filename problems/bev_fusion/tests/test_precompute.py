"""Tests of net/prep/precompute.py (its former self-test).

    python -P -m bev_fusion.tests.test_precompute      (from the problem folder, with its module path)
"""

import numpy as np
import torch

from bev_fusion.net.prep.geometry import camera_visibility
from bev_fusion.net.prep.precompute import MAPS, Precompute, _FANS, _radar_cells


def _selftest():
    from bev_fusion.net.prep.geometry import depth_centres
    from bev_fusion.net.model.lift_pd import Grid, _front_camera
    grid = Grid()
    K, T, hw = _front_camera(torch.float32)
    B = 2

    def pose(x, y, yaw):                                          # a radar's sensor -> vehicle matrix (yaw about z)
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        return torch.tensor([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0.5], [0, 0, 0, 1]])
    rig = torch.stack([pose(3.41, 0.0, 0.0), pose(2.42, 0.8, 1.04), pose(2.42, -0.8, -1.04), pose(-1.0, 0.9, 2.1), pose(-1.0, -0.9, -2.1)])
    batch = {"K": K.expand(B, 3, 3, 3).clone(), "T_cam_car": T.expand(B, 3, 4, 4).clone(), "hw": hw, "ida": None,
             "radar_poses": rig.expand(B, 5, 4, 4).clone(),
             "radar": {"x": torch.tensor([10.2, 30.0, 500.0]), "y": torch.tensor([0.3, -5.0, 0.0]), "b": torch.tensor([0, 1, 1]),
                       "vr": torch.tensor([0.0, 3.0, 0.0])}}
    pre = Precompute(grid, grid.z_levels, depth_centres(), oversample=1)
    out = pre(dict(batch))
    ref = camera_visibility(grid, grid.z_levels, depth_centres(), K, T, hw, None, 1)
    assert out["vis_cam"].shape == (B, 1, grid.nx, grid.ny) and torch.equal(out["vis_cam"][0, 0], ref.float()), "camera map != lift_pd"
    assert 0.2 < ref.float().mean() < 0.9, ref.float().mean()
    gx, gy = np.meshgrid(grid.x_min + (np.arange(grid.nx) + 0.5) * grid.res, grid.y_min + (np.arange(grid.ny) + 0.5) * grid.res, indexing="ij")
    assert torch.equal(out["vis_radar"][0, 0], torch.from_numpy(_radar_cells()(gx, gy, rig.numpy())).float()), "radar fan != the radar_cells of GtGenerator"
    assert 0.9 < out["vis_radar"].mean() <= 1.0, out["vis_radar"].mean()              # the five fans cover nearly the whole front grid
    assert torch.equal(out["vis_fusion"], out["vis_cam"] * out["vis_radar"]), "fusion area = camera x radar"
    one = pre(dict(batch, radar_poses=rig[:1].expand(B, 1, 4, 4).clone().repeat(1, 5, 1, 1)), ("vis_radar",))["vis_radar"][0, 0]
    ix = lambda x: int((x - grid.x_min) / grid.res)                                    # noqa: E731
    iy = lambda y: int((y - grid.y_min) / grid.res)                                    # noqa: E731
    assert one[ix(20.25), iy(0.25)] == 1 and one[ix(10.25), iy(25.25)] == 0 and one[ix(2.25), iy(0.25)] == 0 and one.mean() < 0.9, \
        "the fan is a geometry: 20 m ahead in, 25 m to the side at 10 m (68 deg) out, behind the sensor (x = 3.41 m) out"
    n_fans = len(_FANS)
    pre(dict(batch), ("vis_radar",))
    assert len(_FANS) == n_fans, "one computation per distinct rig, kept for later calls"
    asym = pre(dict(batch, radar_poses=rig[:1].expand(B, 1, 4, 4).clone().repeat(1, 5, 1, 1), flip=torch.tensor([True, False])), ("vis_radar",))["vis_radar"]
    assert torch.equal(asym[0, 0], one.flip(-1)) and torch.equal(asym[1, 0], one), \
        "an asymmetric rig: the fan of a flipped sample is mirrored, once (the symmetric real rig cannot show this)"
    try:
        pre(dict(batch, radar_poses=rig.expand(1, 5, 4, 4).clone()), ("vis_radar",))
        raise AssertionError("poses for one sample with a batch of two must be refused")
    except AssertionError as e:
        assert "must be refused" not in str(e), e
    try:
        pre({k: v for k, v in batch.items() if k != "radar_poses"}, ("vis_radar",))
        raise AssertionError("vis_radar without radar poses must not fall back to a constant")
    except KeyError:
        pass
    hits = out["radar_hits"]
    assert hits.shape == (B, 1, grid.nx, grid.ny) and hits[0].sum() == 9 and hits[1].sum() == 9, (hits[0].sum(), hits[1].sum())   # 3 x 3 each
    assert hits[0, 0, ix(10.2), iy(0.3)] == 1 and hits[1, 0, ix(10.2), iy(0.3)] == 0
    assert torch.equal(pre.radar_mask("radar_hits", batch["radar"], B), hits), "radar_mask = the batch map"
    try:
        pre.radar_mask("radar_hit", batch["radar"], B)
        raise AssertionError("an unknown mask name must be refused")
    except ValueError:
        pass
    kept = pre(dict(batch, vis_cam=torch.zeros(B, 1, grid.nx, grid.ny)))
    assert kept["vis_cam"].sum() == 0 and kept["vis_fusion"].sum() == 0, "an existing map must be kept and used"
    empty = pre(dict(batch, radar={"x": torch.zeros(0), "y": torch.zeros(0), "b": torch.zeros(0, dtype=torch.long)}))
    assert empty["radar_hits"].sum() == 0
    no_radar = {k: v for k, v in batch.items() if k != "radar"}
    assert "radar_hits" not in pre(dict(no_radar)), "no radar points: no radar hits, no error"
    fl = pre(dict(batch, flip=torch.tensor([True, False])))       # every map in the batch's frame: the camera map mirrored for sample 0
    assert torch.equal(fl["vis_cam"][0], out["vis_cam"][0].flip(-1)) and torch.equal(fl["vis_cam"][1], out["vis_cam"][1])
    only = pre(dict(batch), keys=("vis_cam",))
    assert set(only) & set(MAPS) == {"vis_cam"}, "only the requested maps are made"
    from bev_fusion.net.model.cached_model import CachedBEVFusionModel
    from bev_fusion.net.model.model import ModelConfig
    for cf in ("dualbev", "bevfusion"):
        mcfg = ModelConfig(camera_features=cf, pretrained="", trunk_pretrained="")
        a = Precompute.from_config(mcfg)(dict(batch), ("vis_cam",))["vis_cam"]
        b = Precompute.from_model(CachedBEVFusionModel(mcfg))(dict(batch), ("vis_cam",))["vis_cam"]
        assert torch.equal(a, b), f"from_config != from_model ({cf})"
    print("precompute selftest passed: camera map = lift_pd.camera_visibility, radar fan = the radar_cells of GtGenerator on the calibrated poses, fusion area = camera x radar, "
          "radar hits from the detections, existing keys kept, maps mirrored, from_config = from_model")


if __name__ == "__main__":
    _selftest()
