"""The precompute step (layer 1 of training/architecture.md): what the network needs that no learned weight produces, made once per
frame and read by the network, the training and the evaluation alike. Today it adds the visibility maps to a batch; calibration, ego
steps, compensated radar points and the frozen image features come from the dataset and the feature caches.

    pre = Precompute.from_model(model)      # grid, heights and depth range of the model's camera slot (or .from_config(ModelConfig))
    batch = pre(batch [, keys])             # adds the maps below (default all), (B, 1, X, Y) in [0, 1], each in the batch's frame
                                            # (mirrored for flipped samples); a key already present is kept

On its own (no network): `python runner/bev_runner.py check | precompute <config.json> ...` (the runner builds the network input of chosen
frames through this step with the dataset; the data loaders are not part of this file, which belongs to the network's bundle).

The network asks only for what its memories read (no extra work on the training and timing paths); the training (D36 loss masks)
and the evaluation (D36 / D37 scoring masks) call the step themselves.

vis_cam    the cells the current camera frame observes (lift_pd.camera_visibility: any height sample of the cell projects into the image
           within the depth range), cached per calibration, so a fixed rig costs one computation per process - the same function runs
           live in a car. The write mask of the camera memory (the network mirrors it with the camera BEV for flipped samples); D36: the
           loss and scoring mask of the camera outputs.
vis_radar  the cells the radar rig can see (D36, Mirko: "obs_radar stays the fan"): the +-60 deg / 70 m fans of the five radars, from their
           calibrated poses (batch["radar_poses"]) through autogt.coverage.radar_cells - the one definition the ground truth uses -, made
           once per rig. The fans cover 99.6 % of this grid (bev_fusion/records/sensor_coverage_2026-11-01.md), so the map restricts little
           today; what matters is that it comes from the geometry and not from a constant.
vis_fusion the fusion area (D37: fusion results are reported on cells observed by camera and radar) = vis_cam x vis_radar; the evaluation
           removes the cells hidden behind a GT box (autogt.coverage.hidden_by_boxes) on top.
radar_hits the cells with at least one radar detection of this frame's compensated sweeps, widened by `dilate` cells (radar points are
           mirrored by the augmentation already): where the radar measured now, the write mask of the radar memory.
radar_static the same cells, each weighted by how static its detections are by the measured Doppler (mean of exp(-(vr / v_static)^2),
           vr ego-compensated, so a return of the static world has vr ~ 0): the write mask of the radar memory with ModelConfig.
           rad_static_gate - only the static world is remembered, a moving car (not in the same place next frame) is not.
Run the self-test: python training/precompute.py
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .lift_pd import camera_visibility
except ImportError:
    from lift_pd import camera_visibility

_FANS = {}                                                    # (grid, the five radar poses) -> the fan (nx, ny), made once per process


def _radar_cells():
    """autogt.coverage.radar_cells (the fan the ground truth uses), imported when first needed: the folder autogt/ is found by walking
    up from this file (the repository, or a run's snapshot that carries its own copy next to code/)"""
    root = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(root, "autogt")) and os.path.dirname(root) != root:
        root = os.path.dirname(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from autogt import coverage
    return coverage.radar_cells


MAPS = ("vis_cam", "vis_radar", "vis_fusion", "radar_hits", "radar_static")


def radar_static(radar, B, grid, v_static=1.0, dilate=1):
    """(B, 1, X, Y): per cell the mean static weight exp(-(vr / v_static)^2) of its detections (vr = the ego-compensated Doppler, about 0
    for a return of the static world), 0 where there is no detection, widened by `dilate` cells like radar_hits (max)"""
    n = B * grid.nx * grid.ny
    dev = radar["x"].device if len(radar.get("x", [])) else "cpu"
    wsum, cnt = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
    if len(radar.get("x", [])):
        ix = ((radar["x"] - grid.x_min) / grid.res).floor().long()
        iy = ((radar["y"] - grid.y_min) / grid.res).floor().long()
        inside = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
        idx = (radar["b"][inside] * grid.nx + ix[inside]) * grid.ny + iy[inside]
        vr = torch.nan_to_num(radar["vr"][inside].float(), nan=float("inf"))          # a Doppler that was not measured is not static
        w = torch.exp(-(vr / v_static) ** 2)
        wsum.index_add_(0, idx, w)
        cnt.index_add_(0, idx, torch.ones_like(w))
    m = (wsum / cnt.clamp(min=1)).view(B, 1, grid.nx, grid.ny)
    return F.max_pool2d(m, 2 * dilate + 1, 1, dilate) if dilate > 0 else m


def radar_hits(radar, B, grid, dilate=1):
    """(B, 1, X, Y): 1 where a cell (or a neighbour within `dilate` cells) holds a radar detection"""
    dev = radar["x"].device if len(radar.get("x", [])) else "cpu"
    occ = torch.zeros(B * grid.nx * grid.ny, device=dev)
    if len(radar.get("x", [])):
        ix = ((radar["x"] - grid.x_min) / grid.res).floor().long()
        iy = ((radar["y"] - grid.y_min) / grid.res).floor().long()
        inside = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
        occ[(radar["b"][inside] * grid.nx + ix[inside]) * grid.ny + iy[inside]] = 1.0
    occ = occ.view(B, 1, grid.nx, grid.ny)
    return F.max_pool2d(occ, 2 * dilate + 1, 1, dilate) if dilate > 0 else occ


class Precompute:
    def __init__(self, grid, z_levels, depth_centres, oversample=1, radar_dilate=1, radar_static_vr=1.0):
        self.grid, self.z_levels, self.depth_centres = grid, z_levels, depth_centres
        self.oversample, self.radar_dilate, self.radar_static_vr = oversample, radar_dilate, radar_static_vr

    @classmethod
    def from_model(cls, model):
        return cls(model.grid, model.cfg.z_levels, model.lift.depth_centres, model.cfg.oversample, radar_static_vr=model.cfg.rad_static_vr)

    @classmethod
    def from_config(cls, mcfg):
        """the same step from a ModelConfig alone (no network built): the grid, heights and depth range of its camera slot"""
        try:
            from .camera_encoder import depth_centres
            from .lift_pd import Grid
        except ImportError:
            from camera_encoder import depth_centres
            from lift_pd import Grid
        if mcfg.camera_features == "bevfusion":
            try:
                from .bevfusion_camera import DEPTHS
            except ImportError:
                from bevfusion_camera import DEPTHS
            dc = DEPTHS
        else:
            dc = depth_centres()
        return cls(Grid(z_levels=mcfg.z_levels, **mcfg.grid_kwargs), mcfg.z_levels, torch.as_tensor(dc, dtype=torch.float32), mcfg.oversample,
                   radar_static_vr=mcfg.rad_static_vr)

    def camera(self, batch):
        K, T = batch["K"], batch["T_cam_car"]
        hw = batch["hw"] if "hw" in batch else batch["img"].shape[-2:]
        vis = [camera_visibility(self.grid, self.z_levels, self.depth_centres, K[b, 0], T[b, 0], hw, batch.get("ida"), self.oversample)
               for b in range(K.shape[0])]
        return torch.stack(vis)[:, None].to(device=K.device, dtype=torch.float32)

    def radar_fan(self, batch):
        """(B, 1, X, Y): the cells inside the fan of one of the five radars, from batch["radar_poses"] (autogt.coverage.radar_cells on the
        cell centres: the definition of the ground truth); one computation per distinct rig"""
        if "radar_poses" not in batch:
            raise KeyError("vis_radar needs batch['radar_poses'] (B, 5, 4, 4), the radar calibration (nuscenes_dataset.collate makes it)")
        assert batch["radar_poses"].shape[0] == batch["K"].shape[0], (tuple(batch["radar_poses"].shape), tuple(batch["K"].shape))
        g, fans = self.grid, []
        for poses in batch["radar_poses"].detach().double().cpu().numpy():
            key = (g.x_min, g.y_min, g.res, g.nx, g.ny, poses.tobytes())
            if key not in _FANS:
                gx, gy = np.meshgrid(g.x_min + (np.arange(g.nx) + 0.5) * g.res, g.y_min + (np.arange(g.ny) + 0.5) * g.res, indexing="ij")
                _FANS[key] = torch.from_numpy(_radar_cells()(gx, gy, poses)).float()
            fans.append(_FANS[key])
        return torch.stack(fans)[:, None].to(batch["K"].device)

    def __call__(self, batch, keys=MAPS):
        """adds the requested maps (default all) to `batch` and returns it; a key already present is kept. Every map is in the
        batch's own frame: the rig maps (camera, fan) are mirrored for flipped samples, the radar hits come from points that the
        augmentation mirrored already. Without radar points there are no radar hits (key not added)."""
        B, dev = batch["K"].shape[0], batch["K"].device
        fl = batch.get("flip")
        mirror = (lambda m: torch.where(fl.to(dev).view(-1, 1, 1, 1), m.flip(-1), m)) if fl is not None and bool(fl.any()) else (lambda m: m)
        need = set(keys) | ({"vis_cam", "vis_radar"} if "vis_fusion" in keys else set())
        if "vis_cam" in need and "vis_cam" not in batch:
            batch["vis_cam"] = mirror(self.camera(batch))
        if "vis_radar" in need and "vis_radar" not in batch:
            batch["vis_radar"] = mirror(self.radar_fan(batch))
        if "vis_fusion" in need and "vis_fusion" not in batch:
            batch["vis_fusion"] = batch["vis_cam"] * batch["vis_radar"]
        if "radar_hits" in need and "radar_hits" not in batch and "radar" in batch:
            batch["radar_hits"] = radar_hits(batch["radar"], B, self.grid, self.radar_dilate).to(dev)
        if "radar_static" in need and "radar_static" not in batch and "radar" in batch:
            batch["radar_static"] = radar_static(batch["radar"], B, self.grid, self.radar_static_vr, self.radar_dilate).to(dev)
        return batch


def _selftest():
    try:
        from .camera_encoder import depth_centres
        from .lift_pd import Grid, _front_camera
    except ImportError:
        from camera_encoder import depth_centres
        from lift_pd import Grid, _front_camera
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
    assert torch.equal(out["vis_radar"][0, 0], torch.from_numpy(_radar_cells()(gx, gy, rig.numpy())).float()), "radar fan != autogt.coverage"
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
    st = out["radar_static"]                                      # the Doppler gate: a static return 1, a return at 3 m/s exp(-9), none 0
    assert st.shape == hits.shape and abs(float(st[0, 0, ix(10.2), iy(0.3)]) - 1.0) < 1e-6 and float(st[1, 0, ix(10.2), iy(0.3)]) == 0
    assert abs(float(st[1, 0, ix(30.0), iy(-5.0)]) - math.exp(-9)) < 1e-6 and float(st[0].sum()) == 9 and (st <= hits).all(), "static <= hits"
    mix = {"x": torch.tensor([10.2, 10.3]), "y": torch.tensor([0.3, 0.3]), "b": torch.tensor([0, 0]), "vr": torch.tensor([0.0, 3.0])}
    sm = radar_static(mix, B, grid, 1.0, 0)
    assert abs(float(sm[0, 0, ix(10.2), iy(0.3)]) - (1 + math.exp(-9)) / 2) < 1e-6, "the mean of the detections in a cell"
    assert abs(float(radar_static(mix, B, grid, 4.0, 0)[0, 0, ix(10.2), iy(0.3)]) - (math.exp(-0) + math.exp(-(3 / 4) ** 2)) / 2) < 1e-6
    nan = {"x": torch.tensor([10.2]), "y": torch.tensor([0.3]), "b": torch.tensor([0]), "vr": torch.tensor([float("nan")])}
    assert torch.isfinite(radar_static(nan, B, grid, 1.0, 1)).all() and radar_static(nan, B, grid, 1.0, 1).sum() == 0, "a NaN Doppler is not static"
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
    try:                                                          # the step without a network = the step of the network
        from .feature_cache import CachedBEVFusionModel
        from .model import ModelConfig
    except ImportError:
        from feature_cache import CachedBEVFusionModel
        from model import ModelConfig
    for cf in ("dualbev", "bevfusion"):
        mcfg = ModelConfig(camera_features=cf, pretrained="", trunk_pretrained="")
        a = Precompute.from_config(mcfg)(dict(batch), ("vis_cam",))["vis_cam"]
        b = Precompute.from_model(CachedBEVFusionModel(mcfg))(dict(batch), ("vis_cam",))["vis_cam"]
        assert torch.equal(a, b), f"from_config != from_model ({cf})"
    print("precompute selftest passed: camera map = lift_pd.camera_visibility, radar fan = autogt.coverage.radar_cells of the calibrated poses, fusion area = camera x radar, "
          "radar hits from the detections, existing keys kept, maps mirrored, from_config = from_model")


if __name__ == "__main__":
    _selftest()
