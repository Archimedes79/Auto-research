"""The precompute step (layer 1 of training/architecture.md): what the network needs that no learned weight produces, made once per
frame and read by the network, the training and the evaluation alike. Today it adds the visibility maps to a batch; calibration, ego
steps, compensated radar points and the frozen image features come from the dataset and the feature caches.

    pre = Precompute.from_model(model)      # grid, heights and depth range of the model's camera slot
    batch = pre(batch [, keys])             # adds the maps below (default all), (B, 1, X, Y) in [0, 1], each in the batch's frame
                                            # (mirrored for flipped samples); a key already present is kept

The network asks only for what its memories read (no extra work on the training and timing paths); the training (D36 loss masks)
and the evaluation (D36 / D37 scoring masks) call the step themselves.

vis_cam    the cells the current camera frame observes (lift_pd.camera_visibility: any height sample of the cell projects into the image
           within the depth range), cached per calibration, so a fixed rig costs one computation per process - the same function runs
           live in a car. The write mask of the camera memory (the network mirrors it with the camera BEV for flipped samples); D36: the
           loss and scoring mask of the camera outputs.
vis_radar  the cells the radar rig can see (D36, Mirko: "obs_radar stays the fan"): the fans of the five radars. PLACEHOLDER = every
           cell (the fans cover 99.6 % of this grid, bev_fusion/records/sensor_coverage_2026-11-01.md) until the GT Discussion chat's
           autogt.coverage.radar_cells(grid, sensor_poses, half, rmax) exists; then it is computed once per rig here, nowhere else.
vis_fusion the fusion area (D37: fusion results are reported on cells observed by camera and radar) = vis_cam x vis_radar; the evaluation
           removes the cells hidden behind a GT box (autogt.coverage.hidden_by_boxes) on top.
radar_hits the cells with at least one radar detection of this frame's compensated sweeps, widened by `dilate` cells (radar points are
           mirrored by the augmentation already): where the radar measured now, the write mask of the radar memory.
Run the self-test: python training/precompute.py
"""
import torch
import torch.nn.functional as F

try:
    from .lift_pd import camera_visibility
except ImportError:
    from lift_pd import camera_visibility


MAPS = ("vis_cam", "vis_radar", "vis_fusion", "radar_hits")


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
    def __init__(self, grid, z_levels, depth_centres, oversample=1, radar_dilate=1):
        self.grid, self.z_levels, self.depth_centres = grid, z_levels, depth_centres
        self.oversample, self.radar_dilate = oversample, radar_dilate

    @classmethod
    def from_model(cls, model):
        return cls(model.grid, model.cfg.z_levels, model.lift.depth_centres, model.cfg.oversample)

    def camera(self, batch):
        K, T = batch["K"], batch["T_cam_car"]
        hw = batch["hw"] if "hw" in batch else batch["img"].shape[-2:]
        vis = [camera_visibility(self.grid, self.z_levels, self.depth_centres, K[b, 0], T[b, 0], hw, batch.get("ida"), self.oversample)
               for b in range(K.shape[0])]
        return torch.stack(vis)[:, None].to(device=K.device, dtype=torch.float32)

    def radar_fan(self, batch):
        """placeholder until autogt.coverage.radar_cells: every cell (the five fans cover 99.6 % of the front grid)"""
        return batch["K"].new_ones(batch["K"].shape[0], 1, self.grid.nx, self.grid.ny, dtype=torch.float32)

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
    batch = {"K": K.expand(B, 3, 3, 3).clone(), "T_cam_car": T.expand(B, 3, 4, 4).clone(), "hw": hw, "ida": None,
             "radar": {"x": torch.tensor([10.2, 30.0, 500.0]), "y": torch.tensor([0.3, -5.0, 0.0]), "b": torch.tensor([0, 1, 1])}}
    pre = Precompute(grid, grid.z_levels, depth_centres(), oversample=1)
    out = pre(dict(batch))
    ref = camera_visibility(grid, grid.z_levels, depth_centres(), K, T, hw, None, 1)
    assert out["vis_cam"].shape == (B, 1, grid.nx, grid.ny) and torch.equal(out["vis_cam"][0, 0], ref.float()), "camera map != lift_pd"
    assert 0.2 < ref.float().mean() < 0.9, ref.float().mean()
    assert out["vis_radar"].min() == 1 and torch.equal(out["vis_fusion"], out["vis_cam"] * out["vis_radar"]), "fusion area = camera x radar"
    hits = out["radar_hits"]
    assert hits.shape == (B, 1, grid.nx, grid.ny) and hits[0].sum() == 9 and hits[1].sum() == 9, (hits[0].sum(), hits[1].sum())   # 3 x 3 each
    ix, iy = int((10.2 - grid.x_min) / grid.res), int((0.3 - grid.y_min) / grid.res)
    assert hits[0, 0, ix, iy] == 1 and hits[1, 0, ix, iy] == 0
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
    print("precompute selftest passed: camera map = lift_pd.camera_visibility, radar fan (placeholder), fusion area = camera x radar, "
          "radar hits from the detections, existing keys kept")


if __name__ == "__main__":
    _selftest()
