"""The precompute step (layer 1 of training/architecture.md): what the network needs that no learned weight produces, made once per
frame and read by the network, the training and the evaluation alike. Today it adds the visibility maps to a batch; calibration, ego
steps, compensated radar points and the frozen image features come from the dataset and the feature caches.

    pre = Precompute.from_model(model)      # grid, heights and depth range of the model's camera slot (or .from_config(ModelConfig))
    batch = pre(batch [, keys])             # adds the maps below (default all), (B, 1, X, Y) in [0, 1], each in the batch's frame
                                            # (mirrored for flipped samples); a key already present is kept

The data loaders are not part of this file, which belongs to the network's bundle.

The network asks only for what its memories read (no extra work on the training and timing paths); the training (D36 loss masks)
and the evaluation (D36 / D37 scoring masks) call the step themselves.

vis_cam    the cells the current camera frame observes (lift_pd.camera_visibility: any height sample of the cell projects into the image
           within the depth range), cached per calibration, so a fixed rig costs one computation per process - the same function runs
           live in a car. The write mask of the camera memory (the network mirrors it with the camera BEV for flipped samples); D36: the
           loss and scoring mask of the camera outputs.
vis_radar  the cells the radar rig can see (D36, the owner: "obs_radar stays the fan"): the +-60 deg / 70 m fans of the five radars, from their
           calibrated poses (batch["radar_poses"]) through GtGenerator.coverage.radar_cells - the one definition the ground truth uses -, made
           once per rig. The fans cover 99.6 % of this grid, so the map restricts little
           today; what matters is that it comes from the geometry and not from a constant.
vis_fusion the fusion area (D37: fusion results are reported on cells observed by camera and radar) = vis_cam x vis_radar; the evaluation
           removes the cells hidden behind a GT box (GtGenerator.coverage.hidden_by_boxes) on top.
radar_hits the cells with at least one radar detection of this frame's compensated sweeps, widened by `dilate` cells (radar points are
           mirrored by the augmentation already): where the radar measured now, the write mask of the radar memory.
           `pre.radar_mask(name, radar, B)` makes the map for ONE radar set: the batch's, or, with ModelConfig.radar_per_frame (t25), each
           camera frame's own sweep of batch["radar_seq"] (the model calls it per frame; the batch keeps no per-frame maps).
Tests: tests/test_precompute.py
"""

import numpy as np
import torch
import torch.nn.functional as F

from net.prep.geometry import camera_visibility

_FANS = {}                                                    # (grid, the five radar poses) -> the fan (nx, ny), made once per process


def _radar_cells():
    """GtGenerator.coverage.radar_cells (the fan the ground truth uses), imported when first needed"""
    from GtGenerator import coverage
    return coverage.radar_cells


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

    @classmethod
    def from_config(cls, mcfg):
        """the same step from a ModelConfig alone (no network built): the grid, heights and depth range of its camera slot"""
        from net.prep.geometry import depth_centres
        from net.prep.geometry import Grid
        if mcfg.camera_features == "bevfusion":
            from net.prep.geometry import DEPTHS
            dc = DEPTHS
        else:
            dc = depth_centres()
        return cls(Grid(z_levels=mcfg.z_levels, **mcfg.grid_kwargs), mcfg.z_levels, torch.as_tensor(dc, dtype=torch.float32), mcfg.oversample)

    def camera(self, batch):
        K, T = batch["K"], batch["T_cam_car"]
        hw = batch["hw"] if "hw" in batch else batch["img"].shape[-2:]
        vis = [camera_visibility(self.grid, self.z_levels, self.depth_centres, K[b, 0], T[b, 0], hw, batch.get("ida"), self.oversample) for b in range(K.shape[0])]
        return torch.stack(vis)[:, None].to(device=K.device, dtype=torch.float32)

    def radar_fan(self, batch):
        """(B, 1, X, Y): the cells inside the fan of one of the five radars, from batch["radar_poses"] (GtGenerator.coverage.radar_cells on the
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

    def radar_mask(self, name, radar, B):
        """(B, 1, X, Y): the write mask `name` ('radar_hits') of ONE radar set (the batch's, or one frame's own sweep of radar_seq, t25)"""
        if name != "radar_hits":
            raise ValueError(f"radar_mask: {name!r} is not 'radar_hits'")
        return radar_hits(radar, B, self.grid, self.radar_dilate)

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
            batch["radar_hits"] = self.radar_mask("radar_hits", batch["radar"], B).to(dev)
        return batch
