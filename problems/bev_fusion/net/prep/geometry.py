"""Camera geometry in front of the first learned weight: the BEV grid, the projection table of a calibration, the visibility mask of a camera and the depth
bins. Plain math without a learned part; the lift (net/model/lift_pd.py), the precompute step and the datasets use it."""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Grid:
    """Voxel grid in vehicle coordinates (cell centres); defaults: front camera, 60 m, 0.5 m cells, 8 height slices."""
    x_min: float = 0.0
    x_max: float = 60.0
    y_min: float = -26.0
    y_max: float = 26.0
    res: float = 0.5
    z_levels: tuple = (-0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75)

    @property
    def nx(self):
        return int(round((self.x_max - self.x_min) / self.res))

    @property
    def ny(self):
        return int(round((self.y_max - self.y_min) / self.res))

    def xy_centres(self, device=None, dtype=torch.float32):
        xs = self.x_min + (torch.arange(self.nx, device=device, dtype=dtype) + 0.5) * self.res
        ys = self.y_min + (torch.arange(self.ny, device=device, dtype=dtype) + 0.5) * self.res
        return xs, ys

    def centres(self):
        """(gx, gy): cell-centre coordinates, both numpy (nx, ny): the interface of GtGenerator.grid.GridSpec (GT v13 reads grid.centres() on the
        grid it is given, and the BEV dataset gives it this one)"""
        xs, ys = self.xy_centres(dtype=torch.float64)
        return np.meshgrid(xs.numpy(), ys.numpy(), indexing="ij")


def build_table(grid, z_levels, K, T_cam_car, image_hw, depth_centres, ida=None):
    """Lookup table of lift PD for one camera, always in float32 (under bf16 autocast the matmuls would project with about 4 px
    error, and the table is cached and reused by later float32 calls). See _build_table."""
    with torch.autocast(z_levels.device.type, enabled=False):
        return _build_table(grid, z_levels, K, T_cam_car, image_hw, depth_centres, ida)


def _build_table(grid, z_levels, K, T_cam_car, image_hw, depth_centres, ida=None):
    """Lookup table of lift PD for one camera.

    z_levels: tensor (Z,) of heights (may require grad: learned elevations); K, T_cam_car, ida: tensors (3x3, 4x4, 3x3);
    image_hw: (H, W) of the augmented image the feature map covers; depth_centres: tensor (D,).
    Returns dict: grid2d (X*Y*Z, 2) normalised image coordinates, grid3d (X*Y*Z, 3) with the normalised depth-bin
    coordinate added, valid (X*Y*Z,) float mask; voxel order is x-major, then y, then z.
    """
    dev, dt = z_levels.device, z_levels.dtype
    xs, ys = grid.xy_centres(dev, dt)
    X, Y, Z = len(xs), len(ys), len(z_levels)
    gx = xs.view(X, 1, 1).expand(X, Y, Z)
    gy = ys.view(1, Y, 1).expand(X, Y, Z)
    gz = z_levels.view(1, 1, Z).expand(X, Y, Z)
    pts = torch.stack([gx, gy, gz, torch.ones_like(gx)], -1).reshape(-1, 4)          # (N, 4) vehicle frame
    pc = pts @ T_cam_car.to(dt).T                                                       # (N, 4) camera frame
    z = pc[:, 2]
    zs = z.clamp(min=1e-3)
    uv1 = torch.stack([pc[:, 0] / zs, pc[:, 1] / zs, torch.ones_like(zs)], -1) @ K.to(dt).T
    if ida is not None:
        uv1 = uv1 @ ida.to(dt).T
    H, W = image_hw
    xn = 2 * (uv1[:, 0] + 0.5) / W - 1
    yn = 2 * (uv1[:, 1] + 0.5) / H - 1
    dc = depth_centres.to(dev, dt)
    D = len(dc)
    # fractional bin index of the voxel's depth: piecewise linear between bin centres
    i1 = torch.searchsorted(dc, z.detach().contiguous()).clamp(1, D - 1)
    i0 = i1 - 1
    frac = i0.to(dt) + (z - dc[i0]) / (dc[i1] - dc[i0])
    dn = (2 * frac + 1) / D - 1                                                         # align_corners=False
    valid = (z > 1e-3) & (xn.abs() <= 1) & (yn.abs() <= 1) & (z >= dc[0]) & (z <= dc[-1])
    return {"grid2d": torch.stack([xn, yn], -1), "grid3d": torch.stack([xn, yn, dn], -1),
            "valid": valid.to(dt), "shape": (X, Y, Z)}


def lift_key(K, T_cam_car, image_hw, ida=None):
    """The projection-table cache key: what decides whether the cached table (which image pixel and depth bin every BEV cell and height maps to) can be reused -
    the camera intrinsics K, the camera-to-car extrinsics T_cam_car, the image size and the crop / resize transform ida. Python numbers only. """
    return (tuple(K.flatten().tolist()), tuple(T_cam_car.flatten().tolist()), tuple(image_hw),
            None if ida is None else tuple(ida.flatten().tolist()))


_VIS_CACHE = {}


def camera_visibility(grid, z_levels, depth_centres, K, T_cam_car, image_hw, ida=None, oversample=1, slice_height=0.5):
    """(X, Y) float mask of the cells a camera observes: any of the cell's height samples projects into the image within
    the depth range (the `valid` of build_table, any over height). The obs_mass of the temporal memory - outside it the
    camera BEV is empty, and writing that with full weight erased what the memory had seen before (temporal_fusion
    write, found 2026-10-26). Independent of the lift type; cached per calibration (nuScenes has few)."""
    z = torch.as_tensor(z_levels, dtype=torch.float32, device=depth_centres.device)
    if oversample > 1:
        off = ((torch.arange(oversample, dtype=z.dtype, device=z.device) + 0.5) / oversample - 0.5) * slice_height
        z = (z.view(-1, 1) + off.view(1, -1)).reshape(-1)
    key = (grid.x_min, grid.x_max, grid.y_min, grid.y_max, grid.res, tuple(z.tolist()), str(z.device),
           len(depth_centres), float(depth_centres[0]), float(depth_centres[-1]),       # the depth range decides the mask too
           tuple(K.flatten().tolist()), tuple(T_cam_car.flatten().tolist()), tuple(int(v) for v in image_hw),
           None if ida is None else tuple(ida.flatten().tolist()))
    if key not in _VIS_CACHE:
        if len(_VIS_CACHE) > 64:
            _VIS_CACHE.clear()
        tab = build_table(grid, z, K.to(z.device), T_cam_car.to(z.device), image_hw, depth_centres, None if ida is None
                          else ida.to(z.device))
        _VIS_CACHE[key] = tab["valid"].view(grid.nx, grid.ny, -1).amax(-1).detach()
    return _VIS_CACHE[key]


DEPTH_BINS = (1.0, 60.0, 0.5)                         # DualBEV grid_config['depth']: 118 bins, left edges


def depth_centres(d0=DEPTH_BINS[0], d1=DEPTH_BINS[1], step=DEPTH_BINS[2]):
    n = int(round((d1 - d0) / step))
    return d0 + step * (torch.arange(n, dtype=torch.float32) + 0.5)


DEPTHS = 1.0 + 0.5 * torch.arange(118, dtype=torch.float32)                  # frustum: bin i at 1.0 + 0.5 i (left edge)
