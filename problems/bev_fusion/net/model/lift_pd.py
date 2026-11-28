"""Lift PD ("pull with depth weight"): camera features -> world voxel volume -> BEV with height as channels.

For every voxel (x, y, z_k) of a grid in vehicle coordinates the lookup table holds its image position and its depth
along the optical axis; the lift gathers the image feature there and multiplies it by the predicted depth probability
at the voxel's own depth:

    V(x, y, z_k) = feature(u, v) * p(z_cam | u, v)

Both gathers are grid_sample calls (bilinear in the image, trilinear in image x depth), so the lift is two gathers and a
product: no scatter, no custom CUDA, exportable. The depth distribution is kept, so the distance uncertainty is kept.
Basis: DualBEV 2403.05402 (HeightTrans, probability sampling), FastBEV++ 2512.08237 (pure-gather form), FB-BEV
2308.02236 (interpolation between adjacent depth bins) - re-implemented here, nothing copied.

Conventions
- vehicle frame: x forward, y left, z up (metres); camera frame: x right, y down, z forward (OpenCV)
- T_cam_car: 4x4, maps vehicle points to camera points; K: 3x3 intrinsics of the ORIGINAL image; ida: 3x3 image
  augmentation (resize / crop / flip) applied after K, so pixel = ida @ K @ P_cam / z
- pixel coordinates have the centre of pixel i at i (OpenCV); the feature map covers the (augmented) image, so the
  normalised sampling coordinate is 2 (u + 0.5) / W_img - 1 for any stride
- depth bins: a 1-D tensor of bin centres (metres, increasing), the same bins the depth head predicts

Tests: tests/test_lift_pd.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from net.prep.geometry import Grid, build_table, camera_visibility, lift_key  # noqa: F401 (preprocessing geometry, read here by name)


def frustum_table(table, feat_hw, image_hw):
    """Re-express a table's image coordinates for feature maps in the frustum convention: cell j of w sits at pixel j (W - 1) / (w - 1)
    (and alike in y), so the feature index of pixel u is j = u (w - 1) / (W - 1); the returned grid2d / grid3d are the normalised
    coordinates grid_sample (align_corners=False) needs for that index. The depth coordinate and the valid mask are unchanged."""
    h, w = feat_hw
    H, W = image_hw

    def remap(xn, size, fsize):
        pix = (xn + 1) * size / 2 - 0.5
        return 2 * (pix * (fsize - 1) / (size - 1) + 0.5) / fsize - 1
    g2 = table["grid2d"]
    xy = torch.stack([remap(g2[:, 0], W, w), remap(g2[:, 1], H, h)], -1)
    out = dict(table)
    out["grid2d"] = xy
    out["grid3d"] = torch.cat([xy, table["grid3d"][:, 2:]], -1)
    return out


def lift_pd(feat, depth_prob, table, use_depth=True):
    """feat (B, C, Hf, Wf), depth_prob (B, D, Hf, Wf) (softmax over D) -> volume (B, C, X, Y, Z).

    use_depth=False gives lift P (pull without depth weight) on the same table.
    """
    B, C = feat.shape[:2]
    X, Y, Z = table["shape"]
    n = X * Y * Z
    g2 = table["grid2d"].view(1, n, 1, 2).expand(B, n, 1, 2).to(feat.dtype)
    f = F.grid_sample(feat, g2, mode="bilinear", padding_mode="border", align_corners=False)     # (B, C, N, 1)
    v = f[..., 0] * table["valid"].to(feat.dtype)
    if use_depth:
        g3 = table["grid3d"].view(1, n, 1, 1, 3).expand(B, n, 1, 1, 3).to(depth_prob.dtype)
        p = F.grid_sample(depth_prob.unsqueeze(1), g3, mode="bilinear", padding_mode="border", align_corners=False)
        v = v * p[:, 0, :, 0, 0].unsqueeze(1)                                                    # (B, 1, N)
    return v.view(B, C, X, Y, Z)


def height_to_channels(vol):
    """(B, C, X, Y, Z) -> (B, Z*C, X, Y): feat[z_car] as channels (Simple-BEV / Fast-BEV style)."""
    B, C, X, Y, Z = vol.shape
    return vol.permute(0, 4, 1, 2, 3).reshape(B, Z * C, X, Y)


class LiftPD(nn.Module):
    """Lift PD for one camera with optional learned elevations; the table is cached unless the heights are learned."""

    def __init__(self, grid, depth_centres, learn_heights=False, use_depth=True, oversample=1, slice_height=0.5,
                 frustum_cells=False):
        """oversample = n > 1: every height slice is the mean of n sub-heights spread over its slice_height - a
        numerical integration over the slice, so thin surfaces between slice centres (the road at z = 0) are not
        missed; synthetic oracle (tests/test_oracle_lift.py, stride 8, depth histogram): lane-line recall 0-20 / 20-40
        / 40-60 m from 0.90 / 0.08 / 0.00 (n = 1) to 1.00 / 1.00 / 0.81 (n = 4).
        frustum_cells: feature cell j sits at pixel j (W - 1) / (w - 1) (the frustum convention of LSS / BEVFusion, whose
        networks were trained with it) instead of the cell centre (j + 0.5) W / w - 0.5 (DualBEV's cached heads, default)."""
        super().__init__()
        self.grid, self.use_depth, self.frustum_cells = grid, use_depth, frustum_cells
        self.oversample, self.slice_height = oversample, slice_height
        self.register_buffer("depth_centres", torch.as_tensor(depth_centres, dtype=torch.float32))
        z = torch.tensor(grid.z_levels, dtype=torch.float32)
        if learn_heights:
            self.z_levels = nn.Parameter(z)
        else:
            self.register_buffer("z_levels", z)
        self._cache = None

    def forward(self, feat, depth_prob, K, T_cam_car, image_hw, ida=None, key=None):
        """key: the projection-table cache key of this calibration (`lift_key`), made by the caller from the host copies of K, T_cam_car and ida (D54: the key
        made here
        from GPU tensors waits for the GPU at every `.tolist()`); None: made here, as before. A wrong key gives a wrong table: pass only `lift_key` of the same
        K, T_cam_car,
        image_hw and ida."""
        learned = isinstance(self.z_levels, nn.Parameter)
        if key is None:
            key = lift_key(K, T_cam_car, image_hw, ida)
        n = self.oversample
        if learned or self._cache is None or self._cache[0] != key:
            z = self.z_levels
            if n > 1:
                off = ((torch.arange(n, dtype=z.dtype, device=z.device) + 0.5) / n - 0.5) * self.slice_height
                z = (z.view(-1, 1) + off.view(1, -1)).reshape(-1)
            table = build_table(self.grid, z, K, T_cam_car, image_hw, self.depth_centres, ida)
            if not learned:
                table = {k: (t.detach() if torch.is_tensor(t) else t) for k, t in table.items()}
                self._cache = (key, table)
        else:
            table = self._cache[1]
        if self.frustum_cells:
            table = frustum_table(table, feat.shape[-2:], image_hw)
        vol = lift_pd(feat, depth_prob, table, self.use_depth)
        if n > 1:
            B, C, X, Y, Zn = vol.shape
            vol = vol.view(B, C, X, Y, Zn // n, n).mean(-1)
        return height_to_channels(vol)

    def visibility(self, K, T_cam_car, image_hw, ida=None):
        """(X, Y) mask of the cells this camera observes (camera_visibility with this lift's heights)."""
        return camera_visibility(self.grid, self.z_levels.detach(), self.depth_centres, K, T_cam_car, image_hw, ida,
                                 self.oversample, self.slice_height)


# ----------------------------------------------------------------------------------------------- self-test
def _front_camera(dtype=torch.float64, x0=0.0):
    """A synthetic front camera: 1.5 m high, looking along +x, f = 633 px on an 800 x 448 image (half resolution)."""
    K = torch.tensor([[633.0, 0, 399.5], [0, 633.0, 223.5], [0, 0, 1]], dtype=dtype)
    R = torch.tensor([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=dtype)                # car -> cam axes
    t = -R @ torch.tensor([x0, 0.0, 1.5], dtype=dtype)
    T = torch.eye(4, dtype=dtype)
    T[:3, :3], T[:3, 3] = R, t
    return K, T, (448, 800)
