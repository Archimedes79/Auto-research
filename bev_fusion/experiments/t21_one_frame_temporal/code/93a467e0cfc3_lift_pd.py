"""Lift PD ("pull with depth weight"): camera features -> world voxel volume -> BEV with height as channels.

For every voxel (x, y, z_k) of a grid in vehicle coordinates the lookup table holds its image position and its depth
along the optical axis; the lift gathers the image feature there and multiplies it by the predicted depth probability
at the voxel's own depth:

    V(x, y, z_k) = feature(u, v) * p(z_cam | u, v)

Both gathers are grid_sample calls (bilinear in the image, trilinear in image x depth), so the lift is two gathers and a
product: no scatter, no custom CUDA, exportable. The depth distribution is kept, so the distance uncertainty is kept.
Basis: DualBEV 2403.05402 (HeightTrans, probability sampling), FastBEV++ 2512.08237 (pure-gather form), FB-BEV
2308.02236 (interpolation between adjacent depth bins) - re-implemented here, nothing copied.
See bev_fusion/plan.md section 2.

Conventions
- vehicle frame: x forward, y left, z up (metres); camera frame: x right, y down, z forward (OpenCV)
- T_cam_car: 4x4, maps vehicle points to camera points; K: 3x3 intrinsics of the ORIGINAL image; ida: 3x3 image
  augmentation (resize / crop / flip) applied after K, so pixel = ida @ K @ P_cam / z
- pixel coordinates have the centre of pixel i at i (OpenCV); the feature map covers the (augmented) image, so the
  normalised sampling coordinate is 2 (u + 0.5) / W_img - 1 for any stride
- depth bins: a 1-D tensor of bin centres (metres, increasing), the same bins the depth head predicts

Run the self-test: python training/lift_pd.py   (CPU is enough; prints the timing on CUDA if available)
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    else:
        v = v / depth_prob.shape[1]
    return v.view(B, C, X, Y, Z)


def height_to_channels(vol):
    """(B, C, X, Y, Z) -> (B, Z*C, X, Y): feat[z_car] as channels (Simple-BEV / Fast-BEV style)."""
    B, C, X, Y, Z = vol.shape
    return vol.permute(0, 4, 1, 2, 3).reshape(B, Z * C, X, Y)


def lift_key(K, T_cam_car, image_hw, ida=None):
    """The projection-table cache key: what decides whether the cached table (which image pixel and depth bin every BEV cell and height maps to) can be reused - the camera
    intrinsics K, the camera-to-car extrinsics T_cam_car, the image size and the crop / resize transform ida. Python numbers only. Made from CPU tensors it costs no wait for the
    GPU (`.tolist()` of a GPU tensor does): the batch carries the CPU copies (batch["host"], train.host_side)."""
    return (tuple(K.flatten().tolist()), tuple(T_cam_car.flatten().tolist()), tuple(image_hw),
            None if ida is None else tuple(ida.flatten().tolist()))


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
        """key: the projection-table cache key of this calibration (`lift_key`), made by the caller from the host copies of K, T_cam_car and ida (D54: the key made here
        from GPU tensors waits for the GPU at every `.tolist()`); None: made here, as before. A wrong key gives a wrong table: pass only `lift_key` of the same K, T_cam_car,
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


_VIS_CACHE = {}


_HOST_CONST = {}


def _host_consts(z_levels, depth_centres, oversample, slice_height):
    """(z as a tuple of floats with the oversampled sub-heights, len(depth_centres), first, last) of two model buffers, read from the GPU once: they do not change (the
    entry holds the tensors, so the address is not reused while it is cached, and a change in place counts through `_version`)."""
    dev = depth_centres.device
    zk = (z_levels.data_ptr(), z_levels._version, tuple(z_levels.shape)) if torch.is_tensor(z_levels) else tuple(float(v) for v in z_levels)     # the heights may be a plain tuple
    ck = (zk, depth_centres.data_ptr(), depth_centres._version, tuple(depth_centres.shape), str(dev), oversample, slice_height)
    if ck not in _HOST_CONST:
        if len(_HOST_CONST) > 64:
            _HOST_CONST.clear()
        z = torch.as_tensor(z_levels, dtype=torch.float32, device=dev)
        if oversample > 1:
            off = ((torch.arange(oversample, dtype=z.dtype, device=z.device) + 0.5) / oversample - 0.5) * slice_height
            z = (z.view(-1, 1) + off.view(1, -1)).reshape(-1)
        _HOST_CONST[ck] = (z_levels, depth_centres, tuple(z.tolist()), len(depth_centres), float(depth_centres[0]), float(depth_centres[-1]))
    return _HOST_CONST[ck][2:]


def camera_visibility(grid, z_levels, depth_centres, K, T_cam_car, image_hw, ida=None, oversample=1, slice_height=0.5, host=None):
    """(X, Y) float mask of the cells a camera observes: any of the cell's height samples projects into the image within
    the depth range (the `valid` of build_table, any over height). The obs_mass of the temporal memory - outside it the
    camera BEV is empty, and writing that with full weight erased what the memory had seen before (temporal_fusion
    write, found 2026-10-26). Independent of the lift type; cached per calibration (nuScenes has few).
    host (D54): (K, T_cam_car, ida) as CPU tensors, the same values as the arguments: the cache key is made from them and from the host copies of the two buffers
    (`_host_consts`), so a cached mask costs no wait for the GPU (the key made from GPU tensors costs about ten)."""
    def heights():
        z = torch.as_tensor(z_levels, dtype=torch.float32, device=depth_centres.device)
        if oversample > 1:
            off = ((torch.arange(oversample, dtype=z.dtype, device=z.device) + 0.5) / oversample - 0.5) * slice_height
            z = (z.view(-1, 1) + off.view(1, -1)).reshape(-1)
        return z
    if host is None:
        z = heights()
        key = (grid.x_min, grid.x_max, grid.y_min, grid.y_max, grid.res, tuple(z.tolist()), str(z.device),
               len(depth_centres), float(depth_centres[0]), float(depth_centres[-1]),       # the depth range decides the mask too
               tuple(K.flatten().tolist()), tuple(T_cam_car.flatten().tolist()), tuple(int(v) for v in image_hw),
               None if ida is None else tuple(ida.flatten().tolist()))
    else:
        zt, nd, d0, d1 = _host_consts(z_levels, depth_centres, oversample, slice_height)
        Kh, Th, idah = host
        key = (grid.x_min, grid.x_max, grid.y_min, grid.y_max, grid.res, zt, str(depth_centres.device), nd, d0, d1,
               tuple(Kh.flatten().tolist()), tuple(Th.flatten().tolist()), tuple(int(v) for v in image_hw),
               None if idah is None else tuple(idah.flatten().tolist()))
    if key not in _VIS_CACHE:
        z = heights() if host is not None else z
        if len(_VIS_CACHE) > 64:
            _VIS_CACHE.clear()
        tab = build_table(grid, z, K.to(z.device), T_cam_car.to(z.device), image_hw, depth_centres, None if ida is None
                          else ida.to(z.device))
        _VIS_CACHE[key] = tab["valid"].view(grid.nx, grid.ny, -1).amax(-1).detach()
    return _VIS_CACHE[key]


# ----------------------------------------------------------------------------------------------- self-test
def _front_camera(dtype=torch.float64, x0=0.0):
    """A synthetic front camera: 1.5 m high, looking along +x, f = 633 px on an 800 x 448 image (half resolution)."""
    K = torch.tensor([[633.0, 0, 399.5], [0, 633.0, 223.5], [0, 0, 1]], dtype=dtype)
    R = torch.tensor([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=dtype)                # car -> cam axes
    t = -R @ torch.tensor([x0, 0.0, 1.5], dtype=dtype)
    T = torch.eye(4, dtype=dtype)
    T[:3, :3], T[:3, 3] = R, t
    return K, T, (448, 800)


def _selftest():
    torch.manual_seed(0)
    dt = torch.float64
    grid = Grid()
    dc = torch.arange(1.0, 60.01, 0.5, dtype=dt)                                       # 119 bins (any bins work)
    D = len(dc)
    K, T, hw = _front_camera(dt)
    zl = torch.tensor(grid.z_levels, dtype=dt)
    tab = build_table(grid, zl, K, T, hw, dc)
    X, Y, Z = tab["shape"]
    assert (X, Y, Z) == (120, 104, 8)

    # 1 geometry: the voxel straight ahead at camera height projects to the principal point, depth = its x
    xs, ys = grid.xy_centres(dtype=dt)
    ix, iy, iz = 39, 52, 4                                    # x = 19.75 m, y = +0.25 m, z = 1.25 m
    i = (ix * Y + iy) * Z + iz
    x, y, z = xs[ix].item(), ys[iy].item(), zl[iz].item()
    u_exp = 399.5 + 633.0 * (-y) / x
    v_exp = 223.5 + 633.0 * (1.5 - z) / x
    xn, yn, dn = tab["grid3d"][i].tolist()
    assert abs((xn + 1) * 800 / 2 - 0.5 - u_exp) < 1e-9 and abs((yn + 1) * 448 / 2 - 0.5 - v_exp) < 1e-9
    frac = ((dn + 1) * D - 1) / 2
    assert abs(dc[0].item() + 0.5 * frac - x) < 1e-9, "depth bin index"
    assert tab["valid"][i] == 1

    # 2 uniform depth: PD = P / D
    feat = torch.randn(1, 4, 56, 100, dtype=dt)
    uni = torch.full((1, D, 56, 100), 1.0 / D, dtype=dt)
    vp = lift_pd(feat, uni, tab, use_depth=False)
    vpd = lift_pd(feat, uni, tab, use_depth=True)
    m = tab["valid"].view(X, Y, Z).bool()
    assert torch.allclose(vpd[0, :, m], vp[0, :, m] / D, atol=1e-12), "uniform depth"

    # 3 one-hot depth at bin b: only voxels within one bin of depth dc[b] receive the feature, weight 1 at dc[b]
    b = 38                                                     # 20.0 m
    one = torch.zeros(1, D, 56, 100, dtype=dt)
    one[:, b] = 1
    ones = torch.ones(1, 1, 56, 100, dtype=dt)
    v1 = lift_pd(ones, one, tab)[0, 0]
    depth = xs.view(X, 1, 1).expand(X, Y, Z)                  # depth along the optical axis = x for this camera
    hit = v1 > 1e-9
    assert hit.any() and (depth[hit] - dc[b]).abs().max() < 0.5 + 1e-9, "one-hot depth support"
    near = (depth - dc[b]).abs() < 1e-9
    assert torch.allclose(v1[near & m], torch.ones_like(v1[near & m])), "one-hot depth peak"

    # 4 rig independence: camera 1 m further forward, same image -> BEV shifted by 2 cells
    K2, T2, _ = _front_camera(dt, x0=1.0)
    tab2 = build_table(grid, zl, K2, T2, hw, dc)
    a = lift_pd(feat, uni, tab, use_depth=False)
    c = lift_pd(feat, uni, tab2, use_depth=False)
    assert torch.allclose(c[:, :, 2:], a[:, :, :-2], atol=1e-12), "rig shift"

    # 8 frustum convention: a feature ramp along the width read at a voxel gives the feature index u (w - 1) / (W - 1) of its pixel u
    h_f, w_f = 14, 25
    ramp = torch.arange(w_f, dtype=dt).view(1, 1, 1, w_f).expand(1, 1, h_f, w_f).contiguous()
    tf = frustum_table(tab, (h_f, w_f), hw)
    got = lift_pd(ramp, torch.ones(1, len(dc), h_f, w_f, dtype=dt) / len(dc), tf, use_depth=False)[0, 0].reshape(-1)
    pix_u = (tab["grid2d"][:, 0] + 1) * hw[1] / 2 - 0.5
    sel = (tab["valid"] > 0) & (pix_u >= 0) & (pix_u <= hw[1] - 1)
    assert sel.sum() > 1000
    assert torch.allclose(got[sel], (pix_u * (w_f - 1) / (hw[1] - 1))[sel], atol=1e-9), "frustum cells"
    ctr = lift_pd(ramp, torch.ones(1, len(dc), h_f, w_f, dtype=dt) / len(dc), tab, use_depth=False)[0, 0].reshape(-1)
    idx_c = (pix_u + 0.5) * w_f / hw[1] - 0.5
    sel_c = sel & (idx_c >= 0) & (idx_c <= w_f - 1)                                  # inside the cell centres (border clamping outside)
    assert torch.allclose(ctr[sel_c], idx_c[sel_c], atol=1e-9), "centre cells (default)"
    assert (got[sel_c] - ctr[sel_c]).abs().max() > 0.1, "the two conventions differ by up to half a cell"
    mod_f = LiftPD(grid, dc, use_depth=False, frustum_cells=True)
    mod_c = LiftPD(grid, dc, use_depth=False)
    Kf, Tf, _ = _front_camera(torch.float32)
    f_in = torch.randn(1, 3, h_f, w_f)
    assert not torch.allclose(mod_f(f_in, f_in[:, :1], Kf, Tf, hw), mod_c(f_in, f_in[:, :1], Kf, Tf, hw)), "option changes the lift"

    # 5 gradients reach the feature, the depth distribution and learned heights (gradcheck on a tiny grid)
    small = Grid(x_min=4, x_max=8, y_min=-1, y_max=1, res=1.0, z_levels=(0.0, 1.0))
    f_s = torch.randn(1, 2, 7, 12, dtype=dt, requires_grad=True)
    p_s = torch.softmax(torch.randn(1, 6, 7, 12, dtype=dt), 1).requires_grad_(True)
    dc_s = torch.linspace(3.0, 9.0, 6, dtype=dt)
    z_s = torch.tensor([0.3, 1.1], dtype=dt, requires_grad=True)
    Ks = torch.tensor([[10.0, 0, 5.5], [0, 10.0, 3.0], [0, 0, 1]], dtype=dt)

    def run(f, p, zz):
        tb = build_table(small, zz, Ks, T, (7, 12), dc_s)
        return lift_pd(f, p, tb)
    assert torch.autograd.gradcheck(run, (f_s, p_s, z_s), eps=1e-6, atol=1e-5), "gradcheck"

    # 6 module: height into channels, table cache, learned heights get a gradient
    mod = LiftPD(grid, dc.float(), learn_heights=True)
    out = mod(feat.float(), uni.float(), K.float(), T.float(), hw)
    assert out.shape == (1, 8 * 4, 120, 104)
    out.sum().backward()
    assert mod.z_levels.grad is not None and torch.isfinite(mod.z_levels.grad).all()

    # 7 vertical oversampling = the mean over the sub-heights of each slice (checked against explicit sub-slices)
    n = 4
    mod4 = LiftPD(grid, dc.float(), oversample=n)
    o4 = mod4(feat.float(), uni.float(), K.float(), T.float(), hw)
    off = ((torch.arange(n) + 0.5) / n - 0.5) * 0.5
    zsub = (torch.tensor(grid.z_levels).view(-1, 1) + off.view(1, -1)).reshape(-1)
    ref = lift_pd(feat.float(), uni.float(), build_table(grid, zsub, K.float(), T.float(), hw, dc.float()))
    ref = height_to_channels(ref.view(1, 4, 120, 104, 8, n).mean(-1))
    assert o4.shape == out.shape and torch.allclose(o4, ref, atol=1e-6), "oversampling"
    print("lift_pd selftest ok: geometry, uniform = P / D, one-hot depth, rig shift, gradcheck, module, oversampling, frustum cells")

    if torch.cuda.is_available():
        import time
        dev = "cuda"
        mod = LiftPD(grid, dc.float()).to(dev)
        f = torch.randn(1, 64, 56, 100, device=dev)
        p = torch.softmax(torch.randn(1, D, 56, 100, device=dev), 1)
        Kc, Tc = K.float().to(dev), T.float().to(dev)
        for _ in range(10):
            mod(f, p, Kc, Tc, hw)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            mod(f, p, Kc, Tc, hw)
        torch.cuda.synchronize()
        print(f"lift PD on {torch.cuda.get_device_name(0)}: {(time.perf_counter() - t0) * 10:.3f} ms per frame "
              f"(1 camera, C=64, D={D}, 120 x 104 x 8 voxels; GPU may be shared - not a timing number)")


if __name__ == "__main__":
    _selftest()
