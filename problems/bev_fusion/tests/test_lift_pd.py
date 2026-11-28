"""Tests of net/model/lift_pd.py (its former self-test).

    python tests/test_lift_pd.py      (from the problem folder, with its module path)
"""
import numpy as np
import torch

from net.model.lift_pd import LiftPD, _front_camera, frustum_table, height_to_channels, lift_pd
from net.prep.geometry import Grid, build_table


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

    # 1b the GT generator (v13) reads grid.centres(): numpy cell centres, both (nx, ny), x forward along the first axis
    gx, gy = grid.centres()
    assert gx.shape == gy.shape == (120, 104) and gx.dtype == np.float64
    assert gx[0, 0] == 0.25 and gx[119, 0] == 59.75 and gy[0, 0] == -25.75 and gy[0, 103] == 25.75 and (gx[:, 0] == gx[:, 5]).all()

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
