"""Tests of net/loss/loss_extras.py (its former self-test).

    python tests/test_loss_extras.py      (from the problem folder, with its module path)
"""
import math

import torch

from net.loss.loss_extras import cross_time_consistency, group_gradient_cosines
from net.model.lift_pd import Grid


def _blob(grid, cx, cy, s=1.2, device="cpu"):
    xs, ys = grid.xy_centres(device)
    return torch.exp(-(((xs.view(-1, 1) - cx) ** 2 + (ys.view(1, -1) - cy) ** 2) / (2 * s * s)))[None, None]


def _selftest():
    torch.manual_seed(0)
    # 2 group gradient cosines: parallel, opposite and orthogonal groups
    f = torch.randn(4, 3, requires_grad=True)
    g_par, g_opp, g_orth = (f * 2).sum(), (-f * 3).sum(), (f[:, 0] * 0 + f[:, 1].sum() * 0 + (f * torch.tensor([[0.0, 0.0, 1.0]]).expand_as(f)).sum() * 0).sum()
    base = (f * torch.tensor([[1.0, 0.0, 0.0]])).sum()                          # gradient (1, 0, 0) per row
    orth = (f * torch.tensor([[0.0, 1.0, 0.0]])).sum()
    names, C, nrm = group_gradient_cosines({"base": base, "par": base * 2, "opp": -base, "orth": orth}, f)
    i = {n: k for k, n in enumerate(names)}
    assert abs(C[i["base"], i["par"]] - 1) < 1e-6 and abs(C[i["base"], i["opp"]] + 1) < 1e-6 and abs(C[i["base"], i["orth"]]) < 1e-6
    assert abs(nrm[i["par"]] / nrm[i["base"]] - 2) < 1e-5

    # 3 cross-time consistency: a blob moving at 8 m/s, ego static, the probabilities at dt = 0 and +-0.5 s exact
    grid = Grid(x_min=0.0, x_max=32.0, y_min=-8.0, y_max=8.0, res=0.5)
    X, Y = grid.nx, grid.ny
    P = {0.0: _blob(grid, 12.0, -2.0), 0.5: _blob(grid, 16.0, -2.0), -0.5: _blob(grid, 8.0, -2.0)}
    st = {0.5: [[(0.5, 0.0, 0.0)]], -0.5: [[(-0.5, 0.0, 0.0)]]}
    v_true = torch.zeros(1, 2, X, Y)
    v_true[:, 0] = 8.0 * (P[0.0][:, 0] > 0.02).float()                          # the field is nonzero on the object only
    r = cross_time_consistency(P, st, v_true, grid)
    r0 = cross_time_consistency(P, st, torch.zeros(1, 2, X, Y), grid)           # no motion: the blob does not match
    assert r["total"] < 0.5 * r0["total"] and r["matched@+0.5"] > 0.9, (float(r["total"]), float(r0["total"]))
    # trivial solution: a motion field that pushes every cell out of the grid costs the constant penalty, not 0
    far = torch.zeros(1, 2, X, Y)
    far[:, 0] = 400.0
    rf = cross_time_consistency(P, st, far, grid, penalty=0.5)
    assert rf["total"] > 0.4 and rf["matched@+0.5"] < 0.05, float(rf["total"])
    # ego motion exact: a static pattern seen from a moved ego (v = 10 m/s, omega = 0.3 rad/s, dt = 0.5 s): P_dt rendered
    #   from the exact motion; with v_hat = 0 the consistency is small, with a wrong ego step it is not
    xs, ys = grid.xy_centres()
    v_e, w_e, dte = 10.0, 0.3, 0.5
    th = w_e * dte
    dxe, dye = v_e / w_e * math.sin(th), v_e / w_e * (1 - math.cos(th))

    def world(x, y):                                                            # a smooth static world, ego frame at t = 0
        return torch.exp(-(((x - 16.0) ** 2 + (y - 1.0) ** 2) / 8.0)) + 0.6 * torch.exp(-(((x - 8.0) ** 2 + (y + 3.0) ** 2) / 6.0))
    gx, gy = xs.view(-1, 1).expand(X, Y), ys.view(1, -1).expand(X, Y)
    qx, qy = gx - dxe, gy - dye
    # the cell (gx, gy) of the frame at dt sees the world point R(th) (gx, gy) + d in the frame at 0
    wx, wy = math.cos(th) * gx - math.sin(th) * gy + dxe, math.sin(th) * gx + math.cos(th) * gy + dye
    Pe = {0.0: world(gx, gy)[None, None], 0.5: world(wx, wy)[None, None]}
    rst = cross_time_consistency(Pe, {0.5: [[(0.5, v_e, w_e)]]}, torch.zeros(1, 2, X, Y), grid)
    rbad = cross_time_consistency(Pe, {0.5: [[(0.5, 0.0, 0.0)]]}, torch.zeros(1, 2, X, Y), grid)
    # (the totals contain the constant penalty for the cells that enter the field of view through the ego motion: judge the
    #  error over the matched cells)
    assert rst["ferr@+0.5"] < 0.5 * rbad["ferr@+0.5"] and rst["berr@+0.5"] < 0.5 * rbad["berr@+0.5"], \
        (float(rst["ferr@+0.5"]), float(rbad["ferr@+0.5"]), float(rst["berr@+0.5"]), float(rbad["berr@+0.5"]))
    # stop-gradient directions: forward carries a gradient to P_0 only, backward to P_dt only (through the gather)
    Pg = {0.0: P[0.0].clone().requires_grad_(True), 0.5: P[0.5].clone().requires_grad_(True)}
    rr = cross_time_consistency(Pg, {0.5: st[0.5]}, v_true, grid)
    g_fwd = torch.autograd.grad(rr["fwd@+0.5"], [Pg[0.0], Pg[0.5]], retain_graph=True, allow_unused=True)
    assert g_fwd[0] is not None and g_fwd[0].abs().sum() > 0 and (g_fwd[1] is None or g_fwd[1].abs().sum() == 0)
    g_bwd = torch.autograd.grad(rr["bwd@+0.5"], [Pg[0.0], Pg[0.5]], retain_graph=True, allow_unused=True)
    assert g_bwd[1] is not None and g_bwd[1].abs().sum() > 0 and (g_bwd[0] is None or g_bwd[0].abs().sum() == 0)
    # the motion field gets a gradient (the self-supervised flow signal); cells of weight 0 are ignored
    vg = v_true.clone().requires_grad_(True)
    cross_time_consistency(P, st, vg, grid)["total"].backward()
    assert vg.grad is not None and vg.grad.abs().sum() > 0
    wv = {0.0: torch.ones(1, 1, X, Y), 0.5: torch.zeros(1, 1, X, Y), -0.5: torch.zeros(1, 1, X, Y)}
    assert float(cross_time_consistency(P, st, v_true, grid, weights=wv)["total"]) == 0.0, "no visible cell: no loss"
    print("loss_extras test ok: group gradient cosines (parallel / opposite / orthogonal), "
          f"cross-time consistency (moving blob {float(r['total']):.3f} vs no motion {float(r0['total']):.3f}; "
          f"pushed-out field {float(rf['total']):.2f} not 0; ego motion exact: matched-cell error "
          f"{float(rst['ferr@+0.5']):.3f} / {float(rst['berr@+0.5']):.3f} (fwd / bwd) vs wrong step "
          f"{float(rbad['ferr@+0.5']):.3f} / {float(rbad['berr@+0.5']):.3f}; stop-gradient per direction; motion field "
          f"gets a gradient)")


if __name__ == "__main__":
    _selftest()
