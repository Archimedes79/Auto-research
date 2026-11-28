"""Tests of net/model/motion_block.py (its former self-test).

    python -P -m bev_fusion.tests.test_motion_block      (from the problem folder, with its module path)
"""
import torch

from bev_fusion.net.model.lift_pd import Grid
from bev_fusion.net.model.motion_block import MotionBlock, NR, backward_warp, block_inputs, pool_radar, radar_raster


def _selftest():
    torch.manual_seed(0)
    g = Grid(x_min=0.0, x_max=60.0, y_min=-26.0, y_max=26.0, res=0.5)          # the model grid, 120 x 104
    X, Y = g.nx, g.ny
    xs, ys = g.xy_centres()
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")

    def blob(cx, cy):
        return torch.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / 2.0)[None, None]
    # 1. warp sign: a blob at (30, 0) now that moved with v = (8, -4) m/s over dt = 0.5 s was at (26, 2): the old frame
    #    warped with the true v matches the current one, with v = 0 it does not
    cur, old = blob(30.0, 0.0), blob(26.0, 2.0)
    v = torch.tensor([8.0, -4.0]).view(1, 2, 1, 1).expand(1, 2, X, Y)
    dt = torch.tensor([0.5])
    e_true = (backward_warp(old, v, dt, g.res) - cur).abs().sum()
    e_zero = (backward_warp(old, torch.zeros_like(v), dt, g.res) - cur).abs().sum()
    assert e_true < 0.05 * e_zero, (float(e_true), float(e_zero))
    print(f"warp sign ok: |warp(old, v_true) - cur| = {float(e_true):.3f} vs v = 0: {float(e_zero):.3f}")
    # 2. radar raster: two detections in one cell average, one elsewhere; Doppler residual 0 for the radial truth
    det = {"b": torch.tensor([0, 0, 1]), "x": torch.tensor([10.1, 10.3, 40.0]), "y": torch.tensor([0.1, 0.2, -5.0]),
           "psi": torch.tensor([0.0, 0.0, -0.1244]), "vr": torch.tensor([2.0, 4.0, 5.0]), "dt": torch.tensor([0.1, 0.3, 0.0])}
    r = radar_raster(det, 2, g)
    i, j = int((10.1 - g.x_min) / g.res), int((0.1 - g.y_min) / g.res)
    assert r[0, 0, i, j] == 1 and abs(float(r[0, 2, i, j]) - 3.0) < 1e-6 and abs(float(r[0, 5, i, j]) - 0.2) < 1e-6
    assert abs(float(r[0, 3, i, j]) - 1.0) < 1e-6 and int(r[1, 0].sum()) == 1 and int(r[0, 0].sum()) == 1
    pr = pool_radar(r, 4)
    assert pr.shape == (2, NR, X // 4, Y // 4) and abs(float(pr[0, 2].max()) - 3.0) < 1e-5
    print("radar raster ok: mean Doppler, ray and age per cell, pooled per level")
    # 3. the block: zero start (v = 0 on every level), shapes, gradient reaches every learned part
    C = 64
    blk = MotionBlock(C, g)
    curf, oldf = torch.randn(2, C, X, Y), torch.randn(2, C, X, Y)
    out = blk(curf, oldf, torch.ones(2, 1, X, Y), r, torch.tensor([0.33, 0.33]))
    assert len(out) == 3 and all(o.shape == (2, 2, X, Y) for o in out) and all(float(o.detach().abs().max()) == 0 for o in out)
    sum(o.square().mean() + o.mean() for o in out).backward()
    # at the zero start only the last layer of each level gets a gradient (the rest is behind a zero layer); after one step all
    assert all(float(lv.net[-1].weight.grad.abs().sum()) > 0 for lv in blk.levels)
    opt = torch.optim.SGD(blk.parameters(), lr=0.1)
    opt.step()
    blk.zero_grad()
    out = blk(curf, oldf, torch.ones(2, 1, X, Y), r, torch.tensor([0.33, 0.33]))
    sum(o.square().mean() + o.mean() for o in out).backward()
    dead = [n for n, p in blk.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0]
    assert not dead, dead
    print(f"block ok: 3 levels at {X} x {Y}, zero start, gradient reaches all {sum(1 for _ in blk.parameters())} tensors "
          f"after one step; {sum(p.numel() for p in blk.parameters()) / 1e3:.0f} k parameters")
    # 4. it can learn: fit the velocity of a moving blob from (cur, old) in a few hundred steps of Adam
    torch.manual_seed(1)
    blk = MotionBlock(2, g, c=16)
    opt = torch.optim.Adam(blk.parameters(), lr=3e-3)
    vt = torch.tensor([6.0, 3.0])
    for it in range(150):
        cx, cy = 15 + 30 * torch.rand(1).item(), -15 + 30 * torch.rand(1).item()
        c1, o1 = blob(cx, cy), blob(cx - vt[0].item() * 0.5, cy - vt[1].item() * 0.5)
        f_cur, f_old = torch.cat([c1, c1], 1), torch.cat([o1, o1], 1)
        m = c1[:, 0] > 0.3
        outs = blk(f_cur, f_old, torch.ones(1, 1, X, Y), torch.zeros(1, NR, X, Y), torch.tensor([0.5]))
        loss = sum((o.permute(0, 2, 3, 1)[m] - vt).abs().mean() for o in outs)
        opt.zero_grad()
        loss.backward()
        opt.step()
    err = float((outs[-1].permute(0, 2, 3, 1)[m] - vt).norm(dim=-1).mean())
    assert err < 2.0, err
    print(f"learning ok: blob velocity error after 150 steps {err:.2f} m/s (|v| {float(vt.norm()):.1f})")
    # 5. block_inputs (the joint form, t15): the older frame is the window frame ego-aligned to k, dt its age, radar rastered;
    #    a pure forward drive of 10 m/s for 0.33 s moves a feature 3.3 m back; the oldest frame by default; one frame refused
    fr = torch.zeros(1, 3, 2, X, Y)
    fr[0, 2, 0, 60, 52] = 1.0                                             # a point at x = 30.25 m in the oldest frame
    steps = [[], [tuple(torch.tensor([t]) for t in (0.17, 10.0, 0.0))], [tuple(torch.tensor([t]) for t in (0.33, 10.0, 0.0))]]
    det = {"b": torch.zeros(1), "x": torch.tensor([20.1]), "y": torch.tensor([0.1]), "psi": torch.zeros(1), "vr": torch.tensor([-2.0]), "dt": torch.tensor([0.1])}
    cur, old_al, mass, rad, dt = block_inputs(fr, steps, det, g)
    assert cur.shape == (1, 2, X, Y) and abs(float(dt[0]) - 0.33) < 1e-6 and float(rad[0, 2, 40, 52]) == -2.0
    xc = float((old_al[0, 0] * gx).sum() / old_al[0, 0].sum())
    assert abs(xc - (30.25 - 3.3)) < 0.3, xc                               # the ego drove 3.3 m towards it
    for bad in ((fr[:, :1], steps[:1]), (fr, [[], [], []])):              # one frame; an older frame without ego steps
        try:
            block_inputs(bad[0], bad[1], det, g)
            raise AssertionError("must be refused")
        except ValueError:
            pass
    # turning and two hops: a point of the oldest frame lands where chain_positions puts it (the ego turns 0.4 rad/s, 10 m/s)
    from bev_fusion.net.model.splat import chain_positions
    fr2 = torch.zeros(1, 3, 2, X, Y)
    fr2[0, 2, 0, 70, 60] = 1.0                                           # x 35.25 m, y 4.25 m
    hop = tuple(torch.tensor([t]) for t in (0.165, 10.0, 0.4))
    st2 = [[], [hop], [hop, hop]]
    _, al2, _, _, dt2 = block_inputs(fr2, st2, det, g)
    p_exp = chain_positions(torch.tensor([[[35.25, 4.25]]]), [hop, hop])[0, 0]
    xc2 = float((al2[0, 0] * gx).sum() / al2[0, 0].sum())
    yc2 = float((al2[0, 0] * gy).sum() / al2[0, 0].sum())
    assert abs(float(dt2[0]) - 0.33) < 1e-6 and abs(xc2 - float(p_exp[0])) < 0.3 and abs(yc2 - float(p_exp[1])) < 0.3, (xc2, yc2, p_exp)
    print(f"block_inputs ok: older frame ego-aligned (x {xc:.2f} m, expected {30.25 - 3.3:.2f}), dt, radar raster, one frame refused")
    print("motion_block self-test passed")


if __name__ == "__main__":
    _selftest()
