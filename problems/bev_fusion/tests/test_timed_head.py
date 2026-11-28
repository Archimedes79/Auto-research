"""Tests of net/model/timed_head.py (its former self-test).

    python -P -m bev_fusion.tests.test_timed_head      (from the problem folder, with its module path)
"""
import math

import torch
import torch.nn as nn

from bev_fusion.net.model.lift_pd import Grid
from bev_fusion.net.model.timed_head import TimeMover, TimedHead


def _selftest():
    from bev_fusion.net.loss.motion_loss import motion_loss
    torch.manual_seed(0)
    grid, C = Grid(), 16
    X, Y = grid.nx, grid.ny
    xs = grid.x_min + (torch.arange(X) + 0.5) * grid.res
    ys = grid.y_min + (torch.arange(Y) + 0.5) * grid.res

    def blob(x0, y0, s=1.5):
        b = torch.exp(-((xs[:, None] - x0) ** 2 + (ys[None, :] - y0) ** 2) / (2 * s * s))
        S = torch.zeros(1, C, X, Y)
        S[0, 0] = b
        return S

    def centroid_x(m):
        return float((m[0, 0] * xs[:, None]).sum() / m[0, 0].sum())

    pick = nn.Conv2d(C, 1, 1, bias=False)              # a head that reads channel 0
    with torch.no_grad():
        pick.weight.zero_()
        pick.weight[0, 0, 0, 0] = 1.0
    mover = TimeMover(C, grid)
    th = TimedHead(pick, C, mover)
    S = blob(30.0, 0.0)
    # 1 dt = 0: exactly the plain head (zero-started refine, identity splat), no unknown cells
    y, aux = th(S, 0.0, 10.0, 0.0, return_aux=True)
    assert torch.allclose(y, pick(S), atol=1e-6), "dt = 0 must be the plain head"
    assert float(aux["mass"].min()) > 0.99, "dt = 0: all cells known"
    # 2 ego motion: static content moves back by v dt (the car drives toward it); cells beyond the old far edge are unknown
    for dt, x_exp in ((0.5, 25.0), (-0.5, 35.0)):
        y, aux = th(S, dt, 10.0, 0.0, return_aux=True)
        assert abs(centroid_x(y) - x_exp) < 0.1, (dt, centroid_x(y))
    unk = lambda dt: float((th(S, dt, 10.0, 0.0, return_aux=True)[1]["mass"] < 0.5).float().mean())
    u5, u10 = unk(0.5), unk(1.0)
    assert 0.07 < u5 < 0.10 and 0.15 < u10 < 0.19 and u10 > u5, (u5, u10)       # about v dt / 60 m of the map is unknown
    # 3 object velocity: set the velocity head by hand to +4 m/s: the blob lands at x + v_obj dt (ego standing still)
    with torch.no_grad():
        mover.motion.f.net[2].bias.copy_(torch.tensor([40.0 * math.atanh(0.1), 0.0]))      # v_hat = 4.0 m/s along x
    y, aux = th(S, 0.5, 0.0, 0.0, return_aux=True)
    assert abs(float(aux["v_hat"][0, 0].mean()) - 4.0) < 1e-3
    assert abs(centroid_x(y) - 32.0) < 0.1, centroid_x(y)
    with torch.no_grad():
        mover.motion.f.net[2].bias.zero_()
    # 3b a field from elsewhere (the motion block, t15): v_field replaces the head and lands the blob the same way; carry moves the
    #    field with the state (the box velocity at dt); without them the output is bit-equal to the plain call
    vf = torch.zeros(1, 2, X, Y)
    vf[:, 0] = 4.0
    y, aux = th(S, 0.5, 0.0, 0.0, return_aux=True, v_field=vf)
    assert abs(centroid_x(y) - 32.0) < 0.1 and aux["v_hat"] is vf and aux["carry"] is None, centroid_x(y)
    y2, aux2 = th(S, 0.5, 0.0, 0.0, return_aux=True, v_field=vf, carry=vf)
    assert torch.equal(y2, y) and aux2["carry"].shape == vf.shape
    landed = aux2["mass"][0, 0] > 0.99
    assert torch.allclose(aux2["carry"][0, 0][landed], torch.full_like(aux2["carry"][0, 0][landed], 4.0), atol=1e-4)
    # a NON-constant carry is moved, not copied: a ramp r(x) = x carried with the ego driving 10 m/s for 0.5 s reads r(x + 5) where it landed
    ramp = xs.view(1, 1, X, 1).expand(1, 1, X, Y).clone()
    _, aux3 = th(S, 0.5, 10.0, 0.0, return_aux=True, v_field=torch.zeros(1, 2, X, Y), carry=ramp)
    inner = (aux3["mass"][0, 0] > 0.99) & (xs.view(X, 1) < xs[-1] - 6.0)
    exp = (xs.view(X, 1) + 5.0).expand(X, Y)
    assert inner.sum() > 1000 and torch.allclose(aux3["carry"][0, 0][inner], exp[inner], atol=1e-3), "the carry is moved with the state"
    # the loose edges: carry without v_field uses the head on the STATE only; carry needs return_aux; a mover without head needs v_field
    _, aux4 = th(S, 0.5, 0.0, 0.0, return_aux=True, carry=ramp)
    assert aux4["carry"].shape == ramp.shape and torch.equal(aux4["v_hat"], mover.motion(S))
    try:
        th(S, 0.5, 0.0, 0.0, carry=ramp)
        raise AssertionError("carry without return_aux must raise")
    except ValueError:
        pass
    headless = TimeMover(C, grid)
    headless.motion = None
    try:
        headless(S, 0.5, 0.0, 0.0)
        raise AssertionError("a mover without its head must refuse a call without v_field")
    except ValueError:
        pass
    assert torch.equal(headless(S, 0.5, 0.0, 0.0, v_field=vf)[0], mover(S, 0.5, 0.0, 0.0, v_field=vf)[0])
    # 4 the motion loss: zero for the right velocity, background pulled to rest, parked objects (< 1 m/s) count as 0
    v_gt = torch.zeros(1, 2, X, Y)
    v_gt[0, 0, 40:50, 50:55] = 6.0
    mask = torch.zeros(1, 1, X, Y)
    mask[0, 0, 38:52, 48:57] = 1.0
    assert float(motion_loss(v_gt.clone(), v_gt, mask)) < 1e-6
    assert float(motion_loss(torch.zeros_like(v_gt), v_gt, mask)) > 0.5
    # 5 several heads share ONE mover; gradients reach the film, the refine block and the motion head
    from bev_fusion.net.model.bev_trunk import SemanticHead
    from bev_fusion.net.model.object_head import ObjectHead
    sem, obj = TimedHead(SemanticHead(C, 5), C, mover), TimedHead(ObjectHead(C), C, mover)
    assert sem.mover is obj.mover and sem.mover.motion.f.net[0].weight is obj.mover.motion.f.net[0].weight
    S2 = torch.randn(2, C, X, Y)
    ys = sem(S2, 0.5, 8.0, 0.1)
    yo = obj(S2, [0.5, -0.5], [8.0, 6.0], [0.1, -0.1])
    assert ys.shape == (2, 5, X, Y) and set(yo) == {"hm", "off", "size", "rot", "rad", "vel"}
    mover.zero_grad()
    for p in list(sem.parameters()) + list(obj.parameters()):
        p.grad = None
    (ys.mean() + yo["hm"].mean()).backward()
    assert sem.film.weight.grad is not None and obj.out.weight.grad is not None
    assert mover.motion.f.net[2].weight.grad is not None, "the motion head gets a gradient through the splat"
    n_shared = len({id(p) for p in list(sem.parameters()) + list(obj.parameters())})
    n_sum = len(list(sem.parameters())) + len(list(obj.parameters()))
    assert n_shared < n_sum, "the mover's parameters are shared, not duplicated"
    print(f"timed_head selftest ok: dt = 0 is the plain head, ego motion lands at x -+ v dt, unknown share {u5:.3f} / {u10:.3f} "
          f"at dt 0.5 / 1.0 s, object velocity lands at x + v dt, motion loss, two heads share one mover, gradients")


if __name__ == "__main__":
    _selftest()
