"""Gate of the temporal propagation block (training/temporal.py): smoke tests with known answers, then end-to-end
speed learning without velocity labels.

Smoke tests (each prints PASS / FAIL): identity at dt = 0; exact integer shift by ego motion; landing points against
the EXACT constant-turn-rate motion with changing speed and yaw rate; batch with different ego motion per sample =
each sample alone; padded step lists of different lengths; point mode = grid mode; a two-hypothesis mixture splits
the mass; Doppler evidence selects the right hypothesis (also through aliasing); forward then backward returns;
float32 / CUDA = float64 / CPU; empty maps and sources outside the grid stay finite; gradcheck (bilinear path).
End-to-end test: a world of two object types with fixed velocities (A, B) and static stripes, random ego motion; the
source map (types at t - 0.5 s) is propagated by the block and compared with the target map (MSE) - NO velocity
label. Learned velocity per type after training, for: (1) one hypothesis, bilinear, starting at 0 (the plain splat);
(2) one hypothesis with a Gaussian kernel annealed from 2 cells to bilinear; (3) four hypotheses + static, bilinear;
(4) (3) + Doppler evidence (radial velocity along rays from the origin, noise 0.1 m/s).
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_temporal.py [--quick]
"""
import math
import random
import sys

import torch

sys.path.insert(0, "training")
from lift_pd import Grid  # noqa: E402
from temporal import (FlowPyramid, FlowRefine, GapFill, doppler_update, TemporalPropagation, gaussian_splat, landing, propagate, sample_at,  # noqa: E402
                      steps_tensor, wrap)

QUICK = "--quick" in sys.argv
G = Grid(x_min=0, x_max=20, y_min=-10, y_max=10, res=0.5)                     # 40 x 40 cells
D64 = torch.float64
RESULTS = []


def check(name, ok, info=""):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  (' + info + ')') if info else ''}")


def exact(px, py, steps, vx, vy):
    """exact constant-turn-rate ego motion over the steps, the object moving with a world-constant velocity."""
    ex, ey, eh, t = 0.0, 0.0, 0.0, 0.0
    for dt, v, w in steps:
        th = w * dt
        lx, ly = (v * dt, 0.0) if abs(w) < 1e-12 else (v / w * math.sin(th), v / w * (1 - math.cos(th)))
        ex, ey = ex + math.cos(eh) * lx - math.sin(eh) * ly, ey + math.sin(eh) * lx + math.cos(eh) * ly
        eh += th
        t += dt
    qx, qy = px + vx * t - ex, py + vy * t - ey
    return math.cos(eh) * qx + math.sin(eh) * qy, -math.sin(eh) * qx + math.cos(eh) * qy


def spike(c=1, at=(20, 20), B=1):
    x = torch.zeros(B, c, G.nx, G.ny, dtype=D64)
    x[:, :, at[0], at[1]] = 1.0
    return x


def smoke():
    torch.manual_seed(0)
    tp = TemporalPropagation(cin=2, grid=G, n_hyp=2).double()
    x = torch.randn(1, 2, G.nx, G.ny, dtype=D64)
    tp.set_velocity((0.0, 0.0), static_weight=0.0)
    r = tp(x, [(0.0, 0.0, 0.0)])
    check("identity at dt = 0 (whole map)", torch.allclose(r["feat"], x, atol=1e-9) and torch.allclose(r["mass"], torch.ones_like(r["mass"])))

    r = tp(x, [(0.1, 10.0, 0.0)])                                           # 1 m forward = 2 cells back
    check("ego motion 1 m = exact 2-cell shift", torch.allclose(r["feat"][:, :, :-2], x[:, :, 2:], atol=1e-9))

    steps = [(1 / 12, 8.0 + i, 0.5 - 0.15 * i) for i in range(6)]
    tp.set_velocity((6.0, -1.0))
    pts = torch.tensor([[[12.0, 3.0], [5.0, -6.0]]], dtype=D64)
    rp = tp.forward_points(torch.ones(1, 2, 2, dtype=D64), pts, steps)
    tgt = rp["target"][0, 2:]                                               # the learned hypothesis (index 1) rows
    err = max(math.hypot(tgt[j, 0].item() - e[0], tgt[j, 1].item() - e[1])
              for j, e in enumerate(exact(p[0], p[1], steps, 6.0, -1.0) for p in pts[0].tolist()))
    check("landing points vs exact turning motion (6 frames, changing v and omega)", err < 0.01, f"max error {err:.4f} m")

    tp.set_velocity((0.0, 0.0), static_weight=0.0)
    xs = spike(c=2, B=3)
    per = [[(0.1, 10.0, 0.0)], [(0.2, 5.0, 0.3)], [(0.05, -4.0, -0.2), (0.05, -4.0, -0.2)]]
    rb = tp(xs, per)
    alone = torch.cat([tp(xs[b:b + 1], per[b])["feat"] for b in range(3)])
    check("batch with different ego motion per sample = each sample alone", torch.allclose(rb["feat"], alone, atol=1e-10))
    st = steps_tensor(per, 3, dtype=D64)
    check("padded step lists (lengths 1, 1, 2): padding is the identity step", st.shape == (3, 2, 3) and st[0, 1].abs().sum() == 0)

    xm = torch.randn(1, 2, G.nx, G.ny, dtype=D64)
    rg = tp(xm, [(0.2, 5.0, 0.3)])
    from splat import grid_points
    rpm = tp.forward_points(xm.flatten(2), grid_points(G, dtype=D64)[None], [(0.2, 5.0, 0.3)])
    check("point mode at the cell centres = grid mode", torch.allclose(rg["feat"], rpm["feat"], atol=1e-10))

    tm = TemporalPropagation(cin=1, grid=G, n_hyp=1, static_hyp=True).double()
    tm.set_velocity((4.0, 0.0), static_weight=0.0)                          # logits equal -> pi = 0.5 / 0.5
    r = tm(spike(), [(0.5, 0.0, 0.0)])
    f = r["feat"][0, 0]                       # every cell is a source, so the mass is 1; the spike's feature splits
    check("two hypotheses with equal score split the spike 0.5 / 0.5", abs(f[20, 20].item() - 0.5) < 1e-9 and abs(f[24, 20].item() - 0.5) < 1e-9)

    td = TemporalPropagation(cin=1, grid=G, n_hyp=4, static_hyp=True, init_speed=6.0, v_alias=10.0, doppler_sigma=0.3).double()
    v_hyp, _ = td.hypotheses(torch.zeros(1, 1, 1, dtype=D64))
    true = v_hyp[0, 2, :, 0]                                                # one of the fan's velocities is the truth
    ang = torch.tensor([0.3, 1.2], dtype=D64)
    u = torch.stack([torch.cos(ang), torch.sin(ang)], 0).view(1, 2, 2, 1).permute(0, 2, 1, 3)   # (1, J=2, 2, 1)
    d = (true.view(1, 1, 2, 1) * u).sum(2) + 20.0                           # + 2 v_alias: aliased measurement
    dop = {"u": u, "d": d, "valid": torch.ones(1, 2, 1, dtype=D64)}
    r = td.forward_points(torch.zeros(1, 1, 1, dtype=D64), torch.tensor([[[10.0, 0.0]]], dtype=D64), [(0.1, 0.0, 0.0)], dop)
    check("Doppler evidence selects the right hypothesis (through aliasing)", r["pi"][0, :, 0].argmax().item() == 2,
          "pi " + " ".join(f"{p:.2f}" for p in r["pi"][0, :, 0].tolist()))

    tp.set_velocity((3.0, 1.0))
    a = tp(spike(c=2), [(0.5, 6.0, 0.2)])["feat"]
    tp2 = TemporalPropagation(cin=1, grid=G, n_hyp=2).double()
    tp2.set_velocity((3.0, 1.0))
    pts = tp.forward_points(torch.ones(1, 2, 1, dtype=D64), torch.tensor([[[10.25, 0.25]]], dtype=D64), [(0.5, 6.0, 0.2)])["target"][:, 1:2]
    back = TemporalPropagation(cin=1, grid=G, n_hyp=2).double()
    back.set_velocity((3.0, 1.0))
    # forward then backward with the velocity rotated into the target axes returns to the start
    th = 0.2 * 0.5
    vb = (math.cos(th) * 3.0 + math.sin(th) * 1.0, -math.sin(th) * 3.0 + math.cos(th) * 1.0)
    back.set_velocity(vb)
    ret = back.forward_points(torch.ones(1, 1, 1, dtype=D64), pts, [(-0.5, 6.0, 0.2)])["target"][0, 1]
    check("forward then backward in time returns to the start", (ret - torch.tensor([10.25, 0.25], dtype=D64)).norm() < 0.02,
          f"error {(ret - torch.tensor([10.25, 0.25], dtype=D64)).norm().item():.4f} m")

    x32 = torch.randn(1, 2, G.nx, G.ny)
    t32 = TemporalPropagation(cin=2, grid=G, n_hyp=2)
    t32.set_velocity((5.0, 2.0))
    t64 = TemporalPropagation(cin=2, grid=G, n_hyp=2).double()
    t64.set_velocity((5.0, 2.0))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    o32 = t32.to(dev)(x32.to(dev), [(0.3, 7.0, 0.1)])["feat"].cpu().double()
    o64 = t64(x32.double(), [(0.3, 7.0, 0.1)])["feat"]
    check(f"float32 on {dev} = float64 on cpu", torch.allclose(o32, o64, atol=1e-4), f"max diff {(o32 - o64).abs().max().item():.2e}")

    z = tp(torch.zeros(1, 2, G.nx, G.ny, dtype=D64), [(1.0, 30.0, 1.0)])
    far = tp.forward_points(torch.ones(1, 2, 3, dtype=D64), torch.tensor([[[500.0, 0.0], [-50.0, 3.0], [5.0, 900.0]]], dtype=D64), [(0.1, 1.0, 0.0)])
    check("empty map and sources far outside stay finite (mass 0)", bool(torch.isfinite(z["feat"]).all()) and far["mass"].sum().item() == 0)

    gs = Grid(x_min=0, x_max=4, y_min=-2, y_max=2, res=0.5)
    f = torch.randn(1, 2, 5, dtype=D64, requires_grad=True)
    p = (torch.rand(1, 5, 2, dtype=D64) * 2 + torch.tensor([1.0, -1.0], dtype=D64)).requires_grad_(True)
    w = torch.rand(1, 5, dtype=D64, requires_grad=True)
    ok = torch.autograd.gradcheck(lambda a, b, c: gaussian_splat(a, b, c, gs, 0.0, 2)[0], (f, p, w), eps=1e-6, atol=1e-5)
    check("gradcheck of the bilinear splat (features, positions, weights)", ok)
    check("aliasing wrap", abs(wrap(torch.tensor(17.0), 10.0).item() + 3.0) < 1e-6)

    from splat import grid_points as gp
    check("sample_at at the cell centres = the map", torch.allclose(sample_at(x, gp(G, dtype=D64)[None], G).view_as(x), x, atol=1e-9))
    tu = TemporalPropagation(cin=2, grid=G, n_hyp=1).double()
    tu.set_velocity((3.0, -2.0))
    vu = torch.tensor([3.0, -2.0], dtype=D64).view(1, 2, 1, 1).expand(1, 2, G.nx, G.ny)
    check("propagate (one v per cell) = TemporalPropagation with that v",
          torch.allclose(propagate(x, [(0.3, 5.0, 0.2)], vu, G)[0], tu(x, [(0.3, 5.0, 0.2)])["feat"], atol=1e-9))
    fr = FlowRefine(2, G, radius=3, proj=2).double()
    with torch.no_grad():
        for conv in (fr.f, fr.g):
            conv.weight.copy_(torch.eye(2, dtype=D64).view(2, 2, 1, 1))
            conv.bias.zero_()
    cur = torch.zeros_like(spike(c=2))
    cur[0, :, 22, 19] = 1.0                                                  # the spike moved +2, -1 cells
    corr = fr.correlation(fr.f(spike(c=2)), fr.g(cur), landing(steps_tensor([(0.5, 0.0, 0.0)], 1, dtype=D64), torch.zeros(1, 2, G.nx, G.ny, dtype=D64), G))
    k = corr[0, :, 20, 20].argmax().item()
    check("correlation at the source cell peaks at the true displacement", tuple(fr.offs[k].tolist()) == (1.0, -0.5),
          f"peak offset {tuple(fr.offs[k].tolist())} m")
    v0 = torch.randn(1, 2, G.nx, G.ny, dtype=D64)
    vs = fr(spike(c=2), cur, [(0.5, 4.0, 0.1)], v0, iters=4)
    check("FlowRefine starts as the identity (v0 kept, 4 iterations)", len(vs) == 4 and torch.equal(vs[-1], v0))
    for p in fr.net[-1].parameters():
        torch.nn.init.normal_(p, std=0.1)
    fr(x, x.roll(1, 2), [(0.5, 4.0, 0.1)], v0, iters=2)[-1].pow(2).sum().backward()
    check("gradient reaches every FlowRefine parameter", all(p.grad is not None and p.grad.abs().sum() > 0 for p in fr.parameters()))

    fpy = FlowPyramid(2, G, factors=(4, 2, 1), iters=(2, 1, 1)).double()
    vs = fpy(x, x.roll(2, 2), [(0.5, 4.0, 0.1)], v0)
    check("FlowPyramid: 4 refinements (2 at 2 m, 1 at 1 m, 1 at 0.5 m), starts as the identity",
          len(vs) == 4 and all(vk.shape == v0.shape for vk in vs) and torch.allclose(vs[-1], v0, atol=1e-12)
          and fpy.levels[0].grid.res == 2.0, f"coarse window +-{3 * fpy.levels[0].grid.res:.0f} m")
    for lv in fpy.levels:
        for p_ in lv.net[-1].parameters():
            torch.nn.init.normal_(p_, std=0.1)
    sum(vk.pow(2).sum() for vk in fpy(x, x.roll(2, 2), [(0.5, 4.0, 0.1)], v0)).backward()   # a loss on EVERY output
    check("gradient reaches every FlowPyramid parameter (loss on every iteration, as in training)",
          all(p_.grad is not None and p_.grad.abs().sum() > 0 for p_ in fpy.parameters()))
    vp = torch.tensor([[[0.0], [2.0]]], dtype=D64)                         # prior (0, 2) m/s, one cell
    Pp = (torch.eye(2, dtype=D64) * 25.0).view(1, 2, 2, 1)                  # sigma 5 m/s
    ux = torch.tensor([1.0, 0.0], dtype=D64).view(1, 1, 2, 1)               # one ray along x, Doppler 5 m/s
    one = torch.ones(1, 1, 1, dtype=D64)
    v1, P1 = doppler_update(vp, Pp, ux, 5.0 * one, one, sigma_d=0.1)
    check("Doppler update, one ray: radial part from the radar, tangential part and its variance kept",
          abs(v1[0, 0, 0].item() - 5.0) < 0.01 and abs(v1[0, 1, 0].item() - 2.0) < 1e-9 and abs(P1[0, 1, 1, 0].item() - 25.0) < 1e-9
          and P1[0, 0, 0, 0].item() < 0.011, f"v ({v1[0, 0, 0].item():.3f}, {v1[0, 1, 0].item():.3f}), var x {P1[0, 0, 0, 0].item():.4f}, y {P1[0, 1, 1, 0].item():.1f}")
    v2, _ = doppler_update(vp + torch.tensor([[[4.0], [0.0]]], dtype=D64), Pp, ux, (5.0 - 20.0) * one, one, sigma_d=0.1, v_alias=10.0)
    check("Doppler update unwraps an aliased measurement to the alias nearest the prior", abs(v2[0, 0, 0].item() - 5.0) < 0.01,
          f"measured -15, alias 10 -> v_x {v2[0, 0, 0].item():.3f}")
    ang = torch.tensor([0.4, 1.3], dtype=D64)
    u2 = torch.stack([torch.cos(ang), torch.sin(ang)], 1).view(1, 2, 2, 1)
    vt = torch.tensor([6.0, -3.0], dtype=D64)
    v3, _ = doppler_update(torch.zeros(1, 2, 1, dtype=D64), Pp * 1e4, u2, (u2[0, :, :, 0] @ vt).view(1, 2, 1),
                           torch.ones(1, 2, 1, dtype=D64), sigma_d=0.1)
    check("Doppler update, two rays + a vague prior: the full velocity", (v3[0, :, 0] - vt).norm() < 0.01,
          f"v ({v3[0, 0, 0].item():.3f}, {v3[0, 1, 0].item():.3f}) vs (6, -3)")
    v4, P4 = doppler_update(vp, Pp, ux, 5.0 * one, 0 * one)
    check("Doppler update without valid detections = the prior", torch.allclose(v4, vp) and torch.allclose(P4, Pp))
    gd = (vp.clone().requires_grad_(True), (torch.rand(1, 1, 1, dtype=D64) * 4).requires_grad_(True))
    check("gradcheck of the Doppler update (prior velocity, measurement)",
          torch.autograd.gradcheck(lambda a, dd: doppler_update(a, Pp, ux, dd, one, sigma_d=0.3)[0], gd, eps=1e-6, atol=1e-6))
    fp = torch.randn(1, 2, 5, dtype=D64)
    pp = torch.rand(1, 5, 2, dtype=D64) * 8 + torch.tensor([4.0, -4.0], dtype=D64)
    wp = torch.rand(1, 5, dtype=D64)
    a0 = gaussian_splat(fp, pp, wp, G, 0.0, 2)
    a1 = gaussian_splat(fp, pp, wp, G, 0.0, 2, importance=torch.ones(1, 5, dtype=D64))
    check("importance 1 = the plain splat", torch.equal(a0[0], a1[0]) and torch.equal(a0[1], a1[1]))
    two = torch.tensor([[[1.0, 0.0]]], dtype=D64)                           # features 1 and 0 land on one cell centre
    pos2 = torch.tensor([[[10.25, 0.25], [10.25, 0.25]]], dtype=D64)
    o, mss = gaussian_splat(two, pos2, torch.ones(1, 2, dtype=D64), G, 0.0, 2, importance=torch.tensor([[99.0, 1.0]], dtype=D64))
    check("importance decides who wins a cell (99 : 1 -> 0.99), the mass stays 2", abs(o[0, 0, 20, 20].item() - 0.99) < 1e-9
          and abs(mss[0, 0, 20, 20].item() - 2.0) < 1e-9)
    ti = TemporalPropagation(cin=2, grid=G, n_hyp=2, importance=True).double()
    ti.set_velocity((3.0, -2.0))
    tu2 = TemporalPropagation(cin=2, grid=G, n_hyp=2).double()
    tu2.set_velocity((3.0, -2.0))
    check("TemporalPropagation(importance=True) starts as the plain block",
          torch.allclose(ti(x, [(0.3, 5.0, 0.2)])["feat"], tu2(x, [(0.3, 5.0, 0.2)])["feat"], atol=1e-12))
    gi = (torch.randn(1, 2, 5, dtype=D64, requires_grad=True),
          (torch.rand(1, 5, 2, dtype=D64) * 2 + torch.tensor([1.0, -1.0], dtype=D64)).requires_grad_(True),
          torch.rand(1, 5, dtype=D64, requires_grad=True), (torch.rand(1, 5, dtype=D64) + 0.5).requires_grad_(True))
    ok = torch.autograd.gradcheck(lambda a, b, c, d: gaussian_splat(a, b, c, Grid(x_min=0, x_max=4, y_min=-2, y_max=2, res=0.5), 0.0, 2,
                                                                      importance=d)[0], gi, eps=1e-6, atol=1e-5)
    check("gradcheck of the importance-weighted splat", ok)
    re = tm(spike(), [(0.5, 0.0, 0.0)])
    check("entropy of two equal hypotheses = ln 2", abs(re["entropy"][0, 20, 20].item() - math.log(2)) < 1e-9)
    tm.temperature = 0.1
    with torch.no_grad():
        tm.net[4].bias[2] = 1.0                                              # static logit 1 vs 0
    pt = tm(spike(), [(0.5, 0.0, 0.0)])["pi"][0, :, 20, 20]
    check("temperature 0.1 sharpens pi (logits 1 vs 0 -> 0.99995)", abs(pt[0].item() - 1 / (1 + math.exp(-10))) < 1e-9,
          f"pi {pt.tolist()}")
    tm.temperature = 1.0

    te = TemporalPropagation(cin=2, grid=G, n_hyp=0).double()              # ego only: the static hypothesis alone
    te.set_velocity((0.0, 0.0), static_weight=0.0)
    check("n_hyp = 0 (ego only) = the 2-cell shift", torch.allclose(te(x, [(0.1, 10.0, 0.0)])["feat"][:, :, :-2], x[:, :, 2:], atol=1e-9))

    gf = GapFill(2).double()
    tp.set_velocity((0.0, 0.0), static_weight=0.0)
    r = tp(x, [(0.5, 8.0, 0.2)])                                            # 4 m forward: a hole at the far edge
    check("GapFill starts as the identity", torch.equal(gf(r["feat"], r["mass"]), r["feat"]))
    for p in gf.parameters():
        torch.nn.init.normal_(p, std=0.5)
    out = gf(r["feat"], r["mass"])
    known = (r["mass"] >= 1).expand_as(out)
    check("GapFill (random weights) never changes a known cell, fills the holes",
          torch.equal(out[known], r["feat"][known]) and (out[~known] - r["feat"][~known]).abs().max() > 0,
          f"{int((r['mass'] < 1).sum())} of {G.nx * G.ny} cells with mass < 1")


# ------------------------------------------------------------------------------------------------ end to end
VA, VB = (8.0, 2.0), (-2.0, -6.0)            # object velocities (world, source axes); far from any initial guess


def world(rng):
    """source map (3, X, Y): type A, type B, static stripes; the objects as lists of positions; the ego steps."""
    v, w = rng.uniform(0, 8), rng.uniform(-0.3, 0.3)
    steps = [(0.5 / 6, v, w)] * 6
    a = [(rng.uniform(4, 16), rng.uniform(-7, 7)) for _ in range(rng.randint(1, 3))]
    b = [(rng.uniform(4, 16), rng.uniform(-7, 7)) for _ in range(rng.randint(1, 3))]
    lanes = [rng.uniform(-8, -3), rng.uniform(3, 8)]
    return a, b, lanes, steps


def render(a, b, lanes, steps, at_target):
    xs, ys = G.xy_centres(dtype=D64)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    out = torch.zeros(3, G.nx, G.ny, dtype=D64)
    for ch, objs, vel in ((0, a, VA), (1, b, VB)):
        for (px, py) in objs:
            cx, cy = exact(px, py, steps, *vel) if at_target else (px, py)
            out[ch] = torch.maximum(out[ch], torch.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * 0.6 ** 2)))
    for yl in lanes:                                                     # static stripes y = yl in the source frame
        for x0 in torch.arange(1.0, 19.0, 0.25).tolist():
            cx, cy = exact(x0, yl, steps, 0.0, 0.0) if at_target else (x0, yl)
            ix, iy = int((cx - G.x_min) / G.res), int((cy - G.y_min) / G.res)
            if 0 <= ix < G.nx and 0 <= iy < G.ny:
                out[2, ix, iy] = 1.0
    return out.float()


def e2e(name, n_hyp, static_hyp, sigma_start, doppler, iters, init_speed=5.0, seed=0):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tp = TemporalPropagation(cin=3, grid=G, n_hyp=n_hyp, static_hyp=static_hyp, init_speed=init_speed,
                             sigma0=sigma_start, radius=3 if sigma_start > 0 else 2, v_alias=None, doppler_sigma=0.3)
    if n_hyp == 1 and not static_hyp:
        tp.set_velocity((0.0, 0.0))                                      # the plain splat starts at zero velocity
        torch.nn.init.normal_(tp.net[4].weight, std=1e-3)
    opt = torch.optim.Adam(tp.parameters(), lr=3e-3)
    xs, ys = G.xy_centres()
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    for it in range(iters):
        if sigma_start > 0:
            tp.sigma0 = sigma_start * max(0.0, 1 - it / (0.7 * iters))     # anneal to the bilinear kernel
        batch = [world(rng) for _ in range(4)]
        src = torch.stack([render(*w_, at_target=False) for w_ in batch])
        tgt = torch.stack([render(*w_, at_target=True) for w_ in batch])
        dop = None
        if doppler:
            r = torch.sqrt(gx ** 2 + gy ** 2).clamp(min=1e-3)
            u = torch.stack([gx / r, gy / r])                                      # rays from a radar at the origin
            vel = src[:, 0:1] * torch.tensor(VA).view(1, 2, 1, 1) + src[:, 1:2] * torch.tensor(VB).view(1, 2, 1, 1)
            occ = (src[:, :2].sum(1) > 0.3).float()
            dd = (vel * u).sum(1) + 0.1 * torch.randn_like(occ)
            dop = {"u": u.view(1, 1, 2, G.nx, G.ny).expand(4, 1, 2, G.nx, G.ny), "d": dd.unsqueeze(1), "valid": occ.unsqueeze(1)}
        out = tp(src, [w_[3] for w_ in batch], dop)
        loss = ((out["feat"] - tgt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        rng = random.Random(1234)
        errs = {"A": [], "B": []}
        for _ in range(8):
            w_ = world(rng)
            src = render(*w_, at_target=False)[None]
            dop = None
            if doppler:
                r = torch.sqrt(gx ** 2 + gy ** 2).clamp(min=1e-3)
                u = torch.stack([gx / r, gy / r])
                vel = src[:, 0:1] * torch.tensor(VA).view(1, 2, 1, 1) + src[:, 1:2] * torch.tensor(VB).view(1, 2, 1, 1)
                occ = (src[:, :2].sum(1) > 0.3).float()
                dop = {"u": u.view(1, 1, 2, G.nx, G.ny), "d": ((vel * u).sum(1)).unsqueeze(1), "valid": occ.unsqueeze(1)}
            out = tp(src, [w_[3]], dop)
            for key, ch, vt in (("A", 0, VA), ("B", 1, VB)):
                m = src[0, ch] > 0.6
                if m.any():
                    v = out["v_mean"][0][:, m].mean(1)
                    errs[key].append(math.hypot(v[0].item() - vt[0], v[1].item() - vt[1]))
    ea, eb = sum(errs["A"]) / len(errs["A"]), sum(errs["B"]) / len(errs["B"])
    print(f"  {name:52s} velocity error A {ea:5.2f} m/s, B {eb:5.2f} m/s   (final loss {loss.item():.4f})", flush=True)
    return ea, eb


if __name__ == "__main__":
    print("== smoke tests")
    smoke()
    print(f"== {sum(RESULTS)} / {len(RESULTS)} smoke tests pass")
    iters = 150 if QUICK else 600
    print(f"== end-to-end speed learning without velocity labels ({iters} steps of 4 scenes; truth A = {VA}, B = {VB} m/s)")
    res = [e2e("(1) one hypothesis, bilinear, starts at 0 (plain)", 1, False, 0.0, False, iters),
           e2e("(2) one hypothesis, Gaussian 2 cells annealed to bilinear", 1, False, 2.0, False, iters),
           e2e("(3) four hypotheses + static, bilinear", 4, True, 0.0, False, iters),
           e2e("(4) four hypotheses + static + Doppler evidence", 4, True, 0.0, True, iters)]
