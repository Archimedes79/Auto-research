"""Gate: what does radar Doppler tell about an object's 2D velocity? (plan.md section 10.1, Doppler consistency)

PART 1 - Doppler alone.
A return measures only the radial component d = v . u (u the unit line of sight from its sensor). Several returns of
the same object with different u are combined in the information form A = sum u u^T, b = sum d u (the A, b channels of
the radar encoder); where A is well conditioned, v_hat = A^-1 b gives the full 2D velocity.
Synthetic world (tests/support/synthetic_world.py): moving boxes, detections on their visible edges, noise: range 0.1 m,
azimuth 0.5 deg, Doppler 0.1 m/s (assumed values, not checked against a sensor data sheet). Per moving box with >= 2
returns: v_hat by least squares (with a tiny ridge), its error split into the radial component (along the mean line of
sight) and the tangential one, by range band. Cases: 1 radar 1 sweep; 3 radars (front, front-left, front-right as
mounted in nuScenes, roughly) 1 sweep; 1 radar 6 sweeps over 0.5 s (ego and object move, so u changes; older returns
expressed in the current frame, the object velocity assumed constant in the world).
Known answer: the radial error must be at the noise level in every case; the tangential error shows what Doppler
alone cannot give - the part the two-frame flow (studies/early/flow_from_frames.py) must supply.

PART 2 - a SOFT heading term and a flow term in the same information form (2026-11-10).
Every source is a Gaussian constraint on v, a 2x2 matrix and a 2-vector, in units of 1 / sigma_d^2 (sigma_d = 0.1 m/s):
    Doppler   A = sum u u^T,  b = sum d u                              (exact along the line of sight)
    heading   w_h h_perp h_perp^T,  0                                  (the velocity ACROSS the heading axis is ~ 0 +- sigma_h)
    flow      w_f I,  w_f v_flow                                       (both components, noisy: sigma_f = 1.5 m/s, ASSUMED, a stand-in
                                                                        for the camera flow of the motion block)
    v = (sum Lambda + ridge)^-1 sum eta   (a 2x2 solve: differentiable, no hypotheses, no argmax)
The heading enters only as an AXIS (h_perp h_perp^T does not change under h -> -h): reversing and front / back confusion are
allowed, the sign comes from the Doppler. It is a weak constraint, never a hard one: sigma_h = max(kappa |v_flow|, 0.1 m/s)
with kappa the ML scale of the across-heading velocity of the data (the stand-in for what training learns; in-sample, one
scalar), kappa -> large switches the term off.
The world sets every box velocity exactly along its heading (no slip, no curves), so the angle between the PREDICTED heading
axis and the true velocity direction is drawn explicitly: heading error sigma + slip sigma, combined in quadrature. Assumed
values: heading error 10 / 20 deg (a stand-in for the object head's yaw error, NOT measured - the number that decides the
approach); slip 2.4 deg straight and 5.4 deg in curves (sigma of the angle between velocity and heading axis of cars / trucks /
buses on the nuScenes mini annotations, p90 4.0 / 8.9 deg; one-off measurement of 2026-11-10: velocity = finite difference of the
same instance between neighbouring keyframes, moving = > 1 m/s, turning = yaw rate >= 8 deg/s; annotation headings are probably
too good). 30 % of the headings are flipped by 180 deg. The curve case of 6 sweeps also turns the boxes (yaw rate
sd 0.25 rad/s), so the accumulated Doppler of the old sweeps is stale and only the heading of NOW is current.
The tables report median / p90 of the tangential velocity error (m/s), 1 radar; the rules were written before the first run
(below, thresholds not changed afterwards):
    G2a  exact heading, 1 sweep:   D+H tangential median <= 0.5 x D
    G2b  every condition, 1 sweep, and the 6-sweep curve case:   D+H tangential median <= 1.1 x D (a calibrated soft term never
         makes it worse; with a useless heading it must switch itself off)
    G2c  the fused estimate is unchanged under h -> -h (max difference < 1e-9)
    G2d  straight condition, every crossing-angle bin:   D+H+F <= 1.05 x D+F (adding information does not hurt)
    G2e  radial median <= 0.3 m/s for every variant of the 1-sweep tables (the soft terms must not spoil what the Doppler measures)
    G2f  ADDED AFTER THE FIRST RUN (G2b is weak: D alone has only a 1e-6 ridge, so ANY soft term shrinks its heavy tail, even a random
         heading): every condition, 1 sweep: D+H+F <= 1.05 x D+F (on top of the flow, which the system has anyway, the heading never hurts)
    Read the tables against D+F, not against D: the gain of the heading is D+F -> D+H+F. The exact-heading row is an ideal (the world
    puts every velocity exactly along its heading), the others depend on the ASSUMED heading error.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_doppler.py
"""
import math
import random
import sys
from dataclasses import replace

import torch

sys.path.insert(0, "tests/support")
from synthetic_world import radar, random_scene  # noqa: E402

RADARS_1 = [(3.6, 0.0, 0.0)]
RADARS_3 = [(3.6, 0.0, 0.0), (3.4, 0.6, math.radians(80)), (3.4, -0.6, math.radians(-80))]
NOISE = dict(sigma_r=0.1, sigma_phi_deg=0.5, sigma_d=0.1)
BANDS = ((0, 20), (20, 40), (40, 60))
SIGMA_F = 1.5                                       # m/s, ASSUMED noise of the flow velocity (per component)
SIGMA_FLOOR = 0.1                                   # m/s, a heading line is never trusted more than one Doppler return
SLIP_STRAIGHT, SLIP_CURVE = 2.4, 5.4                # deg, sigma of (velocity direction - heading axis), nuScenes mini annotations
# name, sigma of the angle between the predicted heading axis and the true velocity direction (deg; None = a random heading)
CONDITIONS = (("exact heading", 0.0),
              ("straight: heading 10 + slip 2.4 deg", math.hypot(10.0, SLIP_STRAIGHT)),
              ("curve: heading 10 + slip 5.4 deg", math.hypot(10.0, SLIP_CURVE)),
              ("curve, poor heading: 20 + 5.4 deg", math.hypot(20.0, SLIP_CURVE)),
              ("heading useless (random)", None))
STRAIGHT, CURVE = CONDITIONS[1], CONDITIONS[2]


def returns(scene, radars, rng):
    """list of (box, u (2,), doppler) of all detections of all radars on boxes, current frame."""
    out = []
    for pose in radars:
        det = radar(scene, pose=pose, points_per_box=3, rng=rng, **NOISE)
        for i in range(len(det["range"])):
            k = det["box"][i].item()
            if k < 0:
                continue
            x, y = det["xy"][i].tolist()
            dx, dy = x - pose[0], y - pose[1]
            n = math.hypot(dx, dy)
            out.append((k, (dx / n, dy / n), det["doppler"][i].item()))
    return out


def rotate(u, th):
    c, s = math.cos(th), math.sin(th)
    return (c * u[0] + s * u[1], -s * u[0] + c * u[1])


def solve(rows, ridge=1e-6):
    A = torch.zeros(2, 2, dtype=torch.float64)
    b = torch.zeros(2, dtype=torch.float64)
    for (ux, uy), d in rows:
        u = torch.tensor([ux, uy], dtype=torch.float64)
        A += torch.outer(u, u)
        b += d * u
    ev = torch.linalg.eigvalsh(A)
    return torch.linalg.solve(A + ridge * torch.eye(2, dtype=torch.float64), b), (ev[-1] / ev[0].clamp(min=1e-12)).item()


def step_turning(scene, dt, rates):
    """scene.step(dt) with every box also turning by its own yaw rate (rad/s): its velocity and heading rotate."""
    nxt = scene.step(dt)
    boxes = []
    for b, w in zip(nxt.boxes, rates):
        c, s = math.cos(w * dt), math.sin(w * dt)
        boxes.append(replace(b, vx=c * b.vx - s * b.vy, vy=s * b.vx + c * b.vy, heading=b.heading + w * dt))
    return replace(nxt, boxes=boxes)


def collect(n_scenes, radars, sweeps, rng, turn_sd=0.0):
    """Moving boxes (>= 0.5 m/s) with at least one return -> list of dicts: rows [(u, d)] expressed in the current frame, v = the
    true velocity now, range (m), mu = the mean line of sight (unit). Sweeps oldest first, returns of older sweeps rotated into
    the current frame; the boxes turn with a yaw rate ~ N(0, turn_sd) when turn_sd > 0."""
    objs = []
    for _ in range(n_scenes):
        scene = random_scene(rng)
        rates = [rng.gauss(0.0, turn_sd) for _ in scene.boxes] if turn_sd > 0 else None
        dt = 0.5 / max(1, sweeps - 1) if sweeps > 1 else 0.0
        frames = [scene]
        for _ in range(sweeps - 1):
            frames.append(step_turning(frames[-1], dt, rates) if rates else frames[-1].step(dt))
        current = frames[-1]
        rows = {}
        for j, sc in enumerate(frames):
            th_to_now = sc.omega * dt * (len(frames) - 1 - j)              # ego yaw between this sweep and now
            for k, u, d in returns(sc, radars, rng):
                rows.setdefault(k, []).append((rotate(u, th_to_now), d))
        for k, rs in rows.items():
            b = current.boxes[k]
            if math.hypot(b.vx, b.vy) < 0.5:
                continue
            mu = torch.tensor([sum(u[0] for u, _ in rs), sum(u[1] for u, _ in rs)], dtype=torch.float64)
            mu = mu / mu.norm()
            objs.append(dict(rows=rs, v=(b.vx, b.vy), range=math.hypot(b.cx, b.cy), mu=mu.tolist()))
    return objs


def case(name, n_scenes, radars, sweeps, rng):
    errs = {band: ([], [], []) for band in BANDS}                    # radial, tangential, condition
    for o in collect(n_scenes, radars, sweeps, rng):
        if len(o["rows"]) < 2:
            continue
        v_hat, cond = solve(o["rows"])
        mu = torch.tensor(o["mu"], dtype=torch.float64)
        e = v_hat - torch.tensor(o["v"], dtype=torch.float64)
        for lo, hi in BANDS:
            if lo <= o["range"] < hi:
                errs[(lo, hi)][0].append(abs((e @ mu).item()))
                errs[(lo, hi)][1].append(abs(e[0] * mu[1] - e[1] * mu[0]).item())
                errs[(lo, hi)][2].append(cond)
    med = lambda a: sorted(a)[len(a) // 2] if a else float("nan")   # noqa: E731
    cells = []
    for band in BANDS:
        r, t = errs[band][:2]
        cells.append(f"{med(r):5.2f} / {med(t):6.2f} (n={len(r)})")
    print(f"{name:28s} | " + " | ".join(cells))
    return errs


# ----------------------------------------------------------------------------------------------- part 2
def fuse(rows, axis=None, w_h=0.0, v_flow=None, w_f=0.0, ridge=1e-6):
    """information form in units of 1 / sigma_d^2: Doppler A = sum u u^T, b = sum d u; + w_h h_perp h_perp^T (soft heading axis,
    angle `axis`); + w_f (I, v_flow) (flow). Closed 2x2 solve -> (vx, vy)."""
    axx = axy = ayy = bx = by = 0.0
    for (ux, uy), d in rows:
        axx += ux * ux
        axy += ux * uy
        ayy += uy * uy
        bx += d * ux
        by += d * uy
    if axis is not None and w_h > 0:
        hx, hy = -math.sin(axis), math.cos(axis)                      # across the axis
        axx += w_h * hx * hx
        axy += w_h * hx * hy
        ayy += w_h * hy * hy
    if v_flow is not None and w_f > 0:
        axx += w_f
        ayy += w_f
        bx += w_f * v_flow[0]
        by += w_f * v_flow[1]
    axx += ridge
    ayy += ridge
    det = axx * ayy - axy * axy
    return ((ayy * bx - axy * by) / det, (axx * by - axy * bx) / det)


def prepare(objs, seed=11):
    """the random draws every condition shares: flow noise, heading-error z, flip, a random angle (so the conditions are nested)."""
    r = random.Random(seed)
    for o in objs:
        o["flow"] = (o["v"][0] + r.gauss(0.0, SIGMA_F), o["v"][1] + r.gauss(0.0, SIGMA_F))
        o["z"], o["flip"], o["uni"] = r.gauss(0.0, 1.0), r.random() < 0.3, r.uniform(-math.pi, math.pi)


def axis_of(o, sigma_deg):
    """the predicted heading axis: the true velocity direction + an error of sigma_deg (None: a random angle); 30 % flipped by pi."""
    ang = o["uni"] if sigma_deg is None else math.atan2(o["v"][1], o["v"][0]) + math.radians(sigma_deg) * o["z"]
    return ang + (math.pi if o["flip"] else 0.0)


def speed_scale(o):
    return max(math.hypot(*o["flow"]), 1.0)                          # the speed the network can see without the heading: the flow


def calibrate(objs, sigma_deg):
    """kappa = rms over the objects of (true velocity across the predicted heading axis) / (flow speed): the ML scale of the
    Gaussian across-heading term (in-sample, one scalar: what training would learn)."""
    acc = []
    for o in objs:
        ax = axis_of(o, sigma_deg)
        q = -math.sin(ax) * o["v"][0] + math.cos(ax) * o["v"][1]
        acc.append((q / speed_scale(o)) ** 2)
    return math.sqrt(sum(acc) / len(acc))


def variants(o, ax, kappa):
    w_h = (NOISE["sigma_d"] / max(kappa * speed_scale(o), SIGMA_FLOOR)) ** 2
    w_f = (NOISE["sigma_d"] / SIGMA_F) ** 2
    return {"D": fuse(o["rows"]), "D+H": fuse(o["rows"], ax, w_h), "D+F": fuse(o["rows"], None, 0.0, o["flow"], w_f),
            "D+H+F": fuse(o["rows"], ax, w_h, o["flow"], w_f)}


def errors(o, v):
    """(radial, tangential) |error| of v against the true velocity, split along / across the mean line of sight."""
    ex, ey = v[0] - o["v"][0], v[1] - o["v"][1]
    mx, my = o["mu"]
    return abs(ex * mx + ey * my), abs(ex * my - ey * mx)


def evaluate(objs, sigma_deg):
    """-> {variant: [(radial, tangential) per object]}, kappa"""
    kappa = calibrate(objs, sigma_deg)
    out = {k: [] for k in ("D", "D+H", "D+F", "D+H+F")}
    for o in objs:
        for k, v in variants(o, axis_of(o, sigma_deg), kappa).items():
            out[k].append(errors(o, v))
    return out, kappa


def median(a):
    return sorted(a)[len(a) // 2] if a else float("nan")


def pct(a, q):
    s = sorted(a)
    return s[min(len(s) - 1, int(q * len(s)))] if s else float("nan")


def cell(pairs, col=1):
    v = [p[col] for p in pairs]
    return f"{median(v):5.2f}/{pct(v, 0.9):5.2f}"


NAMES = ("D", "D+H", "D+F", "D+H+F")


def header(first):
    print(f"{first:42s} | " + " | ".join(f"{n:>11s}" for n in NAMES))


def part2():
    fails, gates = [], {}
    # ---- 1 radar, 1 sweep, objects with >= 2 returns (the cases of part 1) and with exactly 1 return (D alone is singular there)
    objs = collect(300, RADARS_1, 1, random.Random(5))
    prepare(objs)
    multi = [o for o in objs if len(o["rows"]) >= 2]
    single = [dict(o, rows=[o["rows"][0]], mu=list(o["rows"][0][0])) for o in multi]     # the same boxes cut to their first return (a sparse radar)
    print(f"\nPART 2: 1 radar, 1 sweep, {len(multi)} moving boxes with >= 2 returns; "
          "tangential error median / p90 (m/s); D Doppler, H soft heading axis, F flow (sigma 1.5 m/s, assumed)")
    header("objects with >= 2 returns")
    res = {}
    for name, sig in CONDITIONS:
        r, kappa = evaluate(multi, sig)
        res[name] = r
        print(f"{name:42s} | " + " | ".join(f"{cell(r[n]):>11s}" for n in NAMES) + f"   kappa {kappa:.3f}")
        gates[("rad", name)] = max(median([p[0] for p in r[n]]) for n in NAMES)
        gates[("tan", name)] = (median([p[1] for p in r["D"]]), median([p[1] for p in r["D+H"]]))
        gates[("hf", name)] = (median([p[1] for p in r["D+H+F"]]), median([p[1] for p in r["D+F"]]))
    # ---- by crossing angle: c = |cos(angle between the line of sight and the true motion)|, small = across the line of sight
    print("\nby crossing angle c = |cos(line of sight, motion)|, condition " + STRAIGHT[0])
    header("bin")
    bins = ((0.0, 0.3, "c < 0.3 (across the line of sight)"), (0.3, 0.7, "0.3 <= c < 0.7"), (0.7, 1.01, "c >= 0.7 (along the line of sight)"))
    cs = [abs(o["mu"][0] * o["v"][0] + o["mu"][1] * o["v"][1]) / math.hypot(*o["v"]) for o in multi]
    for lo, hi, label in bins:
        idx = [i for i, c in enumerate(cs) if lo <= c < hi]
        sub = {n: [res[STRAIGHT[0]][n][i] for i in idx] for n in NAMES}
        print(f"{label + f' (n={len(idx)})':42s} | " + " | ".join(f"{cell(sub[n]):>11s}" for n in NAMES))
        gates[("bin", lo)] = (median([p[1] for p in sub["D+H+F"]]), median([p[1] for p in sub["D+F"]]))
    # ---- objects with a single return
    print(f"\nthe same {len(single)} boxes cut to ONE return (a sparse radar: the Doppler alone gives the radial component only, ridge solution)")
    header("condition")
    for name, sig in (CONDITIONS[0], STRAIGHT, CURVE):
        r, kappa = evaluate(single, sig)
        print(f"{name:42s} | " + " | ".join(f"{cell(r[n]):>11s}" for n in NAMES) + f"   kappa {kappa:.3f}")
    # ---- invariance under the heading flip (reversing)
    worst = 0.0
    for o in multi:
        ax = axis_of(o, STRAIGHT[1])
        a = fuse(o["rows"], ax, 0.02, o["flow"], 0.004)
        b = fuse(o["rows"], ax + math.pi, 0.02, o["flow"], 0.004)
        worst = max(worst, abs(a[0] - b[0]), abs(a[1] - b[1]))
    gates["flip"] = worst
    # ---- 1 radar, 6 sweeps, the boxes turning
    turning = collect(150, RADARS_1, 6, random.Random(5), turn_sd=0.25)
    prepare(turning)
    t2 = [o for o in turning if len(o["rows"]) >= 2]
    print(f"\n1 radar, 6 sweeps (0.5 s), boxes turning (yaw rate sd 0.25 rad/s), {len(t2)} boxes: the Doppler of the old sweeps is stale")
    header("condition (heading of NOW)")
    for name, sig in (CONDITIONS[0], CURVE):
        r, kappa = evaluate(t2, sig)
        print(f"{name:42s} | " + " | ".join(f"{cell(r[n]):>11s}" for n in NAMES) + f"   kappa {kappa:.3f}")
        gates[("turn", name)] = (median([p[1] for p in r["D"]]), median([p[1] for p in r["D+H"]]))

    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)
    print("\nrules of part 2 (G2a-G2e written before the first run, G2f added after it):")
    d, h = gates[("tan", CONDITIONS[0][0])]
    check(h <= 0.5 * d, f"G2a exact heading, 1 sweep: D+H tangential median {h:.2f} <= 0.5 x D {d:.2f}")
    for name, _ in CONDITIONS:
        d, h = gates[("tan", name)]
        check(h <= 1.1 * d, f"G2b {name}: D+H {h:.2f} <= 1.1 x D {d:.2f}")
    for k, (d, h) in [(k, v) for k, v in gates.items() if k[0] == "turn"]:
        check(h <= 1.1 * d, f"G2b 6 sweeps turning, {k[1]}: D+H {h:.2f} <= 1.1 x D {d:.2f}")
    check(gates["flip"] < 1e-9, f"G2c fused estimate unchanged under h -> -h (max difference {gates['flip']:.1e})")
    for lo, hi, label in bins:
        hf, f = gates[("bin", lo)]
        check(hf <= 1.05 * f, f"G2d {label}: D+H+F {hf:.2f} <= 1.05 x D+F {f:.2f}")
    for name, _ in CONDITIONS:
        hf, f = gates[("hf", name)]
        check(hf <= 1.05 * f, f"G2f (added after the first run) {name}: D+H+F {hf:.2f} <= 1.05 x D+F {f:.2f}")
    worst_rad = max(v for k, v in gates.items() if k[0] == "rad")
    check(worst_rad <= 0.3, f"G2e radial median of every variant <= 0.3 m/s (worst {worst_rad:.2f})")
    return not fails


if __name__ == "__main__":
    print("median velocity error of moving boxes (m/s): radial / tangential per range band "
          + " | ".join(f"{lo}-{hi} m" for lo, hi in BANDS))
    res = {}
    for name, radars, sweeps in (("1 radar, 1 sweep", RADARS_1, 1), ("3 radars, 1 sweep", RADARS_3, 1),
                                 ("1 radar, 6 sweeps (0.5 s)", RADARS_1, 6)):
        res[name] = case(name, 150, radars, sweeps, random.Random(5))
    med = lambda a: sorted(a)[len(a) // 2]   # noqa: E731
    radial_ok = all(med(e[b][0]) <= 0.3 for e in res.values() for b in BANDS if e[b][0])
    multi = res["1 radar, 6 sweeps (0.5 s)"]
    tang_ok = all(med(multi[b][1]) < 1.0 for b in BANDS if multi[b][1])
    ok = radial_ok and tang_ok
    print(f"GATE Doppler: radial median <= 0.3 m/s in every case and band: {radial_ok}; tangential median < 1 m/s "
          f"with 6 sweeps: {tang_ok} -> {'PASS' if ok else 'FAIL'}")
    ok2 = part2()
    print(f"GATE soft heading + flow (part 2): {'PASS' if ok2 else 'FAIL'}")
    sys.exit(0 if ok and ok2 else 1)
