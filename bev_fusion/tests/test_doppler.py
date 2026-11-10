"""Gate: what does radar Doppler tell about an object's 2D velocity? (plan.md section 10.1, Doppler consistency)

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
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_doppler.py
"""
import math
import random
import sys

import torch

sys.path.insert(0, "tests/support")
from synthetic_world import radar, random_scene  # noqa: E402

RADARS_1 = [(3.6, 0.0, 0.0)]
RADARS_3 = [(3.6, 0.0, 0.0), (3.4, 0.6, math.radians(80)), (3.4, -0.6, math.radians(-80))]
NOISE = dict(sigma_r=0.1, sigma_phi_deg=0.5, sigma_d=0.1)
BANDS = ((0, 20), (20, 40), (40, 60))


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


def case(name, n_scenes, radars, sweeps, rng):
    errs = {band: ([], [], []) for band in BANDS}                    # radial, tangential, condition
    for _ in range(n_scenes):
        scene = random_scene(rng)
        rows = {}
        # sweeps oldest first: scene at -(sweeps-1) dt ... 0; returns expressed in the current (last) frame
        dt = 0.5 / max(1, sweeps - 1) if sweeps > 1 else 0.0
        past = scene
        frames = [past]
        for _ in range(sweeps - 1):
            past = past.step(dt)
            frames.append(past)
        current = frames[-1]
        for j, sc in enumerate(frames):
            th_to_now = sc.omega * dt * (len(frames) - 1 - j)              # ego yaw between this sweep and now
            for k, u, d in returns(sc, radars, rng):
                rows.setdefault(k, []).append((rotate(u, th_to_now), d))
        for k, rs in rows.items():
            b = current.boxes[k]
            if math.hypot(b.vx, b.vy) < 0.5 or len(rs) < 2:
                continue
            v_hat, cond = solve(rs)
            mu = torch.tensor([sum(u[0] for u, _ in rs), sum(u[1] for u, _ in rs)], dtype=torch.float64)
            mu = mu / mu.norm()
            e = v_hat - torch.tensor([b.vx, b.vy], dtype=torch.float64)
            rng_box = math.hypot(b.cx, b.cy)
            for lo, hi in BANDS:
                if lo <= rng_box < hi:
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
    sys.exit(0 if ok else 1)
