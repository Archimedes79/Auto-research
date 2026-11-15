"""Simple-first ladder of the temporal block (plan.md 7.3, Mirko 2026-10-26: "build it simple, debug it well, prove it
works in theory"): each stage must measurably improve on the one before, without velocity labels.

World (40 x 40 cells at 0.5 m, x 0..20, y -10..10): objects of two types with fixed world velocities A = (8, 2) and
B = (-2, -6) m/s (Gaussian blobs, channels 0 and 1) and static lane stripes along x (channel 2), hidden under the
objects; random ego motion (speed 0..8 m/s, yaw rate +-0.3 rad/s, 6 steps over 0.5 s). The source map is the world at
t - 0.5 s, the target the world at t. Holes = target cells that no source cell reaches under the TRUE motion: the new
field of view at the far edge and the cells an object has left (the stripe under it becomes visible).
Stages (trained on the warp MSE only; held-out scenes):
  0 copy             the source map as the prediction (no motion)
  1 ego              ego motion only (static hypothesis)
  2 ego + fill       1 + GapFill (dilated convs paint the cells with mass < 1)
  3 hyp              four learned velocity hypotheses + static (temporal.py)
  4 hyp + fill       3 + GapFill
Metrics: MSE on all cells, on known cells, in the holes; velocity error of the types A, B (stage 3, 4).
Claims checked (PASS / FAIL, every seed): ego < copy (all); hyp < ego (known); fill < no fill (holes, both for ego
and hyp) and fill does not worsen the known cells (<= +1 %; "known" is the true-motion mask, whose border cells the
fill may paint - the fill never changes a cell with mass >= 1, studies/early/temporal_block.py); hyp velocity error < 1 m/s.
Image: temp/checks/temporal_stages.png (one held-out scene: source, target, the stages, the hole mask).
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python studies/early/temporal_stages.py [--quick]
"""
import math
import random
import sys
import time

import torch

sys.path.insert(0, "training")
from lift_pd import Grid  # noqa: E402
from temporal import GapFill, TemporalPropagation, gaussian_splat  # noqa: E402

QUICK = "--quick" in sys.argv
G = Grid(x_min=0, x_max=20, y_min=-10, y_max=10, res=0.5)
VA, VB = (8.0, 2.0), (-2.0, -6.0)
XS, YS = G.xy_centres()
GX, GY = torch.meshgrid(XS, YS, indexing="ij")
torch.set_num_threads(4)
DEV = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 3e9 else "cpu"


def ego_pose(steps):
    """exact constant-turn-rate ego pose (x, y, heading) and elapsed time after the steps."""
    ex, ey, eh, t = 0.0, 0.0, 0.0, 0.0
    for dt, v, w in steps:
        th = w * dt
        lx, ly = (v * dt, 0.0) if abs(w) < 1e-12 else (v / w * math.sin(th), v / w * (1 - math.cos(th)))
        ex, ey = ex + math.cos(eh) * lx - math.sin(eh) * ly, ey + math.sin(eh) * lx + math.cos(eh) * ly
        eh += th
        t += dt
    return ex, ey, eh, t


def move(px, py, pose, vx, vy):
    """source-frame points with world velocity -> target-frame points (tensors or floats)."""
    ex, ey, eh, t = pose
    qx, qy = px + vx * t - ex, py + vy * t - ey
    return math.cos(eh) * qx + math.sin(eh) * qy, -math.sin(eh) * qx + math.cos(eh) * qy


def world(rng):
    steps = [(0.5 / 6, rng.uniform(0, 8), rng.uniform(-0.3, 0.3))] * 6
    a = [(rng.uniform(4, 16), rng.uniform(-7, 7)) for _ in range(rng.randint(1, 3))]
    b = [(rng.uniform(4, 16), rng.uniform(-7, 7)) for _ in range(rng.randint(1, 3))]
    lanes = [s * rng.uniform(1.5, 8.5) for s in (-1, 1)] + ([rng.uniform(-8.5, 8.5)] if rng.random() < 0.5 else [])
    return {"a": a, "b": b, "lanes": lanes, "steps": steps}


def render(w, at_target):
    pose = ego_pose(w["steps"]) if at_target else (0.0, 0.0, 0.0, 0.0)
    out = torch.zeros(3, G.nx, G.ny)
    for ch, objs, vel in ((0, w["a"], VA), (1, w["b"], VB)):
        for px, py in objs:
            cx, cy = move(px, py, pose, *vel)
            out[ch] = torch.maximum(out[ch], torch.exp(-((GX - cx) ** 2 + (GY - cy) ** 2) / (2 * 0.6 ** 2)))
    x0 = torch.arange(-15.0, 45.0, 0.1)
    for yl in w["lanes"]:
        cx, cy = move(x0, torch.full_like(x0, yl), pose, 0.0, 0.0)
        ix, iy = ((cx - G.x_min) / G.res).floor().long(), ((cy - G.y_min) / G.res).floor().long()
        ok = (ix >= 0) & (ix < G.nx) & (iy >= 0) & (iy < G.ny)
        out[2, ix[ok], iy[ok]] = 1.0
    out[2] *= (out[:2].max(0).values < 0.3).float()                     # the stripes are hidden under the objects
    return out


def holes(w, src):
    """(X, Y) bool: target cells no source cell reaches under the true motion (true velocity per source cell)."""
    vx = torch.where(src[0] > 0.3, VA[0], torch.where(src[1] > 0.3, VB[0], 0.0))
    vy = torch.where(src[0] > 0.3, VA[1], torch.where(src[1] > 0.3, VB[1], 0.0))
    cx, cy = move(GX, GY, ego_pose(w["steps"]), vx, vy)
    pos = torch.stack([cx.flatten(), cy.flatten()], -1)[None]
    _, mass = gaussian_splat(torch.ones(1, 1, pos.shape[1]), pos, torch.ones(1, pos.shape[1]), G, 0.0, 2)
    return mass[0, 0] < 0.5


def sample(rng):
    w = world(rng)
    src, tgt = render(w, False), render(w, True)
    return src, tgt, w["steps"], holes(w, src)


class Stage(torch.nn.Module):
    def __init__(self, n_hyp, fill):
        super().__init__()
        self.n_hyp = n_hyp
        self.tp = TemporalPropagation(cin=3, grid=G, n_hyp=max(n_hyp, 0), static_hyp=True, init_speed=5.0)
        if n_hyp == 0:
            self.tp.set_velocity((0.0, 0.0), static_weight=0.0)          # only the static hypothesis: ego motion
        self.fill = GapFill(3) if fill else None

    def forward(self, src, steps):
        if self.n_hyp < 0:                                                # stage 0: copy
            return src, None
        r = self.tp(src, steps)
        return (r["feat"] if self.fill is None else self.fill(r["feat"], r["mass"])), r


def train(model, data, iters, rng, bs=4, lr=3e-3):
    params = [p for p in model.parameters() if p.requires_grad]
    if model.n_hyp < 0 or (model.n_hyp == 0 and model.fill is None):
        return                                                            # nothing to learn
    opt = torch.optim.Adam(params, lr=lr)
    for it in range(iters):
        batch = [data[rng.randrange(len(data))] for _ in range(bs)]
        out, _ = model(torch.stack([b[0] for b in batch]).to(DEV), [b[2] for b in batch])
        loss = ((out - torch.stack([b[1] for b in batch]).to(DEV)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def evaluate(model, data):
    se = {"all": 0.0, "known": 0.0, "holes": 0.0}
    n = {"all": 0, "known": 0, "holes": 0}
    vel = {"A": [], "B": []}
    for src, tgt, steps, hole in data:
        out, r = model(src[None].to(DEV), [steps])
        e = ((out[0].cpu() - tgt) ** 2).sum(0)                                  # (X, Y) summed over channels
        for k, m in (("all", torch.ones_like(hole)), ("known", ~hole), ("holes", hole)):
            se[k] += e[m].sum().item()
            n[k] += int(m.sum()) * 3
        if r is not None and model.n_hyp > 0:
            for key, ch, vt in (("A", 0, VA), ("B", 1, VB)):
                m = src[ch] > 0.6
                if m.any():
                    v = r["v_mean"][0].cpu()[:, m].mean(1)
                    vel[key].append(math.hypot(v[0].item() - vt[0], v[1].item() - vt[1]))
    res = {k: se[k] / max(1, n[k]) for k in se}
    for k in vel:
        res["vel " + k] = sum(vel[k]) / len(vel[k]) if vel[k] else float("nan")
    return res


STAGES = (("0 copy", -1, False), ("1 ego", 0, False), ("2 ego + fill", 0, True), ("3 hyp", 4, False),
          ("4 hyp + fill", 4, True))


def picture(models, data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rgb(m):                                       # x forward = up, +y = left; A red, B green, stripes blue
        return m.clamp(0, 1).permute(1, 2, 0).flip(1).numpy()

    src, tgt, steps, hole = data[0]
    panels = [("source t-0.5 s", rgb(src)), ("target t", rgb(tgt))]
    with torch.no_grad():
        for name, model in models:
            out, _ = model(src[None].to(DEV), [steps])
            panels.append((name, rgb(out[0].cpu())))
    panels.append(("holes (true motion)", hole.float().flip(1).numpy()))
    fig, ax = plt.subplots(1, len(panels), figsize=(2.2 * len(panels), 2.6))
    for a, (t, im) in zip(ax, panels):
        a.imshow(im, origin="lower", cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1)
        a.set_title(t, fontsize=8)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    t0 = time.time()
    n_train, n_test, iters = (200, 12, 200) if QUICK else (600, 32, 800)
    seeds = (0,) if QUICK else (0, 1)
    test = [sample(random.Random(10_000 + i)) for i in range(n_test)]
    frac = sum(int(d[3].sum()) for d in test) / (n_test * G.nx * G.ny)
    print(f"device {DEV}; world: {n_train} training scenes, {n_test} held-out, {iters} steps of 4 scenes; holes {100 * frac:.1f} % of the cells", flush=True)
    claims = []
    for seed in seeds:
        rng = random.Random(seed)
        data = [sample(rng) for _ in range(n_train)]
        torch.manual_seed(seed)
        res, models = {}, []
        for name, n_hyp, fill in STAGES:
            m = Stage(n_hyp, fill).to(DEV)
            train(m, data, iters, random.Random(seed + 7))
            res[name] = evaluate(m, test)
            models.append((name, m))
            r = res[name]
            print(f"  seed {seed}  {name:14s} MSE all {r['all']:.5f}  known {r['known']:.5f}  holes {r['holes']:.5f}"
                  f"   vel err A {r['vel A']:.2f}  B {r['vel B']:.2f} m/s   ({time.time() - t0:.0f} s)", flush=True)
        c, e, ef, h, hf = (res[s[0]] for s in STAGES)
        claims += [(f"seed {seed}: ego beats copy (all cells)", e["all"] < c["all"]),
                   (f"seed {seed}: hypotheses beat ego (known cells)", h["known"] < e["known"]),
                   (f"seed {seed}: fill lowers the hole error (ego)", ef["holes"] < 0.7 * e["holes"]),
                   (f"seed {seed}: fill lowers the hole error (hyp)", hf["holes"] < 0.7 * h["holes"]),
                   (f"seed {seed}: fill does not worsen the known cells (ego, <= +1 %)", ef["known"] <= 1.01 * e["known"]),
                   (f"seed {seed}: hypotheses learn both velocities (< 1 m/s)", max(h["vel A"], h["vel B"]) < 1.0),
                   (f"seed {seed}: ... also with the fill (< 1 m/s)", max(hf["vel A"], hf["vel B"]) < 1.0)]
        if seed == seeds[0]:
            picture(models[1:], test, "temp/checks/temporal_stages.png")
    print("== claims")
    for name, ok in claims:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"== {sum(ok for _, ok in claims)} / {len(claims)} claims hold; image temp/checks/temporal_stages.png")
