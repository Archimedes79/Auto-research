"""Does the velocity field EXPLAIN the motion, or is the next frame only predicted well? (Mirko, 2026-10-26)

Sequence of BEV maps with moving boxes (tests/synthetic_world.py: cars, trucks, pedestrians with ground velocities,
lane lines, exact ego motion; frames F0, F1, F2 dt apart, like consecutive nuScenes GT keyframes). A small model of
its own - encoder E (3 x 3 convs) per frame, the temporal block (training/temporal.py: TemporalPropagation velocity
hypotheses + FlowRefine, K iterations, bounded steps) and decoder D - predicts F1 from F0 ONLY through the motion:
  P = D(propagate(E(F0), v)),  v = FlowRefine(E(F0), E(F1), v0 = hypothesis mean)     (E(F1) only for the lookup)
trained on BCE(P, F1) - no velocity label. The decoder could learn to move things itself (a conv net can shift);
the tests separate a good prediction from a velocity field that explains it:
  1 prediction      IoU of D(P) with F1 (vehicles)
  2 velocity        end-point error (m/s) against the GT box velocity on the box cells; on lane cells (aperture)
  3 explanation     the RAW map F0 moved with v alone (no encoder, no decoder) vs F1: IoU on vehicles
  4 extrapolation   v carried to frame 1 (moved with itself, rotated with the ego) moves F1 to t + 2 dt vs F2:
                    a field that only fits one pair does not extrapolate, a physical one does
  5 controls        v = 0 (ego motion only) through the same D and as a raw move; GT velocity as a raw move (the
                    reference of 3 and 4, limited only by the bilinear splat)
Claims (PASS / FAIL): EPE on vehicles < 1 m/s; raw move (3) and extrapolation (4) reach >= 90 % of the GT-velocity
IoU; the learned field beats v = 0 by > 0.1 IoU in 3; the prediction (1) is not much better than the raw move (3)
(< 0.1 IoU: the decoder does not carry the motion).
Image: temp/checks/flow_explains.png. Run from the bev_fusion folder:
  wsl ~/miniconda3/envs/gcar/bin/python tests/test_flow_explains.py [--quick] [--seed N] [--dt 0.25] [--prior 1e-3]
      [--sharp] (softmax-splatting importance + pi entropy 5e-3) [--pyr] (FlowPyramid 2 + 1 + 1 iterations instead of
      4 iterations of FlowRefine at 0.5 m)
"""
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "training")
sys.path.insert(0, "tests")
from lift_pd import Grid  # noqa: E402
from synthetic_world import gt_bev, random_scene  # noqa: E402
from temporal import FlowPyramid, FlowRefine, TemporalPropagation, propagate  # noqa: E402

QUICK = "--quick" in sys.argv


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


SEED, DT, PRIOR, K = arg("--seed", 0), arg("--dt", 0.25), arg("--prior", 1e-3), 4
SHARP, PYR, ENT = "--sharp" in sys.argv, "--pyr" in sys.argv, arg("--ent", 5e-3)
TAG = ("_sharp" if SHARP else "") + ("_pyr" if PYR else "")
G = Grid(x_min=0.0, x_max=40.0, y_min=-20.0, y_max=20.0, res=0.5)          # 80 x 80 cells
DEV = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 3e9 else "cpu"
C = 16


def frames_seeded(seed):
    return frames(random.Random(seed))


def dataset(seeds, name):
    """render the sequences of the given seeds (4 processes), cached in temp/checks/ (scratch, not in git)."""
    path = f"temp/checks/flow_explains_{name}_dt{DT}_{seeds[0]}_{len(seeds)}.pt"
    if os.path.exists(path):
        return torch.load(path)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(4) as pool:
        out = pool.map(frames_seeded, seeds, chunksize=8)
    torch.save(out, path)
    return out


def frames(rng):
    """-> maps (3, 3, X, Y) of F0, F1, F2 (lanes, vehicles, pedestrians), GT velocity of F0 (2, X, Y), box id of F0,
    ego steps from one frame to the next."""
    s0 = random_scene(rng, n_boxes=(3, 7), x_range=(6, 34), y_range=(-15, 15))
    s1 = s0.step(DT)
    s2 = s1.step(DT)
    g = [gt_bev(s, G, ss=2) for s in (s0, s1, s2)]
    maps = torch.stack([torch.stack([x["lanes"], x["vehicles"], x["pedestrians"]]) for x in g])
    return maps, g[0]["velocity"], g[0]["box_id"], [(DT, s0.v, s0.omega)]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, C, 3, padding=1))
        self.tp = TemporalPropagation(cin=C, grid=G, n_hyp=4, static_hyp=True, init_speed=5.0, importance=SHARP)
        self.rf = FlowPyramid(C, G, factors=(4, 2, 1), iters=(2, 1, 1)) if PYR else FlowRefine(C, G, radius=3)
        self.dec = nn.Sequential(nn.Conv2d(C + 1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 3, 1))

    def forward(self, f0, f1, steps, v_override=None):
        e0, e1 = self.enc(f0), self.enc(f1)
        r = self.tp(e0, steps)
        self.ent = r["entropy"]
        imp = self.tp.importance_of(e0.flatten(2)).view(e0.shape[0], G.nx, G.ny) if SHARP else None
        if v_override is not None:
            flows = [v_override]
        elif PYR:
            flows = [r["v_mean"]] + self.rf(e0, e1, steps, r["v_mean"])
        else:
            flows = [r["v_mean"]] + self.rf(e0, e1, steps, r["v_mean"], iters=K)
        preds = []
        for v in flows:
            p, mass = propagate(e0, steps, v, G, importance=imp)
            preds.append(self.dec(torch.cat([p, mass.clamp(max=1.0)], 1)))
        mix = self.dec(torch.cat([r["feat"], r["mass"].clamp(max=1.0)], 1))
        return flows, preds, mix


def train(model, data, iters, rng, bs=8, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for it in range(iters):
        batch = [data[rng.randrange(len(data))] for _ in range(bs)]
        m = torch.stack([b[0] for b in batch]).to(DEV)
        steps = [b[3] for b in batch]
        flows, preds, mix = model(m[:, 0], m[:, 1], steps)
        tgt = m[:, 1]
        loss = F.binary_cross_entropy_with_logits(mix, tgt)
        for k, p in enumerate(preds[1:]):
            loss = loss + 0.8 ** (len(preds) - 2 - k) * F.binary_cross_entropy_with_logits(p, tgt)
        if SHARP:
            loss = loss + ENT * (m[:, 0].max(1).values * model.ent).mean()
        if PRIOR > 0:
            occ = m[:, 0].max(1).values
            loss = loss + PRIOR * sum((occ * (v / 10.0).pow(2).sum(1)).mean() for v in flows)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if it % max(1, iters // 5) == 0 or it == iters - 1:
            print(f"    iter {it:5d} loss {loss.item():.4f}  ({time.time() - t0:.0f} s)", flush=True)


def rotate_into_next(v, steps):
    """velocity vectors (B, 2, X, Y) from the source axes into the axes after the steps (ego yaw)."""
    th = sum(dt * w for dt, _, w in steps[0])
    c, s = math.cos(th), math.sin(th)
    return torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], 1)


def carry(v, steps):
    """the velocity field of frame 0 moved to frame 1 (with itself) and rotated into its axes (no mass gate)."""
    moved, _ = propagate(v, steps, v, G, mass_gate=False)
    return rotate_into_next(moved, steps)


def iou(a, b, thr=0.5):
    a, b = a > thr, b > thr
    return (a & b).sum().item(), (a | b).sum().item()


@torch.no_grad()
def evaluate(model, test):
    keys = ("pred", "raw", "extrap", "pred v=0", "raw v=0", "raw GT v", "extrap GT v")
    acc = {k: [0, 0] for k in keys}
    epe = {"vehicles": [0.0, 0], "pedestrians": [0.0, 0], "lanes": [0.0, 0]}
    for maps, vgt, bid, steps in test:
        m = maps[None].to(DEV)
        flows, preds, _ = model(m[:, 0], m[:, 1], [steps])
        v = flows[-1]
        zero = torch.zeros_like(v)
        _, p0, _ = model(m[:, 0], m[:, 1], [steps], v_override=zero)
        vg = vgt[None].to(DEV)
        out = {"pred": torch.sigmoid(preds[-1]), "pred v=0": torch.sigmoid(p0[0]),
               "raw": propagate(m[:, 0], [steps], v, G)[0], "raw v=0": propagate(m[:, 0], [steps], zero, G)[0],
               "raw GT v": propagate(m[:, 0], [steps], vg, G)[0],
               "extrap": propagate(m[:, 1], [steps], carry(v, [steps]), G)[0],
               "extrap GT v": propagate(m[:, 1], [steps], carry(vg, [steps]), G)[0]}
        for k, o in out.items():
            ref = m[:, 2] if k.startswith("extrap") else m[:, 1]
            i, u = iou(o[0, 1], ref[0, 1])
            acc[k][0] += i
            acc[k][1] += u
        e = (v[0].cpu() - vgt).norm(dim=0)
        masks = {"vehicles": maps[0, 1] > 0.5, "pedestrians": maps[0, 2] > 0.5,
                 "lanes": (maps[0, 0] > 0.3) & (maps[0, 1:].max(0).values < 0.1)}
        for k, mk in masks.items():
            epe[k][0] += e[mk].sum().item()
            epe[k][1] += int(mk.sum())
    res = {"IoU " + k: i / max(1, u) for k, (i, u) in acc.items()}
    res.update({"EPE " + k: s / max(1, n) for k, (s, n) in epe.items()})
    return res


def flow_rgb(v, occ, vmax=15.0):
    from matplotlib.colors import hsv_to_rgb
    ang = (torch.atan2(v[1], v[0]) / (2 * math.pi)) % 1.0
    val = (v.norm(dim=0) / vmax).clamp(0, 1)
    rgb = torch.tensor(hsv_to_rgb(torch.stack([ang, torch.ones_like(ang), val], -1).numpy()))
    return (rgb * occ.unsqueeze(-1)).flip(1).numpy()


def overlay(a, b):
    """a in red, b in green, overlap yellow (vehicles + pedestrians); lanes grey."""
    oa, ob = a[1:].max(0).values.clamp(0, 1), b[1:].max(0).values.clamp(0, 1)
    lane = 0.35 * b[0].clamp(0, 1)
    return torch.stack([torch.maximum(oa, lane), torch.maximum(ob, lane), lane], -1).flip(1).numpy()


@torch.no_grad()
def picture(model, test, path, n=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["F0 (red) / F1 (green)", "GT velocity", "learned velocity", "D(P) (red) vs F1 (green)",
            "raw F0 moved with v (red) vs F1", "extrapolated (red) vs F2 (green)", "v = 0 raw (red) vs F1"]
    fig, ax = plt.subplots(n, len(cols), figsize=(2.3 * len(cols), 2.5 * n), squeeze=False)
    for r in range(n):
        maps, vgt, bid, steps = test[r]
        m = maps[None].to(DEV)
        flows, preds, _ = model(m[:, 0], m[:, 1], [steps])
        v = flows[-1]
        occ = (maps[0, 1:].max(0).values > 0.3).float()
        ims = [overlay(maps[0], maps[1]), flow_rgb(vgt, occ), flow_rgb(v[0].cpu(), occ),
               overlay(torch.sigmoid(preds[-1][0]).cpu(), maps[1]),
               overlay(propagate(m[:, 0], [steps], v, G)[0][0].cpu(), maps[1]),
               overlay(propagate(m[:, 1], [steps], carry(v, [steps]), G)[0][0].cpu(), maps[2]),
               overlay(propagate(m[:, 0], [steps], torch.zeros_like(v), G)[0][0].cpu(), maps[1])]
        for c, im in enumerate(ims):
            ax[r, c].imshow(im, origin="lower")
            ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=7)
    fig.suptitle(f"dt = {DT} s; velocity: hue = direction, brightness = speed (15 m/s full); x forward = up; "
                 "overlays: red = prediction, green = truth, yellow = both", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    t0 = time.time()
    n_train, n_test, iters = (300, 16, 600) if QUICK else (1500, 48, 3000)
    data = dataset([100_000 * SEED + i for i in range(n_train)], "train")
    test = dataset([50_000 + i for i in range(n_test)], "test")
    print(f"device {DEV}, seed {SEED}, dt {DT} s, prior {PRIOR}, sharp {SHARP}, pyramid {PYR}; {n_train} training sequences, {n_test} held-out, "
          f"{iters} steps of 8 ({time.time() - t0:.0f} s to render)", flush=True)
    torch.manual_seed(SEED)
    model = Model().to(DEV)
    train(model, data, iters, random.Random(SEED + 7))
    model.eval()
    r = evaluate(model, test)
    print("== held-out sequences (IoU of vehicles; EPE in m/s)")
    for k, x in r.items():
        print(f"  {k:22s} {x:.3f}")
    claims = [("velocity explains the vehicles: EPE < 1 m/s", r["EPE vehicles"] < 1.0),
              ("raw F0 moved with v alone reaches >= 90 % of the GT-velocity IoU", r["IoU raw"] >= 0.9 * r["IoU raw GT v"]),
              ("extrapolation to t + 2 dt reaches >= 90 % of the GT-velocity IoU", r["IoU extrap"] >= 0.9 * r["IoU extrap GT v"]),
              ("the learned field beats v = 0 by > 0.1 IoU (raw move)", r["IoU raw"] - r["IoU raw v=0"] > 0.1),
              ("the decoder does not carry the motion: prediction - raw move < 0.1 IoU", r["IoU pred"] - r["IoU raw"] < 0.1)]
    print("== claims")
    for name, ok in claims:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    picture(model, test, f"temp/checks/flow_explains{TAG}.png")
    print(f"== {sum(ok for _, ok in claims)} / {len(claims)} claims hold; image temp/checks/flow_explains{TAG}.png "
          f"({time.time() - t0:.0f} s)")
