"""Motion flow field with iterative refinement (plan.md 7.3; Mirko 2026-10-26: "get a good motion flow field, try 4
refinements"). World of tests/test_temporal_stages.py: objects A (8, 2) m/s and B (-2, -6) m/s, static stripes,
random ego motion over 0.5 s. Trained on the warp MSE only (no velocity label).
Arms:
  H        four velocity hypotheses + static (training/temporal.py TemporalPropagation), flow = pi-weighted mean
  H + R4   H, then 4 iterations of FlowRefine (correlation with the current frame around the landing point, shared
           conv update, v detached between iterations, loss on every iteration weighted 0.8^(K - k))
  0 + R4   FlowRefine from v = 0 (ego motion only) - can the lookup alone find motions of up to 4 m (8 cells)?
  H + R4 + prior   H + R4 with a zero-motion prior: lambda * mean(occupancy * |v / 10|^2) on every flow (aperture
           problem: a stripe sliding along itself does not change the warp, so the warp alone leaves its speed free)
  S        sharp hypotheses: softmax-splatting importance (an object wins over the empty cells landing on it) and an
           entropy penalty on pi (no hedging: the mixture spread objects over several velocities, faint and ringed)
  S + R4   S, then the 4 refinement iterations (warps with the same importance)
  0 + P4 / S + P4   coarse to fine (FlowPyramid): 2 iterations at 2 m cells (window +-6 m), 1 at 1 m, 1 at 0.5 m
The refinement step is bounded by the lookup window (+-1.5 m per iteration; FlowRefine step_max).
Metrics on held-out scenes after each iteration: end-point error of the velocity (m/s) on the cells of A, of B and
of the static stripes; warp MSE on the known cells. Image: temp/checks/flow_refine.png (direction = hue, speed =
brightness, on occupied cells).
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_flow_refine.py [--quick] [--seed N] [--prior 1e-3] [--ent 5e-3]
    [--arms H,H+R4,0+R4,H+R4+prior,S,S+R4,0+P4,S+P4]
"""
import math
import random
import sys
import time

import torch

sys.path.insert(0, "training")
sys.path.insert(0, "tests")
from temporal import FlowPyramid, FlowRefine, TemporalPropagation, propagate  # noqa: E402
from test_temporal_stages import G, VA, VB, sample  # noqa: E402

QUICK = "--quick" in sys.argv
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
DEV = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 3e9 else "cpu"
K = 4
PRIOR = float(sys.argv[sys.argv.index("--prior") + 1]) if "--prior" in sys.argv else 1e-3
ENT = float(sys.argv[sys.argv.index("--ent") + 1]) if "--ent" in sys.argv else 5e-3
ARMS = sys.argv[sys.argv.index("--arms") + 1].split(",") if "--arms" in sys.argv else None


def true_flow(src):
    vx = torch.where(src[0] > 0.3, VA[0], torch.where(src[1] > 0.3, VB[0], 0.0))
    vy = torch.where(src[0] > 0.3, VA[1], torch.where(src[1] > 0.3, VB[1], 0.0))
    return torch.stack([vx, vy])


class Arm(torch.nn.Module):
    def __init__(self, hyp, refine, prior=0.0, sharp=False):
        super().__init__()
        self.prior, self.sharp = prior, sharp
        self.tp = TemporalPropagation(cin=3, grid=G, n_hyp=4, static_hyp=True, init_speed=5.0, importance=sharp) if hyp else None
        self.rf = None if not refine else (FlowPyramid(3, G, factors=(4, 2, 1), iters=(2, 1, 1)) if refine == "P"
                                           else FlowRefine(3, G, radius=3))

    def forward(self, src, tgt, steps):
        """-> list of (flow, warped map); the first entry is the initial flow, then one per refinement iteration;
        and the hypothesis mixture's own warped map (or None)."""
        imp, self.ent = None, None
        if self.tp is not None:
            r = self.tp(src, steps)
            v0, mix = r["v_mean"], r["feat"]
            if self.sharp:
                imp = self.tp.importance_of(src.flatten(2)).view(src.shape[0], G.nx, G.ny)
                self.ent = r["entropy"]
        else:
            v0, mix = src.new_zeros(src.shape[0], 2, G.nx, G.ny), None
        if self.rf is None:
            flows = [v0]
        elif isinstance(self.rf, FlowPyramid):
            flows = [v0] + self.rf(src, tgt, steps, v0)
        else:
            flows = [v0] + self.rf(src, tgt, steps, v0, iters=K)
        return [(v, propagate(src, steps, v, G, importance=imp)[0]) for v in flows], mix


def train(arm, data, iters, rng, bs=8, lr=2e-3):
    opt = torch.optim.Adam(arm.parameters(), lr=lr)
    t0 = time.time()
    for it in range(iters):
        batch = [data[rng.randrange(len(data))] for _ in range(bs)]
        src = torch.stack([b[0] for b in batch]).to(DEV)
        tgt = torch.stack([b[1] for b in batch]).to(DEV)
        steps = [b[2] for b in batch]
        outs, mix = arm(src, tgt, steps)
        loss = 0.0 if mix is None else ((mix - tgt) ** 2).mean()
        if arm.ent is not None:
            loss = loss + ENT * (src.max(1).values.clamp(0, 1) * arm.ent).mean()
        if arm.prior > 0:
            occ = src.max(1).values.clamp(0, 1)
            loss = loss + arm.prior * sum((occ * (v / 10.0).pow(2).sum(1)).mean() for v, _ in outs)
        n = len(outs) - 1
        for k, (_, w) in enumerate(outs[1:]):
            loss = loss + 0.8 ** (n - 1 - k) * ((w - tgt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(arm.parameters(), 5.0)
        opt.step()
        if it % max(1, iters // 4) == 0 or it == iters - 1:
            print(f"    iter {it:5d} loss {loss.item():.5f}  ({time.time() - t0:.0f} s)", flush=True)


@torch.no_grad()
def evaluate(arm, data):
    """-> per stage (initial, iteration 1..K): EPE A, B, static (m/s) and warp MSE on known cells."""
    acc = None
    for src, tgt, steps, hole in data:
        outs, _ = arm(src[None].to(DEV), tgt[None].to(DEV), [steps])
        vt = true_flow(src)
        masks = {"A": src[0] > 0.6, "B": src[1] > 0.6, "static": (src[2] > 0.5) & (src[:2].max(0).values < 0.1)}
        if acc is None:
            acc = [{k: [0.0, 0] for k in ("A", "B", "static", "mse")} for _ in outs]
        for i, (v, w) in enumerate(outs):
            e = (v[0].cpu() - vt).norm(dim=0)
            for k, m in masks.items():
                acc[i][k][0] += e[m].sum().item()
                acc[i][k][1] += int(m.sum())
            known = ~hole
            acc[i]["mse"][0] += ((w[0].cpu() - tgt) ** 2).sum(0)[known].sum().item()
            acc[i]["mse"][1] += int(known.sum()) * 3
    return [{k: s / max(1, n) for k, (s, n) in a.items()} for a in acc]


def flow_rgb(v, occ):
    from matplotlib.colors import hsv_to_rgb
    ang = (torch.atan2(v[1], v[0]) / (2 * math.pi)) % 1.0
    val = (v.norm(dim=0) / 10.0).clamp(0, 1)
    rgb = torch.tensor(hsv_to_rgb(torch.stack([ang, torch.ones_like(ang), val], -1).numpy()))
    return (rgb * occ.unsqueeze(-1)).flip(1).numpy()


def picture(arms, data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    src, tgt, steps, _ = data[0]
    occ = (src.max(0).values > 0.3).float()
    rgb = lambda m: m.clamp(0, 1).permute(1, 2, 0).flip(1).numpy()          # noqa: E731
    fig, ax = plt.subplots(len(arms), 3 + K, figsize=(2.0 * (3 + K), 2.3 * len(arms)), squeeze=False)
    for row, (name, arm) in enumerate(arms):
        with torch.no_grad():
            outs, _ = arm(src[None].to(DEV), tgt[None].to(DEV), [steps])
        panels = [("source / target", rgb(torch.maximum(src, 0.5 * tgt))), ("true flow", flow_rgb(true_flow(src), occ))]
        panels += [(f"{name}: " + ("initial" if i == 0 else f"iteration {i}"), flow_rgb(v[0].cpu(), occ)) for i, (v, _) in enumerate(outs)]
        for a in ax[row]:
            a.axis("off")
        for a, (t, im) in zip(ax[row], panels):
            a.imshow(im, origin="lower")
            a.set_title(t, fontsize=7)
    fig.suptitle("velocity field (hue = direction, brightness = speed, 10 m/s = full); x forward = up", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    t0 = time.time()
    n_train, n_test, iters = (200, 12, 300) if QUICK else (600, 32, 1500)
    rng = random.Random(SEED)
    data = [sample(rng) for _ in range(n_train)]
    test = [sample(random.Random(10_000 + i)) for i in range(n_test)]
    print(f"device {DEV}, seed {SEED}; {n_train} training scenes, {n_test} held-out, {iters} steps of 8 ({time.time() - t0:.0f} s to render)", flush=True)
    arms = []
    for name, hyp, refine, prior, sharp in (("H", True, False, 0.0, False), ("H + R4", True, True, 0.0, False),
                                            ("0 + R4", False, True, 0.0, False), ("H + R4 + prior", True, True, PRIOR, False),
                                            ("S", True, False, 0.0, True), ("S + R4", True, True, 0.0, True),
                                            ("0 + P4", False, "P", 0.0, False), ("S + P4", True, "P", 0.0, True)):
        if ARMS is not None and name.replace(" ", "") not in ARMS:
            continue
        torch.manual_seed(SEED)
        print(f"  {name}", flush=True)
        arm = Arm(hyp, refine, prior, sharp).to(DEV)
        train(arm, data, iters, random.Random(SEED + 7))
        arm.eval()
        for i, r in enumerate(evaluate(arm, test)):
            print(f"  {name:15s} {'initial    ' if i == 0 else f'iteration {i}'}  EPE A {r['A']:5.2f}  B {r['B']:5.2f}  static {r['static']:5.2f} m/s"
                  f"   warp MSE known {r['mse']:.5f}", flush=True)
        arms.append((name, arm))
    tag = "" if ARMS is None else "_" + "_".join(a.replace("+", "p") for a in ARMS)
    picture(arms, test, f"temp/checks/flow_refine{tag}.png")
    print(f"image temp/checks/flow_refine{tag}.png  ({time.time() - t0:.0f} s)")
