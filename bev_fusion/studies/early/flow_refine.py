"""Motion flow field with iterative refinement (plan.md 7.3; Mirko 2026-10-26: "get a good motion flow field, try 4
refinements"). World of studies/early/temporal_stages.py: objects A (8, 2) m/s and B (-2, -6) m/s, static stripes,
random ego motion over 0.5 s. Trained on the warp MSE only (no velocity label).
Arms:
  H        four velocity hypotheses + static (studies/early/temporal.py TemporalPropagation), flow = pi-weighted mean
  H + R4   H, then 4 iterations of FlowRefine (correlation with the current frame around the landing point, shared
           conv update, v detached between iterations, loss on every iteration weighted 0.8^(K - k))
  0 + R4   FlowRefine from v = 0 (ego motion only) - can the lookup alone find motions of up to 4 m (8 cells)?
  H + R4 + prior   H + R4 with a zero-motion prior: lambda * mean(occupancy * |v / 10|^2) on every flow (aperture
           problem: a stripe sliding along itself does not change the warp, so the warp alone leaves its speed free)
  S        sharp hypotheses: softmax-splatting importance (an object wins over the empty cells landing on it) and an
           entropy penalty on pi (no hedging: the mixture spread objects over several velocities, faint and ringed)
  S + R4   S, then the 4 refinement iterations (warps with the same importance)
  0 + P4 / S + P4   coarse to fine (FlowPyramid): 2 iterations at 2 m cells (window +-6 m), 1 at 1 m, 1 at 0.5 m
  W        hypotheses trained winner-takes-all (relaxed MCL, Rupprecht 1612.00197): every hypothesis splatted alone,
           per target cell (1 - eps) min_m err_m + eps mean_m err_m, eps 0.05 - the hypotheses specialise instead of
           averaging; the mixture loss still trains pi
  S + R4 + smooth   S + R4 with an edge-aware smoothness term on every flow: lambda * mean(w |v_i - v_j| / 10) over
           the 4-neighbours, w = exp(-10 |f_i - f_j|) (strong inside an object, weak across its boundary). Reason: the
           warp loss is gameable - the TRUE field gets warp MSE 0.0080 on the known cells, the refined fields 0.0013:
           per-cell velocities that reshape blobs and stripes (against the splat blur and the aliased rendering) beat
           the real motion; the reference row "true velocity" is printed with every run
  H + R4 + label   the supervised control (plan 7.3 arm A): H + R4 plus an L1 velocity label on the occupied cells of
           every flow (as box-supervised detectors do; nuScenes has the labels) - the warp loss stays as an auxiliary;
           shows what the refinement does when the field is anchored
  S + D + R4 + smooth   the combination the single arms point to: sharp hypotheses (best start), Doppler (stripes),
           4 refinements (they recover a wrong start), edge-aware smoothness (limits the gaming); result: the start is the
           best field so far (A 0.20, B 0.48, stripes 0.07 m/s) but the refinement undoes the Doppler (stripes 2.33)
  S + DD + R4 + smooth  the same with the Doppler update applied after EVERY refinement iteration (FlowRefine correct=)
  S + D / S + D + P4   the hypothesis mean updated with radar Doppler (temporal.doppler_update, prior sigma 4 m/s): a
           radar at the origin, a detection in 50 % of the occupied cells (objects AND static stripes: ego-compensated
           clutter has radial velocity 0), noise 0.1 m/s - for stripes ahead the ray is nearly along the stripe, the
           direction the warp cannot see (aperture)
The refinement step is bounded by the lookup window (+-1.5 m per iteration; FlowRefine step_max).
Metrics on held-out scenes after each iteration: end-point error of the velocity (m/s) on the cells of A, of B and
of the static stripes; warp MSE on the known cells. Image: temp/checks/flow_refine.png (direction = hue, speed =
brightness, on occupied cells).
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python studies/early/flow_refine.py [--quick] [--seed N] [--prior 1e-3] [--ent 5e-3]
    [--arms H,H+R4,0+R4,H+R4+prior,S,S+R4,0+P4,S+P4,S+D,S+D+P4,W,S+R4+smooth,H+R4+label,S+D+R4+smooth,S+DD+R4+smooth] [--smooth 0.05]
"""
import random
import sys
import time

import torch

sys.path.insert(0, "training")
sys.path.insert(0, "tests/support")
from flow_viz import flow_rgb, wheel  # noqa: E402
from temporal import FlowPyramid, FlowRefine, TemporalPropagation, doppler_update, propagate  # noqa: E402
from temporal_stages import G, VA, VB, sample  # noqa: E402

QUICK = "--quick" in sys.argv
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
DEV = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 3e9 else "cpu"
K = 4
PRIOR = float(sys.argv[sys.argv.index("--prior") + 1]) if "--prior" in sys.argv else 1e-3
ENT = float(sys.argv[sys.argv.index("--ent") + 1]) if "--ent" in sys.argv else 5e-3
SMOOTH = float(sys.argv[sys.argv.index("--smooth") + 1]) if "--smooth" in sys.argv else 0.05
ARMS = sys.argv[sys.argv.index("--arms") + 1].split(",") if "--arms" in sys.argv else None


def true_flow(src):
    vx = torch.where(src[0] > 0.3, VA[0], torch.where(src[1] > 0.3, VB[0], 0.0))
    vy = torch.where(src[0] > 0.3, VA[1], torch.where(src[1] > 0.3, VB[1], 0.0))
    return torch.stack([vx, vy])


def radar_measure(src, sigma_d=0.1, rate=0.5):
    """simulated Doppler detections: radar at the origin, a detection in a random half of the occupied cells,
    d = u . v_true + noise (objects: their velocity, stripes: 0) -> (u, d, valid), each (B, 1, ..., N)."""
    B, N = src.shape[0], G.nx * G.ny
    xs, ys = G.xy_centres(device=src.device)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    r = torch.sqrt(gx ** 2 + gy ** 2).clamp(min=1e-3)
    u = torch.stack([gx / r, gy / r]).reshape(1, 1, 2, N).expand(B, 1, 2, N)
    vt = torch.stack([true_flow(s_.cpu()) for s_ in src]).to(src.device).reshape(B, 2, N)
    d = (u[:, 0] * vt).sum(1, keepdim=True) + sigma_d * torch.randn(B, 1, N, device=src.device)
    valid = ((src.max(1).values > 0.5) & (torch.rand(B, G.nx, G.ny, device=src.device) < rate)).float().reshape(B, 1, N)
    return u, d, valid


def radar_apply(v, meas, sigma_p=4.0, sigma_d=0.1):
    u, d, valid = meas
    B, N = v.shape[0], G.nx * G.ny
    P = (torch.eye(2, device=v.device) * sigma_p ** 2).view(1, 2, 2, 1).expand(B, 2, 2, N)
    out, _ = doppler_update(v.reshape(B, 2, N), P, u, d, valid, sigma_d=sigma_d)
    return out.view_as(v)


class Arm(torch.nn.Module):
    def __init__(self, hyp, refine, prior=0.0, sharp=False, doppler=False, wta=False, smooth=0.0, label=False,
                 doppler_every=False):
        super().__init__()
        self.doppler_every = doppler_every
        self.prior, self.sharp, self.doppler, self.wta, self.smooth, self.label = prior, sharp, doppler, wta, smooth, label
        self.tp = TemporalPropagation(cin=3, grid=G, n_hyp=4, static_hyp=True, init_speed=5.0, importance=sharp) if hyp else None
        self.rf = None if not refine else (FlowPyramid(3, G, factors=(4, 2, 1), iters=(2, 1, 1)) if refine == "P"
                                           else FlowRefine(3, G, radius=3))

    def forward(self, src, tgt, steps):
        """-> list of (flow, warped map); the first entry is the initial flow, then one per refinement iteration;
        and the hypothesis mixture's own warped map (or None)."""
        imp, self.ent, self.per_hyp = None, None, None
        if self.tp is not None:
            r = self.tp(src, steps)
            if self.wta and self.training:
                self.per_hyp = [propagate(src, steps, r["v_hyp"][:, m], G)[0] for m in range(r["v_hyp"].shape[1])]
            v0, mix = r["v_mean"], r["feat"]
            if self.sharp:
                imp = self.tp.importance_of(src.flatten(2)).view(src.shape[0], G.nx, G.ny)
                self.ent = r["entropy"]
        else:
            v0, mix = src.new_zeros(src.shape[0], 2, G.nx, G.ny), None
        corr = None
        if self.doppler:
            meas = radar_measure(src)
            v0 = radar_apply(v0, meas)
            if self.doppler_every:
                corr = lambda v: radar_apply(v, meas)                     # noqa: E731
        if self.rf is None:
            flows = [v0]
        elif isinstance(self.rf, FlowPyramid):
            flows = [v0] + self.rf(src, tgt, steps, v0, correct=corr)
        else:
            flows = [v0] + self.rf(src, tgt, steps, v0, iters=K, correct=corr)
        return [(v, propagate(src, steps, v, G, importance=imp)[0]) for v in flows], mix


def smoothness(v, feat, alpha=10.0):
    """edge-aware first-order smoothness of a flow (B, 2, X, Y) m/s over the 4-neighbours, weights from the map."""
    tot = 0.0
    for dim in (2, 3):
        dv = (v.diff(dim=dim) / 10.0).abs().sum(1)
        w = torch.exp(-alpha * feat.diff(dim=dim).abs().sum(1))
        tot = tot + (w * dv).mean()
    return tot


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
        if arm.label:
            occ = (src.max(1).values > 0.3).float()
            vt = torch.stack([true_flow(s_.cpu()) for s_ in src]).to(DEV)
            loss = loss + 0.1 * sum((occ * (v - vt).abs().sum(1)).sum() / occ.sum().clamp(min=1) for v, _ in outs)
        if arm.smooth > 0:
            loss = loss + arm.smooth * sum(smoothness(v, src) for v, _ in outs)
        if arm.per_hyp is not None:
            err = torch.stack([((ph - tgt) ** 2).sum(1) for ph in arm.per_hyp])        # (M, B, X, Y)
            loss = loss + (0.95 * err.min(0).values + 0.05 * err.mean(0)).mean() / tgt.shape[1]
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




def picture(arms, data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    src, tgt, steps, _ = data[0]
    occ = (src.max(0).values > 0.3).float()
    rgb = lambda m: m.clamp(0, 1).permute(1, 2, 0).flip(1).numpy()          # noqa: E731
    fig, ax = plt.subplots(len(arms), 4 + K, figsize=(2.0 * (4 + K), 2.3 * len(arms)), squeeze=False)
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
    wheel(ax[0, -1], vmax=10.0)
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
    se, nk = 0.0, 0
    for src, tgt, steps, hole in test:                                  # reference: the TRUE velocity field
        e = ((propagate(src[None], [steps], true_flow(src)[None], G)[0][0] - tgt) ** 2).sum(0)
        se, nk = se + e[~hole].sum().item(), nk + int((~hole).sum()) * 3
    print(f"  true velocity   (reference)  EPE A  0.00  B  0.00  static  0.00 m/s   warp MSE known {se / nk:.5f}", flush=True)
    arms = []
    for name, hyp, refine, prior, sharp in (("H", True, False, 0.0, False), ("H + R4", True, True, 0.0, False),
                                            ("0 + R4", False, True, 0.0, False), ("H + R4 + prior", True, True, PRIOR, False),
                                            ("S", True, False, 0.0, True), ("S + R4", True, True, 0.0, True),
                                            ("0 + P4", False, "P", 0.0, False), ("S + P4", True, "P", 0.0, True),
                                            ("S + D", True, False, 0.0, True), ("S + D + P4", True, "P", 0.0, True),
                                            ("W", True, False, 0.0, False), ("S + R4 + smooth", True, True, 0.0, True),
                                            ("H + R4 + label", True, True, 0.0, False),
                                            ("S + D + R4 + smooth", True, True, 0.0, True),
                                            ("S + DD + R4 + smooth", True, True, 0.0, True)):
        if ARMS is not None and name.replace(" ", "") not in ARMS:
            continue
        torch.manual_seed(SEED)
        print(f"  {name}", flush=True)
        arm = Arm(hyp, refine, prior, sharp, doppler=" D" in name, wta=name == "W", doppler_every=" DD" in name,
                  smooth=SMOOTH if "smooth" in name else 0.0, label="label" in name).to(DEV)
        train(arm, data, iters, random.Random(SEED + 7))
        arm.eval()
        for i, r in enumerate(evaluate(arm, test)):
            print(f"  {name:15s} {'initial    ' if i == 0 else f'iteration {i}'}  EPE A {r['A']:5.2f}  B {r['B']:5.2f}  static {r['static']:5.2f} m/s"
                  f"   warp MSE known {r['mse']:.5f}", flush=True)
        arms.append((name, arm))
    tag = "" if ARMS is None else "_" + "_".join(a.replace("+", "p") for a in ARMS)
    picture(arms, test, f"temp/checks/flow_refine{tag}.png")
    print(f"image temp/checks/flow_refine{tag}.png  ({time.time() - t0:.0f} s)")
