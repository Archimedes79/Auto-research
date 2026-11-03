"""Level D of the radar encoding gate (Mirko, 2026-10-26): which CODEC of the stored fields keeps more information?

Same probe as level C (slot raster K = 3, two 3x3 convs, 1x1 decoder, held-out scenes vs train scenes, tolerance-weighted
loss), but the raster stores the detection fields in different codes and the physical fields are decoded from them:
  abs      r / 60, cos / sin psi (the radar block as it is)
  delta    predictor + residual (DPCM): dr = r - |c - s|, dpsi = psi - bearing(c - s); c = cell centre, s = ray origin
           (sensor position at the measurement time in the output frame = calibration + ego pose: side information that
           the system has at every stage, and that the decoder here gets for the detection it decodes)
  phys     r and psi are NOT stored: they follow from the position and the ray origin, r = |p - s|, psi = angle(p - s)
  fourier  r and psi as multi-scale periodic codes (range wavelengths 60 / 15 / 3.75 m, psi harmonics 1 / 8), unwrapped
           coarse to fine like the Doppler code; no side information needed
Position is always: cell index (implicit in the tensor, exact, translation equivariant) + sub-cell offset (explicit delta,
|o| <= 0.5 cell); the decoder adds the cell centre (a fixed function of the index, nothing learned).
Checks: the codec itself is lossless (decode the raster without learning = the oracle ceiling); then the learned probes.
Levels (second argument): none = D (codecs x capacity C = 64 / 128, ReLU probe); e = E (probe architecture: activation relu / leaky,
default / identity start - the probe starts as a pass-through, as a residual block does); f = F (tolerances x 0.25, two seeds,
leaky + identity start); cv = 5-fold cross-validation over scenes, paired codec comparison with a written rule (arg 3 = seeds per fold).
Results: tests/README.md and plan.md section 3.4.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python -u studies/early/radar_codecs.py [steps] [e | f | cv [seeds]]
"""
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "training")
sys.path.insert(0, "temp/radar")
import radar_bev as rb  # noqa: E402
import radar_recon_test as rt  # noqa: E402
from lift_pd import Grid  # noqa: E402

DEV = rt.DEV
G = Grid()
NX, NY, RES = G.nx, G.ny, G.res
LAM = (60.0, 15.0, 3.75)                                    # range wavelengths: coarse first, each 4 x finer
PSI_H = (1, 8)
R_TOL, PSI_TOL = 0.5, math.radians(3)
DOP_TOL, RCS_TOL, AGE_TOL, OFF_TOL = 0.5, 3.0, 0.1, 0.5
H = rb.HARMONICS
fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


class Skip(nn.Module):
    """encoder features + the raster itself into the 1x1 decoder (the lossless reference)."""

    def __init__(self, cin, C):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(cin, C, 3, padding=1), nn.ReLU(), nn.Conv2d(C, C, 3, padding=1), nn.ReLU())
        self.dec = nn.Sequential(nn.Conv2d(C + cin, C, 1), nn.ReLU(), nn.Conv2d(C, cin, 1))

    def forward(self, x):
        return self.dec(torch.cat([self.enc(x), x], 1))


class Probe(nn.Module):
    """encoder (two 3x3 convs) + 1x1 decoder; act 'relu' or 'leaky' (LeakyReLU 0.5: invertible, a ReLU deletes the negative half of
    every signed channel); ident = centre-tap identity start (the net starts as a near-lossless pass-through, training only
    has to keep it, instead of discovering the identity from noise)."""

    def __init__(self, cin, C, act="relu", ident=False):
        super().__init__()
        A = (lambda: nn.ReLU()) if act == "relu" else (lambda: nn.LeakyReLU(0.5))
        self.enc = nn.Sequential(nn.Conv2d(cin, C, 3, padding=1), A(), nn.Conv2d(C, C, 3, padding=1), A())
        self.dec = nn.Sequential(nn.Conv2d(C, C, 1), A(), nn.Conv2d(C, cin, 1))
        if ident:
            for conv in (self.enc[0], self.enc[2], self.dec[0], self.dec[2]):
                with torch.no_grad():
                    conv.weight.mul_(0.05)
                    for i in range(min(conv.weight.shape[0], conv.weight.shape[1])):
                        conv.weight[i, i, conv.weight.shape[2] // 2, conv.weight.shape[3] // 2] += 1.0
                    conv.bias.zero_()

    def forward(self, x):
        return self.dec(self.enc(x))


class Codec:
    """stored channels per detection (enc) and their inverse to physical fields (dec); tol = tolerance of every channel."""

    def __init__(self, name):
        self.name = name
        geo = {"abs": [R_TOL / 60, PSI_TOL, PSI_TOL],
               "delta": [R_TOL / 1.0, PSI_TOL / 0.05],
               "phys": [],
               "fourier": [R_TOL / 60] + [2 * math.pi * R_TOL / l for l in LAM[1:] for _ in (0, 1)] + [PSI_TOL * k for k in PSI_H for _ in (0, 1)]}[name]
        dop = [k * math.pi * DOP_TOL / rb.V_MAX for k in H for _ in (0, 1)]
        self.tol = geo + dop + [RCS_TOL / rb.RCS_SCALE, AGE_TOL / rb.AGE_SCALE, OFF_TOL, OFF_TOL]   # offsets: 0.5 cell = 0.25 m
        self.P = len(self.tol)
        w = torch.tensor([1.0 / t for t in self.tol], device=DEV)
        self.w = 5.0 * w / w.mean()

    def enc(self, d, ox, oy, cx, cy, sx, sy):
        tail = torch.cat([rb.doppler_code(d["vr"]), (d["rcs"] / rb.RCS_SCALE)[:, None], (d["dt"] / rb.AGE_SCALE)[:, None],
                          ox[:, None], oy[:, None]], 1)
        if self.name == "abs":
            geo = torch.stack([d["r"] / 60, torch.cos(d["psi"]), torch.sin(d["psi"])], 1)
        elif self.name == "delta":
            rho, beta = torch.hypot(cx - sx, cy - sy), torch.atan2(cy - sy, cx - sx)
            dpsi = torch.remainder(d["psi"] - beta + math.pi, 2 * math.pi) - math.pi
            geo = torch.stack([d["r"] - rho, dpsi / 0.05], 1)
        elif self.name == "phys":
            geo = torch.zeros(len(d["r"]), 0, device=DEV)
        else:
            a = [d["r"][:, None] / 60]
            for l in LAM[1:]:
                a.append(torch.stack([torch.sin(2 * math.pi * d["r"] / l), torch.cos(2 * math.pi * d["r"] / l)], 1))
            for k in PSI_H:
                a.append(torch.stack([torch.sin(k * d["psi"]), torch.cos(k * d["psi"])], 1))
            geo = torch.cat(a, 1)
        return torch.cat([geo, tail], 1)

    def dec(self, v, cx, cy, sx, sy):
        """(n, P) -> r, psi, vr, rcs, dt, ox, oy (physical)."""
        ox, oy = v[:, -2], v[:, -1]
        dt, rcs = v[:, -3] * rb.AGE_SCALE, v[:, -4] * rb.RCS_SCALE
        nd = 2 * len(H)
        vr = torch.atan2(v[:, -4 - nd], v[:, -3 - nd]) * rb.V_MAX / math.pi
        g = v[:, :v.shape[1] - 4 - nd]
        px, py = cx + ox * RES, cy + oy * RES
        if self.name == "abs":
            r, psi = g[:, 0] * 60, torch.atan2(g[:, 2], g[:, 1])
        elif self.name == "delta":
            r = torch.hypot(cx - sx, cy - sy) + g[:, 0]
            psi = torch.atan2(cy - sy, cx - sx) + g[:, 1] * 0.05
        elif self.name == "phys":
            r, psi = torch.hypot(px - sx, py - sy), torch.atan2(py - sy, px - sx)
        else:                                                                     # unwrap coarse to fine
            r = g[:, 0] * 60
            for i, l in enumerate(LAM[1:]):
                ph = torch.atan2(g[:, 1 + 2 * i], g[:, 2 + 2 * i]) / (2 * math.pi)
                r = l * (torch.round(r / l - ph) + ph)
            o = 1 + 2 * (len(LAM) - 1)
            psi = torch.atan2(g[:, o], g[:, o + 1])
            period = 2 * math.pi / PSI_H[1]
            ph = torch.atan2(g[:, o + 2], g[:, o + 3]) / PSI_H[1]
            psi = ph + period * torch.round((psi - ph) / period)
        return r, psi, vr, rcs, dt, ox, oy


def geometry(d, ix, iy):
    """cell centre and ray origin s = p - r u of every detection (side information of the sensor / ego pose)."""
    return (G.x_min + (ix + 0.5) * RES, G.y_min + (iy + 0.5) * RES,
            d["x"] - d["r"] * torch.cos(d["psi"]), d["y"] - d["r"] * torch.sin(d["psi"]))


def codec_raster(codec, det, B, K=3):
    """slot raster as rasterize_slots (same cell, same order) with the codec's channels: (B, 1 + (K + 1)(1 + P), NX, NY)."""
    idx, ix, iy, rank, ox, oy = rt.det_slots(det)
    d = {k: det[k][idx] for k in rt.KEYS}
    vals = torch.cat([torch.ones(len(idx), 1, device=DEV), codec.enc(d, ox, oy, *geometry(d, ix, iy))], 1)
    Wc = 1 + codec.P
    cell = (det["b"][idx] * NX + ix) * NY + iy
    out = torch.zeros(B * NX * NY, 1 + (K + 1) * Wc, device=DEV)
    out[:, 0].index_add_(0, cell, torch.ones(len(cell), device=DEV))
    for k in range(K):
        m = rank == k
        out[cell[m], 1 + k * Wc: 1 + (k + 1) * Wc] = vals[m]
    m = rank >= K
    rem = torch.zeros(B * NX * NY, Wc, device=DEV)
    rem.index_add_(0, cell[m], vals[m])
    out[:, 1 + K * Wc:] = torch.cat([(rem[:, :1] > 0).float(), rem[:, 1:] / rem[:, :1].clamp(min=1)], 1)
    return out.view(B, NX, NY, -1).permute(0, 3, 1, 2).contiguous()


def codec_loss(codec, out, tgt, K=3):
    Wc = 1 + codec.P
    loss = F.l1_loss(out[:, 0], torch.log1p(tgt[:, 0]))
    for k in range(K + 1):
        vi = 1 + k * Wc
        valid = (tgt[:, vi] > 0.5).float()
        loss = loss + F.binary_cross_entropy_with_logits(out[:, vi], valid)
        m = valid[:, None]
        loss = loss + ((out[:, vi + 1: vi + Wc] - tgt[:, vi + 1: vi + Wc]).abs() * codec.w.view(1, -1, 1, 1) * m).sum() / (m.sum() * codec.P).clamp(min=1)
    return loss


def train_codec(codec, variant, C, steps, frames, seed=0):
    torch.manual_seed(seed)
    cin = 1 + (rt.K + 1) * (1 + codec.P)
    if variant == "skip":
        net = Skip(cin, C)
    else:
        net = Probe(cin, C, act="leaky" if variant.startswith("leaky") else "relu", ident=variant.endswith("+id"))
    net = net.to(DEV)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = np.random.RandomState(seed)
    for _ in range(steps):
        fs = [frames[i] for i in g.choice(len(frames), 8, replace=False)]
        tgt = codec_raster(codec, rt.batch(fs), 8)
        loss = codec_loss(codec, net(tgt), tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net.eval()


@torch.no_grad()
def eval_codec(codec, net, frames):
    """per original detection: decode the vector of its own slot, compare in physical units. Returns errors and the ok mask."""
    Wc, K = 1 + codec.P, rt.K
    acc = {k: [] for k in rt.TOL}
    for i in range(0, len(frames), 8):
        fs = frames[i:i + 8]
        det = rt.batch(fs)
        tgt = codec_raster(codec, det, len(fs))
        out = tgt if net is None else net(tgt)
        idx, ix, iy, rank, _, _ = rt.det_slots(det)
        d = {k: det[k][idx] for k in rt.KEYS}
        b = det["b"][idx]
        vi = 1 + rank.clamp(max=K) * Wc
        cols = (vi + 1)[:, None] + torch.arange(codec.P, device=DEV)[None]
        vec = out[b[:, None], cols, ix[:, None], iy[:, None]]
        valid = out[b, vi, ix, iy] > (0.5 if net is None else 0.0)
        cx, cy, sx, sy = geometry(d, ix, iy)
        r, psi, vr, rcs, dt, ox, oy = codec.dec(vec, cx, cy, sx, sy)
        e = {"pos (m)": torch.hypot(cx + ox * RES - d["x"], cy + oy * RES - d["y"]),
             "dist (m)": (r - d["r"]).abs(),
             "dir (deg)": torch.rad2deg(torch.remainder(psi - d["psi"] + math.pi, 2 * math.pi) - math.pi).abs(),
             "vr (m/s)": (torch.remainder(vr - d["vr"] + rb.V_MAX, 2 * rb.V_MAX) - rb.V_MAX).abs(),
             "rcs (dB)": (rcs - d["rcs"]).abs(), "age (s)": (dt - d["dt"]).abs()}
        for k in acc:
            acc[k].append(torch.where(valid, e[k], torch.full_like(e[k], float("inf"))))
    e = {k: torch.cat(v) for k, v in acc.items()}
    ok = torch.ones_like(e["pos (m)"], dtype=torch.bool)
    for k, t in rt.TOL.items():
        ok &= e[k] <= t
    return e, ok


def fail_share(e):
    """share of all detections that violate the tolerance of each field (invalid slots count as violations)."""
    return ", ".join(f"{100 * (e[k] > t).float().mean().item():.0f}" for k, t in rt.TOL.items())


def level_d(steps):
    print(f"D  codecs of the stored fields, same probe (2 x 3x3 conv + 1x1 decoder), tolerance-weighted loss, {steps} steps")
    tr, te = rt.load(set(range(8))), rt.load({8, 9})
    oracle = {}
    for name in ("abs", "delta", "phys", "fourier"):                              # the codec itself must be lossless
        e, ok = eval_codec(Codec(name), None, te)
        oracle[name] = ok.float().mean().item()
        print(f"  codec {name:8s} oracle (decode the raster itself, own slot or remainder mean): recovered {100 * oracle[name]:.1f}%")
        check(abs(oracle[name] - oracle["abs"]) < 0.005, f"codec {name}: as lossless as abs (within 0.5 points)")
    ceiling = oracle["abs"]
    print(f"  ceiling = {100 * ceiling:.1f}% (the strict own-slot share is 87.7%; the remainder mean also matches redundant sweeps)")
    print("  codec / variant     | channels in | C   | held-out | train | gap  | held-out median: " + ", ".join(rt.TOL) + " | violating % per field")
    for name, variant, C in (("abs", "ae", 64), ("delta", "ae", 64), ("phys", "ae", 64), ("fourier", "ae", 64),
                             ("abs", "ae", 128), ("delta", "ae", 128), ("phys", "ae", 128), ("delta", "skip", 32)):
        c = Codec(name)
        net = train_codec(c, variant, C, steps, tr)
        e, ok = eval_codec(c, net, te)
        _, ok_tr = eval_codec(c, net, tr)
        med = ", ".join(f"{np.median(e[k][torch.isfinite(e[k])].cpu().numpy()):.3f}" for k in rt.TOL)
        h, t = ok.float().mean().item(), ok_tr.float().mean().item()
        print(f"  {name + ' / ' + variant:19s} | {1 + (rt.K + 1) * (1 + c.P):11d} | {C:3d} | {100 * h:7.1f}% | {100 * t:4.1f}% | {100 * (t - h):+4.1f} | {med} | {fail_share(e)}", flush=True)


def level_f(steps, scale=0.25):
    """tight tolerances (all x scale) and a second seed: does the codec still matter once the probe can pass the identity?"""
    global R_TOL, PSI_TOL, DOP_TOL, RCS_TOL, AGE_TOL, OFF_TOL
    R_TOL, PSI_TOL, DOP_TOL, RCS_TOL, AGE_TOL, OFF_TOL = (v * scale for v in (R_TOL, PSI_TOL, DOP_TOL, RCS_TOL, AGE_TOL, OFF_TOL))
    for k in rt.TOL:
        rt.TOL[k] *= scale
    print(f"F  tolerances x {scale}: " + ", ".join(f"{k} {v:.3f}" for k, v in rt.TOL.items()) + f"; probe leaky+id, C = 64, {steps} steps")
    tr, te = rt.load(set(range(8))), rt.load({8, 9})
    print("  codec / seed        | oracle   | held-out | train | gap  | held-out median: " + ", ".join(rt.TOL) + " | violating % per field")
    for name, seed in (("abs", 0), ("delta", 0), ("phys", 0), ("fourier", 0), ("abs", 1), ("delta", 1)):
        c = Codec(name)
        _, ok_or = eval_codec(c, None, te)
        net = train_codec(c, "leaky+id", 64, steps, tr, seed=seed)
        e, ok = eval_codec(c, net, te)
        _, ok_tr = eval_codec(c, net, tr)
        med = ", ".join(f"{np.median(e[k][torch.isfinite(e[k])].cpu().numpy()):.3f}" for k in rt.TOL)
        h, t = ok.float().mean().item(), ok_tr.float().mean().item()
        print(f"  {name + ' / seed ' + str(seed):19s} | {100 * ok_or.float().mean().item():7.1f}% | {100 * h:7.1f}% | {100 * t:4.1f}% | {100 * (t - h):+4.1f} | {med} | {fail_share(e)}", flush=True)


def _ok_at(e, scale):
    """share of detections with all six fields within the nominal tolerances x scale."""
    ok = torch.ones_like(e["pos (m)"], dtype=torch.bool)
    for k, t in rt.TOL.items():
        ok &= e[k] <= t * scale
    return ok.float().mean().item()


def level_cv(steps, seeds=(0, 1), scales=(1.0, 0.25), names=("abs", "delta", "phys"), out="temp/radar/cv_result.json"):
    """Standard treatment of seed noise and small data: K-fold cross-validation over SCENES (5 folds of 2 scenes; each scene is held
    out once), every fold with 2 seeds, the same folds and seeds for every codec (paired). Reported per codec: held-out kept share,
    kept share / oracle (decoding the raster itself, the ceiling of that fold), and the overfitting EXCESS = (train - held-out of the
    probe) - (train - held-out of the oracle); the oracle's own difference is a property of the data. Comparison of two codecs: the
    paired difference of the fold means (seeds averaged), its mean and SD over the 5 folds, wins, and a t statistic (df 4). Rule
    (written before the run): A is better than B if the mean paired difference of kept / oracle exceeds the noise band
    max(0.02, mean seed SD) AND t > 2.776 (95 %, two-sided); otherwise TIE; overfitting is flagged if the excess exceeds 3 points."""
    import json
    import time

    torch.set_num_threads(2)
    per_scene = {s: rt.load({s}) for s in range(10)}
    folds = [(2 * i, 2 * i + 1) for i in range(5)]
    R, t0 = {}, time.time()
    print(f"CV  5 folds of 2 scenes x {len(seeds)} seeds x codecs {names}; probe leaky + identity start, C = 64, {steps} steps; tolerance scales {scales}")
    for fi, fold in enumerate(folds):
        te = [f for s in fold for f in per_scene[s]]
        tr = [f for s in range(10) if s not in fold for f in per_scene[s]]
        for name in names:
            c = Codec(name)
            e_or_te, e_or_tr = eval_codec(c, None, te)[0], eval_codec(c, None, tr)[0]
            for seed in seeds:
                net = train_codec(c, "leaky+id", 64, steps, tr, seed=seed)
                e_te, e_tr = eval_codec(c, net, te)[0], eval_codec(c, net, tr)[0]
                R[(name, fi, seed)] = {sc: dict(te=_ok_at(e_te, sc), tr=_ok_at(e_tr, sc), or_te=_ok_at(e_or_te, sc), or_tr=_ok_at(e_or_tr, sc)) for sc in scales}
                r = R[(name, fi, seed)][scales[-1]]
                print(f"  fold {fi} scenes {fold} {name:6s} seed {seed}: tol x{scales[-1]}: held-out {100 * r['te']:.1f}% train {100 * r['tr']:.1f}% oracle "
                      f"{100 * r['or_te']:.1f}% / {100 * r['or_tr']:.1f}%   ({(time.time() - t0) / 60:.0f} min)", flush=True)
    with open(out.replace(".json", "_raw.json"), "w", encoding="utf-8") as f:              # raw numbers first: the statistics below cannot lose them
        json.dump({f"{n}|fold{fi}|seed{s}": v for (n, fi, s), v in R.items()}, f, indent=1)
    summary = {}
    for sc in scales:
        print(f"\ntolerances x {sc}")
        print("  codec  | held-out kept (mean +- SD, 10 runs) | kept / oracle (mean +- SD) | seed SD (within fold) | overfitting excess (mean, worst fold)")
        ratio, seed_sd = {}, {}
        for name in names:
            te = np.array([[R[(name, fi, s)][sc]["te"] for s in seeds] for fi in range(5)])
            orc = np.array([[R[(name, fi, s)][sc]["or_te"] for s in seeds] for fi in range(5)])
            exc = np.array([[(R[(name, fi, s)][sc]["tr"] - R[(name, fi, s)][sc]["te"]) - (R[(name, fi, s)][sc]["or_tr"] - R[(name, fi, s)][sc]["or_te"])
                             for s in seeds] for fi in range(5)])
            ratio[name] = (te / orc).mean(1)                                  # per fold, seeds averaged
            seed_sd[name] = float(np.mean(np.std(te / orc, axis=1, ddof=1))) if len(seeds) > 1 else 0.0
            summary[f"{name}@{sc}"] = dict(kept=float(te.mean()), kept_sd=float(te.std(ddof=1)), ratio=float((te / orc).mean()), excess=float(exc.mean()))
            print(f"  {name:6s} | {100 * te.mean():5.1f}% +- {100 * te.std(ddof=1):4.1f}                   | {100 * (te / orc).mean():5.1f}% +- {100 * (te / orc).std(ddof=1):4.1f}"
                  f"        | {100 * seed_sd[name]:4.1f} points          | {100 * exc.mean():+4.1f} / {100 * exc.max():+4.1f} points"
                  + ("   <- OVERFITTING" if exc.mean() > 0.03 else ""))
        for a, b in (("delta", "abs"), ("phys", "abs"), ("phys", "delta")):
            if a not in names or b not in names:
                continue
            d = ratio[a] - ratio[b]
            band = max(0.02, seed_sd[a], seed_sd[b])
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
            verdict = f"{a} BETTER" if (d.mean() > band and t > 2.776) else (f"{b} BETTER" if (-d.mean() > band and -t > 2.776) else "TIE")
            summary[f"{a}-{b}@{sc}"] = dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), wins=int((d > 0).sum()), t=float(t), band=band, verdict=verdict)
            print(f"  paired {a} - {b}: kept/oracle difference per fold {np.round(100 * d, 1).tolist()} points, mean {100 * d.mean():+.1f} +- {100 * d.std(ddof=1):.1f}, "
                  f"wins {int((d > 0).sum())}/5, t = {t:.1f}, band {100 * band:.1f} -> {verdict}")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"steps": steps, "seeds": list(seeds), "summary": summary}, f, indent=1)
    print(f"\nresult file: {out}")
    return summary


def level_e(steps, names=("abs", "delta")):
    print(f"E  architecture of the probe (activation, identity start), codecs {names}, C = 64, {steps} steps")
    tr, te = rt.load(set(range(8))), rt.load({8, 9})
    print("  codec / variant     | channels in | C   | held-out | train | gap  | held-out median: " + ", ".join(rt.TOL) + " | violating % per field")
    for name in names:
        for variant in ("ae", "leaky", "relu+id", "leaky+id"):
            c = Codec(name)
            net = train_codec(c, variant, 64, steps, tr)
            e, ok = eval_codec(c, net, te)
            _, ok_tr = eval_codec(c, net, tr)
            med = ", ".join(f"{np.median(e[k][torch.isfinite(e[k])].cpu().numpy()):.3f}" for k in rt.TOL)
            h, t = ok.float().mean().item(), ok_tr.float().mean().item()
            print(f"  {name + ' / ' + variant:19s} | {1 + (rt.K + 1) * (1 + c.P):11d} | {64:3d} | {100 * h:7.1f}% | {100 * t:4.1f}% | {100 * (t - h):+4.1f} | {med} | {fail_share(e)}", flush=True)


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 2000
    if "cv" in sys.argv[2:]:
        level_cv(steps, seeds=tuple(range(int(sys.argv[3]) if len(sys.argv) > 3 else 2)))
    elif "f" in sys.argv[2:]:
        level_f(steps)
    elif "e" in sys.argv[2:]:
        level_e(steps)
    else:
        level_d(steps)
    print("RADAR CODEC GATE: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
    sys.exit(1 if fails else 0)
