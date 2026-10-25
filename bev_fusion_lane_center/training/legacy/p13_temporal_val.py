"""Loop 11 / E105: temporal fusion of the BEV input in front of the learned trace network (training-free, differentiable: warp + convex combination).
Protocol of E79 (TEST scenes, rows in time order; the first half of every scene uses the clean features, the second half the features of one perturbation, ego poses known).
The input of the trace network is the 42-channel stack (decoder features 32, map logits 7, heat 1, direction 2). Variants (memory = fused input of the previous frame, warped by the ego motion):
  single               no fusion
  ema_all:a            all 42 channels   y = a * warp(memory) + (1 - a) * x   (direction channels rotated with the relative yaw)
  ema_maps:a           only heat + direction are fused, features of the frame kept
  hold_mon             frames flagged by the label-free monitor are replaced by the warped memory
  ema_hold_mon:a       ema_all, flagged frames replaced by the memory
Monitor (no labels): mean heat of the frame below par_thr x median of the last 4 unflagged frames (the first 3 frames of a scene are never flagged).
Usage (gcar, GPU): p11_temporal.py --name e94_a --perts dark,cam_front3_off --variants single,ema_all:0.5 --out e105_a [--max_scenes N]"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p8_lib as L  # noqa: E402
import p10_infer as PI  # noqa: E402
import p10_net as PN  # noqa: E402

OUT, D, DEV = L.OUT, L.D, "cuda"
_XS = None
C_HEAT, C_DIR = 39, (40, 41)
ODO = [0.0, 0.0]  # odometry error (deg per sqrt(s), fraction of the distance); set by --odo / --odo_t
REFN = [4]  # number of unflagged frames in the reference of the heat monitor; set by --ref_n
RAMP = [0]  # gradual onset of the failure over this many frames (0 = abrupt, as E79 / E105); set by --ramp
MON = ["heat"]  # monitor signal: heat (mean heat of the frame), fwd (mean heat of the forward half), quad (minimum over the four quadrants); set by --mon


def monitor_signal(x, mode):
    h = x[C_HEAT].float()
    if mode[0] == "fwd":
        return float(h[100:].mean())
    if mode[0] == "quad":
        return float(min(h[:100, :100].mean(), h[:100, 100:].mean(), h[100:, :100].mean(), h[100:, 100:].mean()))
    return float(h.mean())


def make_grid(Rp, tp, Rc, tc):
    """sampling grid: pixel of the current frame -> position in the previous frame (p_prev = Rp^T (Rc p_cur + tc - tp))."""
    global _XS
    if _XS is None:
        xs = (torch.arange(200, device=DEV, dtype=torch.float64) + 0.5) * 0.5 - 50.0
        _XS = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
    f = lambda a: torch.as_tensor(np.asarray(a, np.float64), device=DEV)
    pw = _XS @ f(Rc).T + f(tc)
    pp = ((pw - f(tp)) @ f(Rp)).reshape(200, 200, 2).float()
    return torch.stack([pp[..., 1] / 50.0, pp[..., 0] / 50.0], -1)[None]


def warp(M, grid):
    m = M[None].float()
    w = F.grid_sample(m, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    v = F.grid_sample(torch.ones_like(m[:, :1]), grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0] > 0.99
    return w[0], v


class PertReader:
    def __init__(self, pn, mean, std):
        self.dec = np.load(f"{D}/pert/{pn}/val_dec32.npy", mmap_mode="r")
        self.lg = np.load(f"{D}/pert/{pn}/val_logit7.npy", mmap_mode="r")
        rows = np.load(f"{D}/pert/{pn}/val_rows.npy")
        self.j = {int(r): i for i, r in enumerate(rows)}
        d = f"{OUT}/e57_{pn}_valset"
        assert (np.load(f"{d}/ids_val.npy") == rows).all()
        self.prob, self.dirf = np.load(f"{d}/prob_val.npy", mmap_mode="r"), np.load(f"{d}/dir_val.npy", mmap_mode="r")
        self.mean, self.std = mean, std

    def __call__(self, row):
        j = self.j[row]
        x = np.concatenate([np.asarray(self.dec[j], np.float32), np.asarray(self.lg[j], np.float32)], 0)
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        return torch.from_numpy(np.concatenate([x, np.asarray(self.prob[j], np.float32)[None], np.asarray(self.dirf[j], np.float32)], 0)).to(DEV)


def frame_tokens(f, g, row):
    """graph tokens of a traced frame (as p10_tokens.py): vertex positions, step tokens, cross-section positions, pieces"""
    n = len(f.xy)
    xy = np.array(f.xy, np.float32).reshape(n, 2)
    tok = np.zeros((n, 128), np.float16)
    cp = np.zeros((n, 7, 2), np.float32)
    has = np.zeros(n, bool)
    for nd, (t, q, c) in f.node_tok.items():
        tok[nd], cp[nd], has[nd] = t, c, True
    return dict(row=int(row), xy=xy, tok=tok, has=has, cp=cp, pieces=[list(map(int, p)) for p in g["nodes"]], edges=sorted(f.edges))


class Stub:
    def __init__(self, tensors):
        self.t = tensors

    def batch(self, ks, dev="cuda"):
        return torch.stack([self.t[k] for k in ks]).float().to(dev)


def fuse_scene(rows, read, f0, P, var, a, thr=0.6):
    """-> list of fused inputs [42,200,200] (GPU, fp16), list of monitor flags, list of heat masses"""
    out, flags, masses, ref = [], [], [], []
    mem = prev = None
    for f, r in enumerate(rows):
        wgt = 0.0 if f < f0 else (min(1.0, (f - f0 + 1) / RAMP[0]) if RAMP[0] > 0 else 1.0)  # gradual onset (E108): blend of the clean and the perturbed 42-channel stack
        if wgt == 0.0 or wgt == 1.0:
            x = read(r, wgt == 1.0)
        else:
            x = (1 - wgt) * read(r, False) + wgt * read(r, True)
        mass = monitor_signal(x, MON)
        rf = np.median(ref[-REFN[0]:]) if len(ref) >= 3 else None  # reference of the heat monitor: median of the last REFN unflagged frames (4 in E105; a longer window, E108b, does not follow a slow decline)
        fl = rf is not None and mass < thr * rf
        if var.endswith("_oracle"):  # upper bound of the hold: flag exactly the frames of the failure phase (at least half perturbed)
            fl = wgt >= 0.5
        flags.append(bool(fl))
        masses.append(mass)
        if not fl:
            ref.append(mass)
        y = x
        if var != "single" and mem is not None:
            Rp, tp, Rc, tc = P["R"][prev], P["t"][prev], P["R"][r], P["t"][r]
            if ODO[0] > 0 or ODO[1] > 0:  # odometry error of the relative pose (as E79): yaw N(0, odo deg * sqrt(dt)), translation N(0, odo_t * distance) per axis, independent per frame pair
                import zlib

                import assoc_lib as AL

                rg = np.random.RandomState((zlib.crc32(f"odo{r}".encode()) + 5) % (2 ** 32))
                dt_ = max(float(P["ts"][r] - P["ts"][prev]), 0.1)
                dist = float(np.linalg.norm(P["t"][r] - P["t"][prev]))
                Rc = Rc @ AL.rot2(rg.randn() * ODO[0] * np.sqrt(dt_))
                tc = tc + rg.randn(2) * ODO[1] * dist
            w, valid = warp(mem, make_grid(Rp, tp, Rc, tc))
            rel = torch.as_tensor(np.asarray(Rc, np.float64).T @ np.asarray(Rp, np.float64), device=DEV, dtype=torch.float32)
            w = w.clone()
            w[40:42] = (rel @ w[40:42].reshape(2, -1)).reshape(2, 200, 200)
            chans = slice(0, 42) if var in ("ema_all", "hold_mon", "ema_hold_mon", "hold_oracle", "ema_hold_oracle") else slice(39, 42)
            if var == "maxdecay":  # E112, spatial hold: per pixel the decayed memory replaces the current frame where its heat is higher (a = decay per frame); no frame-level monitor
                wd = w.clone()
                wd[C_HEAT] = a * wd[C_HEAT]
                use = valid & (wd[C_HEAT] > x[C_HEAT])
                y = torch.where(use[None], wd, x)
            elif var in ("hold_mon", "ema_hold_mon", "hold_oracle", "ema_hold_oracle") and fl:
                y = torch.where(valid[None], w, x)
            elif var in ("ema_all", "ema_maps", "ema_hold_mon", "ema_hold_oracle"):
                y = x.clone()
                y[chans] = torch.where(valid[None], a * w[chans] + (1 - a) * x[chans], x[chans])
        out.append(y.half())
        mem, prev = y.float(), r
    return out, flags, masses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="e94_a")
    ap.add_argument("--perts", default="dark,cam_front3_off,blur,radar_off")
    ap.add_argument("--variants", default="single,ema_all:0.5,ema_all:0.7,ema_maps:0.5,ema_maps:0.7,hold_mon,ema_hold_mon:0.5")
    ap.add_argument("--out", default="e105_a")
    ap.add_argument("--par_thr", type=float, default=0.6)
    ap.add_argument("--mon", default="heat", help="monitor signal: heat | fwd | quad (exploratory variants of E105)")
    ap.add_argument("--min_frames", type=int, default=8)
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--odo", type=float, default=0.0, help="odometry yaw error of the relative pose, deg per sqrt(s)")
    ap.add_argument("--odo_t", type=float, default=0.0, help="odometry translation error as a fraction of the travelled distance")
    ap.add_argument("--tokens", action="store_true", help="store the graph tokens of every frame (for the navigation head under failure, E107)")
    ap.add_argument("--ref_n", type=int, default=4, help="E108b: number of unflagged frames in the reference of the heat monitor (4 = E105)")
    ap.add_argument("--ramp", type=int, default=0, help="E108: gradual onset, the perturbed stack is blended in linearly over this many frames (0 = abrupt)")
    a = ap.parse_args()
    ODO[0], ODO[1] = a.odo, a.odo_t
    RAMP[0] = a.ramp
    REFN[0] = a.ref_n
    MON[0] = a.mon
    os.makedirs(f"{OUT}/{a.out}", exist_ok=True)
    net = PN.TraceNet(use_cov=True, use_ptr=True, use_range=True).to(DEV)
    sd_ = torch.load(f"{OUT}/{a.name}/model.pt", map_location=DEV)
    net.load_state_dict({k: v for k, v in sd_.items() if not k.startswith(("sd_cnn", "route_mlp"))} if a.tokens else sd_)
    net.nav = bool(a.tokens)  # keep the step tokens
    net.eval()
    es = PI.EvalSet("val_clean")
    P = dict(np.load(f"{D}/poses_val.npz"))
    kof = {int(r): k for k, r in enumerate(es.rows)}
    mean, std = es.mean, es.std
    wave_kw = dict(r_cov=1.2, r_head=1.2, four_heads=False, order="near")
    cache = {}
    t0 = time.time()
    for pn in a.perts.split(","):
        rd = PertReader(pn, mean, std)
        rows_p = [r for r in es.rows if r in rd.j]
        by = {}
        for r in rows_p:
            by.setdefault(int(P["scene"][r]), []).append(r)
        seqs = [sorted(v, key=lambda i: P["ts"][i]) for v in by.values() if len(v) >= a.min_frames]
        if a.max_scenes:
            seqs = seqs[: a.max_scenes]
        read = lambda r, fail, rd=rd: rd(r) if fail else es.batch([kof[r]], DEV)[0]
        for vs in a.variants.split(","):
            var, ap_ = (vs.split(":") + ["0"])[:2]
            ap_ = float(ap_)
            res = dict(rows=[], phase=[], scene=[], flag=[], mass=[], graphs=[])
            for rows in seqs:
                f0 = len(rows) // 2
                fused, flags, masses = fuse_scene(rows, read, f0, P, var, ap_, a.par_thr)
                ck = (vs, tuple(rows[:f0]))
                posmap = {r: i - f0 for i, r in enumerate(rows)}  # frame index relative to the onset of the failure
                tens = {kof[r]: fused[i] for i, r in enumerate(rows)}
                stub = Stub(tens)
                parts = []
                for ph, sl in (("cl", slice(0, f0)), ("fail", slice(f0, len(rows)))):
                    rr = rows[sl]
                    if ph == "cl" and ck in cache:
                        g, tk = cache[ck]
                    else:
                        g, _, frs = PI.run_eval(net, stub, [kof[r] for r in rr], bs=16, tau_seed=0.3, tau_on=0.5, k_seed=192, nms=5, wave_kw=wave_kw, verbose=False, return_frames=True)
                        tk = [frame_tokens(f_, gg, r) for f_, gg, r in zip(frs, g, rr)] if a.tokens else [None] * len(rr)
                        if ph == "cl":
                            cache[ck] = (g, tk)
                    parts.append((ph, rr, g, flags[sl], masses[sl], tk))
                for ph, rr, g, fl, ms, tk in parts:
                    for r, gr, f_, m_, t_ in zip(rr, g, fl, ms, tk):
                        res["rows"].append(int(r))
                        res["phase"].append(ph)
                        res.setdefault("pos", []).append(int(posmap[r]))
                        res["scene"].append(int(P["scene"][r]))
                        res["flag"].append(bool(f_))
                        res["mass"].append(float(m_))
                        res["graphs"].append(gr)
                        res.setdefault("tok", []).append(t_)
                del fused, tens, stub
                torch.cuda.empty_cache()
            tag = f"{pn}_{vs.replace(':', '_')}"
            pickle.dump(res, open(f"{OUT}/{a.out}/graphs_{tag}.pkl", "wb"))
            nflag = [sum(f_ for f_, p_ in zip(res["flag"], res["phase"]) if p_ == ph) / max(sum(p_ == ph for p_ in res["phase"]), 1) for ph in ("cl", "fail")]
            print(f"[{pn}] {vs:16s} frames {len(res['rows'])} flagged clean {nflag[0]:.2f} fail {nflag[1]:.2f}  {time.time() - t0:.0f} s", flush=True)
    print("DONE", time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
