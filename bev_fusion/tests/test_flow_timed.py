"""Can ground truth at OTHER times generate the velocity? (Mirko, 2026-10-30: "build it simpler", "allow for training by
predicting bounding boxes and images into future and past frames with positive and negative delta t")

The simple recurrent model (plan 7.1 T2 again): a carried feature state h, moved to the next keyframe by the splat
(ego motion + the velocity of the previous step), updated per keyframe from the current map by a small U-Net with a
GRU gate (reach of +-14 m at 2 m cells: a 0.5 s keyframe gap moves a car up to ~8 m), and a velocity head on h:
    h_k-1, v_k-1 = U(0, E(F_k-1));   h_k, v_k = U(splat(h_k-1, ego, v_k-1), carry(v_k-1), E(F_k))
No hypotheses, no refinement, no gate. Trained on nuScenes GT keyframes k-1, k, k+1 (0.5 s apart); the velocity v_k at
keyframe k is trained ONLY by predicting other times with dt = -0.5 s (k-1) and +0.5 s (k+1):
  --boxes   every cell of every box at k must land where the SAME box (instance token) is at k-1 / k+1: the rigid
            motion of the box (centre and yaw from the world poses, in k's axes), loss |v dt - D| / |dt| (m/s), the
            box cells of each sample weighted to sum 1; cells outside every box are static (|v|, weight 0.1)
  --maps    the maps at k-1 / k+1 are predicted from k: decoder of h moved with v (BCE) + the raw map of k moved with
            v alone (MSE x 3, the explanation term of tests/test_flow_explains.py)
  --label   reference: the annotated velocity at k directly (the weights of --boxes; only the target differs)
  --fwd     only dt = +0.5 s (control: what the backward prediction adds)
  --corr    architecture arm: one local correlation layer at 2 m cells (E(F_k) against the previous E(F) moved to k,
            +-4 cells) into the U-Net bottom - the one piece of the flow block that measures displacement
Evaluation (held-out scenes, the selection of tests/test_flow_explains.py: a moving vehicle in k-1): end-point error
at k against the annotated velocity (vehicles all / moving / static, per box, pedestrians, lanes), raw map of k moved
with v to k+1 and to k-1 (IoU of vehicles, with GT velocity and v = 0 as references), and the recursion over the
cached held-out sequences of 5 keyframes (EPE per keyframe; the first keyframe has no past).
Run from the bev_fusion folder (the GT cache data/cache/gt must exist):
  wsl ~/miniconda3/envs/gcar/bin/python tests/test_flow_timed.py --boxes [--maps] [--label] [--fwd] [--quick] [--seed N]
"""
import glob
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "training")
sys.path.insert(0, "tests")
import scratch  # noqa: E402
from lift_pd import Grid  # noqa: E402
from temporal import propagate  # noqa: E402


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


QUICK, SEED = "--quick" in sys.argv, arg("--seed", 0)
BOXES, MAPS, LABEL, FWD = "--boxes" in sys.argv, "--maps" in sys.argv, "--label" in sys.argv, "--fwd" in sys.argv
CORR, CR = "--corr" in sys.argv, 4                   # + one local correlation layer at 2 m (+-4 cells = +-8 m)
assert BOXES or MAPS or LABEL, "choose a training signal: --boxes, --maps and / or --label"
TAG = ("_boxes" if BOXES else "") + ("_maps" if MAPS else "") + ("_label" if LABEL else "") + ("_fwd" if FWD else "") \
    + ("_corr" if CORR else "") + (f"_s{SEED}" if SEED else "") + ("_quick" if QUICK else "")
G = Grid(x_min=0.0, x_max=60.0, y_min=-26.0, y_max=26.0, res=0.5)            # 120 x 104, as data/cache/gt
DEV = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 3e9 else "cpu"
C = 32
HC = 2 * C if CORR else C                          # carried channels: the GRU state (+ the last encoding with --corr)


# ----------------------------------------------------------------------------------------------------------- data
def maps_of(f):
    lanes = np.maximum(f["lane_divider"], f["road_divider"]).astype(np.float32)
    veh = np.maximum(f["vehicle"], f["two_wheeler"]).astype(np.float32)
    return np.stack([lanes, veh, f["pedestrian"].astype(np.float32)])


def yaw_of(T):
    return math.atan2(T[1, 0], T[0, 0])


def box_motion(cur, other):
    """per-cell target displacement (2, X, Y) in m, k's axes, of every observed box of keyframe k to the pose of the same
    instance in the other keyframe (rigid: centre + yaw change), its mask, and the box index per cell (-1: none)."""
    T, To = cur["meta"]["T_e2g"], other["meta"]["T_e2g"]
    R, t = T[:3, :3], T[:3, 3]
    by_inst = {b["instance"]: b for b in other["meta"]["boxes"]}
    xs, ys = (c.numpy() for c in G.xy_centres())
    px, py = np.meshgrid(xs, ys, indexing="ij")
    inst = cur["instance"]
    D = np.zeros((2,) + inst.shape, np.float32)
    M = np.zeros(inst.shape, np.float32)
    for i, b in enumerate(cur["meta"]["boxes"]):
        cells = inst == i
        o = by_inst.get(b["instance"])
        if not cells.any() or o is None:
            continue
        c = np.asarray(b["center"])[:2]
        co = (R.T @ (np.asarray(o["world"]) - t))[:2]                   # the other pose in k's axes
        dyaw = (o["yaw"] + yaw_of(To)) - (b["yaw"] + yaw_of(T))
        cs, sn = math.cos(dyaw), math.sin(dyaw)
        qx, qy = px[cells] - c[0], py[cells] - c[1]
        D[0][cells] = co[0] + cs * qx - sn * qy - px[cells]
        D[1][cells] = co[1] + sn * qx + cs * qy - py[cells]
        M[cells] = 1.0
    return D, M


def item(path):
    """keyframes k-1, k, k+1 of one cached GT sample -> dict, or None without a moving vehicle in k-1 (the selection of
    tests/test_flow_explains.py). Steps: k -> k-1 (negative dt), k -> k+1, k-1 -> k (the forward chain)."""
    g = np.load(path, allow_pickle=True)["gt"].item()
    p, c, n = g["prev"], g["cur"], g["next"]
    if not ((maps_of(p)[1] > 0.5) & (np.hypot(*p["velocity"]) > 1.0)).any():
        return None
    Db, Mb = box_motion(c, p)
    Df, Mf = box_motion(c, n)
    static = ((c["instance"] < 0) & (c["ignore"] < 0.5)).astype(np.float32)
    return {"maps": torch.from_numpy(np.stack([maps_of(p), maps_of(c), maps_of(n)])),
            "vel": torch.from_numpy(c["velocity"].astype(np.float32)),
            "inst": torch.from_numpy(c["instance"].astype(np.int64)),
            "D": torch.from_numpy(np.stack([Db, Df])), "M": torch.from_numpy(np.stack([Mb, Mf])),
            "static": torch.from_numpy(static),
            "s_back": [tuple(map(float, s)) for s in p["meta"]["steps"]],
            "s_fwd": [tuple(map(float, s)) for s in n["meta"]["steps"]],
            "s_prev": [(-float(dt), float(v), float(w)) for dt, v, w in reversed(p["meta"]["steps"])],
            "dt": (float(p["meta"]["dt"]), float(n["meta"]["dt"]))}


def items(test, n):
    def build(m):
        out = []
        for f in sorted(glob.glob("data/cache/gt/*_0.5_4_*.npz")):
            scene = np.load(f, allow_pickle=True)["gt"].item()["cur"]["meta"]["scene"]
            if (int(scene[:8], 16) % 5 == 0) != test:
                continue
            it = item(f)
            if it is not None:
                out.append(it)
                if len(out) == m:
                    break
        return out
    # temp/checks/flow_timed_nusc_<split>_<n>.pt: a bigger cache serves a smaller n; one older than the GT cache folder (any GT file added or
    # replaced since) or built by another item() is deleted
    return scratch.cached("flow_timed_nusc", build, n, deps=("data/cache/gt",), sig=scratch.source_sig(item),
                          split="test" if test else "train")


# ---------------------------------------------------------------------------------------------------------- model
def rotate_into_next(v, steps):
    th = torch.tensor([sum(dt * w for dt, _, w in st) for st in steps], dtype=v.dtype, device=v.device).view(-1, 1, 1)
    c, s = torch.cos(th), torch.sin(th)
    return torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], 1)


def carry(v, steps):
    """a velocity field moved with itself to the next keyframe and rotated into its axes."""
    return rotate_into_next(propagate(v, steps, v, G, mass_gate=False)[0], steps)


def local_corr(a, b, r):
    """a, b (B, c, X, Y) -> (B, (2r+1)^2, X, Y): sum_c a(p) b(p + d) / c for the displacements d of a +-r window."""
    B, c, X, Y = a.shape
    ub = F.unfold(b, 2 * r + 1, padding=r).view(B, c, (2 * r + 1) ** 2, X * Y)
    return (a.flatten(2)[:, :, None] * ub).sum(1).view(B, -1, X, Y) / c ** 0.5


def block(cin, cout, dil=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=dil, dilation=dil), nn.GroupNorm(8, cout), nn.ReLU())


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(block(3, 32), nn.Conv2d(32, C, 3, padding=1))
        cin = 2 * C + 3                                                 # [h moved, mass, carried v, E(F)]
        self.d0 = block(cin, 32)                                        # 0.5 m
        self.d1 = nn.Sequential(nn.Conv2d(32, 48, 3, 2, 1), nn.ReLU(), block(48, 48))            # 1 m
        if CORR:      # 2 m: the correlation of E(F_k) with the previous E(F) moved to k, displacements +-4 cells (+-8 m)
            self.cproj = nn.Conv2d(C, 16, 1)
            self.d2a = nn.Sequential(nn.Conv2d(48, 64, 3, 2, 1), nn.ReLU())
            self.d2b = nn.Sequential(block(64 + (2 * CR + 1) ** 2, 64), block(64, 64, 2), block(64, 64, 4))
        else:
            self.d2 = nn.Sequential(nn.Conv2d(48, 64, 3, 2, 1), nn.ReLU(), block(64, 64), block(64, 64, 2),
                                    block(64, 64, 4))                                               # 2 m, reach +-14 m
        self.u1 = block(64 + 48, 48)
        self.u0 = block(48 + 32, 32)
        self.out = nn.Conv2d(32, 2 * C + 2, 1)
        self.dec = nn.Sequential(block(C + 1, 32), nn.Conv2d(32, 3, 1))

    def step(self, h, mass, vc, f):
        """h: the moved state, with --corr followed by the moved encoding of the previous map (HC channels)."""
        fe = self.enc(f)
        a = self.d0(torch.cat([h[:, :C], mass, vc / 10.0, fe], 1))
        b = self.d1(a)
        if CORR:
            def proj(x):
                return F.normalize(F.avg_pool2d(self.cproj(x), 4), dim=1)
            c = self.d2b(torch.cat([self.d2a(b), local_corr(proj(fe), proj(h[:, C:]), CR)], 1))
        else:
            c = self.d2(b)
        b = self.u1(torch.cat([F.interpolate(c, size=b.shape[-2:]), b], 1))
        a = self.u0(torch.cat([F.interpolate(b, size=a.shape[-2:]), a], 1))
        o = self.out(a)
        z, cand, v = torch.sigmoid(o[:, :C]), torch.tanh(o[:, C:2 * C]), 10.0 * o[:, 2 * C:]
        hn = (1 - z) * h[:, :C] + z * cand
        return (torch.cat([hn, fe], 1) if CORR else hn), v

    def first(self, f):
        B = f.shape[0]
        return self.step(f.new_zeros(B, HC, G.nx, G.ny), f.new_zeros(B, 1, G.nx, G.ny), f.new_zeros(B, 2, G.nx, G.ny), f)

    def next(self, h, v, steps, f):
        """carry (h, v) over the steps to the next keyframe (v detached: the path is no learning signal) and update."""
        hm, mass = propagate(h, steps, v.detach(), G)
        return self.step(hm, mass.clamp(max=1.0), carry(v.detach(), steps), f)

    def predict(self, h, v, steps):
        """the maps at the time the steps lead to, decoded from h moved with v."""
        hm, mass = propagate(h[:, :C], steps, v, G)
        return self.dec(torch.cat([hm, mass.clamp(max=1.0)], 1))


# ----------------------------------------------------------------------------------------------------------- train
def batch_of(data, idx):
    b = [data[i] for i in idx]
    t = {k: torch.stack([x[k] for x in b]).to(DEV) for k in ("maps", "vel", "D", "M", "static")}
    t.update({k: [x[k] for x in b] for k in ("s_back", "s_fwd", "s_prev", "dt")})
    return t


def run(model, t):
    m = t["maps"]
    h0, v0 = model.first(m[:, 0])
    return model.next(h0, v0, t["s_prev"], m[:, 1])


def loss_of(model, t, h, v):
    m, loss, parts = t["maps"], 0.0, {}
    dirs = [(1, "s_fwd", 2)] if FWD else [(0, "s_back", 0), (1, "s_fwd", 2)]
    if BOXES:
        for j, _, _ in dirs:
            dt = torch.tensor([d[j] for d in t["dt"]], device=DEV).view(-1, 1, 1, 1)
            M = t["M"][:, j:j + 1]
            err = (v * dt - t["D"][:, j]).norm(dim=1, keepdim=True) / dt.abs()
            w = M / M.sum((2, 3), keepdim=True).clamp(min=1.0)          # each sample's box cells sum to 1
            parts[f"box{j}"] = (w * err).sum() / m.shape[0]
            loss = loss + parts[f"box{j}"]
        st = t["static"][:, None]
        parts["static"] = 0.1 * (st * v.norm(dim=1, keepdim=True)).sum() / st.sum().clamp(min=1.0)
        loss = loss + parts["static"]
    if MAPS:
        for j, sk, ti in dirs:
            p = model.predict(h, v, t[sk])
            parts[f"map{j}"] = F.binary_cross_entropy_with_logits(p, m[:, ti]) \
                + 3.0 * ((propagate(m[:, 1], t[sk], v, G)[0] - m[:, ti]) ** 2).mean()
            loss = loss + parts[f"map{j}"]
    if LABEL:          # the same weights as --boxes (only the target differs): box cells sum to 1 per sample, static 0.1
        M = t["M"].amax(1, keepdim=True)
        w = M / M.sum((2, 3), keepdim=True).clamp(min=1.0)
        parts["label"] = (w * (v - t["vel"]).norm(dim=1, keepdim=True)).sum() / m.shape[0]
        st = t["static"][:, None]
        parts["lstatic"] = 0.1 * (st * v.norm(dim=1, keepdim=True)).sum() / st.sum().clamp(min=1.0)
        loss = loss + parts["label"] + (0.0 if BOXES else parts["lstatic"])
    return loss, parts


def train(model, data, iters, rng, bs=8, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=iters, pct_start=0.1)
    t0 = time.time()
    for it in range(iters):
        t = batch_of(data, [rng.randrange(len(data)) for _ in range(bs)])
        h, v = run(model, t)
        loss, parts = loss_of(model, t, h, v)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if it % max(1, iters // 10) == 0 or it == iters - 1:
            print(f"    iter {it:5d} loss {loss.item():.4f} (" + ", ".join(f"{k} {float(x):.3f}" for k, x in parts.items())
                  + f")  {time.time() - t0:.0f} s", flush=True)


# -------------------------------------------------------------------------------------------------------- evaluate
def iou(a, b, thr=0.5):
    a, b = a > thr, b > thr
    return (a & b).sum().item(), (a | b).sum().item()


@torch.no_grad()
def evaluate(model, test):
    epe, acc = {}, {}
    for x in test:
        t = batch_of([x], [0])
        m = t["maps"]
        h, v = run(model, t)
        vg, zero = t["vel"], torch.zeros_like(v)
        for name, sk, ti in (("fwd", "s_fwd", 2), ("back", "s_back", 0)):
            for vn, vv in (("v", v), ("GT v", vg), ("v=0", zero)):
                a = acc.setdefault(f"IoU raw {name} {vn}", [0, 0])
                i, u = iou(propagate(m[:, 1], t[sk], vv, G)[0][0, 1], m[0, ti, 1])
                a[0] += i
                a[1] += u
        vc, vgt, maps, inst = v[0].cpu(), x["vel"], x["maps"][1], x["inst"]
        e = (vc - vgt).norm(dim=0)
        veh = maps[1] > 0.5
        masks = {"vehicles": veh, "vehicles moving (> 1 m/s)": veh & (vgt.norm(dim=0) > 1.0),
                 "vehicles static (< 0.2 m/s)": veh & (vgt.norm(dim=0) < 0.2), "pedestrians": maps[2] > 0.5,
                 "lanes": (maps[0] > 0.05) & (maps[1:].max(0).values < 0.1)}
        for k, mk in masks.items():
            a = epe.setdefault(k, [0.0, 0])
            a[0] += e[mk].sum().item()
            a[1] += int(mk.sum())
        for k in inst.unique().tolist():
            cells = (inst == k) & veh
            if k >= 0 and cells.any():
                a = epe.setdefault("vehicles per box (mean v)", [0.0, 0])
                a[0] += (vc[:, cells].mean(1) - vgt[:, cells].mean(1)).norm().item()
                a[1] += 1
    res = {"EPE " + k: s / max(1, n) for k, (s, n) in epe.items()}
    res.update({k: i / max(1, u) for k, (i, u) in acc.items()})
    return res


@torch.no_grad()
def sequences(model, path="temp/checks/flow_sequences_nusc_test_24_5.pt"):
    """EPE per keyframe over the cached held-out sequences (tests/test_flow_explains.py --sequences): keyframe 0 has no
    past, every later one carries h and v."""
    if not os.path.exists(path):
        print(f"(no {path}: sequence evaluation skipped)")
        return None
    seqs = torch.load(path, weights_only=False)
    T = seqs[0]["maps"].shape[0]
    acc = {(k, t): [0.0, 0] for k in ("vehicles", "moving") for t in range(T)}
    for sq in seqs:
        m, vel = sq["maps"][None].to(DEV), sq["velocity"]
        h, v = model.first(m[:, 0])
        for t in range(T):
            if t > 0:
                h, v = model.next(h, v, [sq["steps"][t - 1]], m[:, t])
            veh = sq["maps"][t, 1] > 0.5
            e = (v[0].cpu() - vel[t]).norm(dim=0)
            for k, mk in (("vehicles", veh), ("moving", veh & (vel[t].norm(dim=0) > 1.0))):
                acc[(k, t)][0] += e[mk].sum().item()
                acc[(k, t)][1] += int(mk.sum())
    res = {key: s / max(1, n) for key, (s, n) in acc.items()}
    print(f"== recursion over {len(seqs)} held-out sequences of {T} keyframes (EPE m/s at keyframe t; t = 0 has no past)")
    print("  t   vehicles   moving vehicles")
    for t in range(T):
        print(f"  {t}   {res[('vehicles', t)]:6.2f}     {res[('moving', t)]:6.2f}", flush=True)
    return res


@torch.no_grad()
def picture(model, test, path, n=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from flow_viz import flow_rgb, wheel

    def overlay(a, b):
        oa, ob = a[1:].max(0).values.clamp(0, 1), b[1:].max(0).values.clamp(0, 1)
        lane = 0.35 * b[0].clamp(0, 1)
        return torch.stack([torch.maximum(oa, lane), torch.maximum(ob, lane), lane], -1).flip(1).numpy()
    cols = ["k (red) / k+1 (green)", "GT velocity at k", "learned velocity at k", "k moved with v (red) vs k+1",
            "k moved with v (red) vs k-1", "k moved with v = 0 (red) vs k+1"]
    fig, ax = plt.subplots(n, len(cols) + 1, figsize=(2.3 * (len(cols) + 1), 2.5 * n), squeeze=False)
    for r in range(n):
        x = test[r]
        t = batch_of([x], [0])
        m = t["maps"]
        _, v = run(model, t)
        mp = x["maps"]
        occ = (mp[1, 1:].max(0).values > 0.3).float()
        ims = [overlay(mp[1], mp[2]), flow_rgb(x["vel"], occ, 15.0), flow_rgb(v[0].cpu(), occ, 15.0),
               overlay(propagate(m[:, 1], t["s_fwd"], v, G)[0][0].cpu(), mp[2]),
               overlay(propagate(m[:, 1], t["s_back"], v, G)[0][0].cpu(), mp[0]),
               overlay(propagate(m[:, 1], t["s_fwd"], torch.zeros_like(v), G)[0][0].cpu(), mp[2])]
        for c, im in enumerate(ims):
            ax[r, c].imshow(im, origin="lower")
            ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=7)
        ax[r, -1].axis("off")
    wheel(ax[0, -1], vmax=15.0)
    fig.suptitle(f"trained by{TAG.replace('_', ' ')}; nuScenes keyframes 0.5 s apart; hue = direction, brightness = speed "
                 "(15 m/s full); x forward = up; red = moved, green = truth, yellow = both", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    t0 = time.time()
    n_train, n_test, iters = (300, 16, 600) if QUICK else (1500, 48, 3000)
    iters = arg("--iters", iters)
    data, test = items(False, n_train), items(True, n_test)
    print(f"device {DEV}, seed {SEED}, signals{TAG}; {len(data)} training samples, {len(test)} held-out, {iters} steps of 8 "
          f"({time.time() - t0:.0f} s to load)", flush=True)
    # layer 0: the box targets must explain the annotated velocity (central difference ~ the annotation)
    gm = [((x["D"][1] / x["dt"][1] + x["D"][0] / x["dt"][0]) / 2 - x["vel"]).norm(dim=0)[(x["M"].prod(0) > 0)
          & (x["maps"][1, 1] > 0.5)] for x in test]
    gm = torch.cat(gm)
    print(f"check: box-motion targets vs annotated velocity on vehicle cells: median {gm.median():.3f}, mean {gm.mean():.3f} m/s "
          f"({gm.numel()} cells)", flush=True)
    torch.manual_seed(SEED)
    model = Model().to(DEV)
    print(f"parameters {sum(p.numel() for p in model.parameters()) / 1e3:.0f} k", flush=True)
    train(model, data, iters, random.Random(SEED + 7))
    torch.save(model.state_dict(), f"temp/checks/model_timed{TAG}.pt")
    model.eval()
    r = evaluate(model, test)
    print("== held-out samples (EPE in m/s at keyframe k; IoU of vehicles)")
    for k, x in r.items():
        print(f"  {k:32s} {x:.3f}")
    sequences(model)
    claims = [("vehicles: EPE < 1 m/s per cell", r["EPE vehicles"] < 1.0),
              ("moving vehicles: EPE < 1.5 m/s per cell", r["EPE vehicles moving (> 1 m/s)"] < 1.5),
              ("raw move to k+1 reaches >= 90 % of the GT-velocity IoU", r["IoU raw fwd v"] >= 0.9 * r["IoU raw fwd GT v"]),
              ("raw move to k-1 reaches >= 90 % of the GT-velocity IoU", r["IoU raw back v"] >= 0.9 * r["IoU raw back GT v"])]
    print("== claims")
    for name, ok in claims:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    picture(model, test, f"temp/checks/flow_timed{TAG}.png")
    print(f"== {sum(ok for _, ok in claims)} / {len(claims)} claims hold; image temp/checks/flow_timed{TAG}.png "
          f"({time.time() - t0:.0f} s)")
