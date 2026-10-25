"""Loop 16 (derived from p14_chain_data.py): data for the planner-side consumer on the output of a BUILDER run; the verifier probabilities of the E94 chain are the builder validity per piece.
Original description: Loop 14 / E117: data for a planner-side consumer of the LEARNED chain (system B).
Per frame of the traced graph (trace network E94a, tokens of p10_tokens.py): edge tokens (the 6 m edges inside the pieces: start / middle / end point), the mean step token of the two end vertices, the probabilities of the piece verifier
(E116, one per piece) and of the junction verifier (E113c, minimum over the junctions that touch the piece), the ego motion of the frame, the driven-path targets. Per SAMPLE (frame x map cell): the route probability of the route head (E104b) on the vertices, averaged over the two ends of every edge.
Cells: train = random draw per sample (right map with pose error sigma in {0, 1, 2, 3} m, wrong map, no map); test = exact / s2 / wrong / none for every frame.
Usage (gcar, GPU): p14_chain_data.py --out chain_a   (writes ~/exp_gcar_b123/p14/<out>_<split>.pt)"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p10_net as PN  # noqa: E402
import p13_jhead as JH  # noqa: E402
import p13_phead as PH  # noqa: E402
import p8_lib as L  # noqa: E402
import sd_lib as SL  # noqa: E402

HOME = os.path.expanduser("~")
NT = 320
SG = np.arange(5, 41, 5)
NMAXV = int(os.environ.get("NMAXV", 320))
CELLS_TEST = {"exact": ("n", 0.0), "s2": ("n", 2.0), "wrong": ("w", 2.0), "none": ("0", 0.0)}


def edge_tokens(it):
    """edges of the pieces (start / end vertex ids), the piece id and the geometry; truncated to the NT edges nearest to the ego"""
    ia, ib, pid = [], [], []
    for k, path in enumerate(it["pieces"]):
        for a, b in zip(path[:-1], path[1:]):
            ia.append(int(a))
            ib.append(int(b))
            pid.append(k)
    ia, ib, pid = np.array(ia, int), np.array(ib, int), np.array(pid, int)
    xy = it["xy"].astype(np.float32)
    if len(ia) > NT:
        pm = 0.5 * (xy[ia] + xy[ib])
        keep = np.sort(np.argsort(np.linalg.norm(pm, axis=1))[:NT])
        ia, ib, pid = ia[keep], ib[keep], pid[keep]
    return ia, ib, pid


def ego_of(P, r):
    prev = int(P["prev"][r])
    if prev < 0:
        return np.zeros(4, np.float32)
    dt = max(float(P["dt"][r]), 0.2)
    v = P["R"][r].T @ (P["t"][r] - P["t"][prev]) / dt
    Rrel = P["R"][prev].T @ P["R"][r]
    yaw = float(np.arctan2(Rrel[1, 0], Rrel[0, 0])) / dt
    return np.array([v[0] / 10, v[1] / 10, yaw * 5, 1.0], np.float32)


def targets(rt):
    y = np.zeros((len(SG), 2), np.float32)
    v = np.zeros(len(SG), bool)
    curved = False
    if rt is None:
        return y, v, curved
    p = np.asarray(rt["path"], np.float64)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    for j, sg in enumerate(SG):
        if cs[-1] >= sg:
            y[j] = [np.interp(sg, cs, p[:, 0]), np.interp(sg, cs, p[:, 1])]
            v[j] = True
    if v[5]:
        ang = np.degrees(np.arctan2(y[5][0] - y[0][0], y[5][1] - y[0][1]))
        curved = abs(ang) > 12.0
    return y, v, bool(curved)


def verifier_probs(items, dev):
    """the builder's validity replaces the piece and junction verifiers: mean validity of the vertices of a piece (junction value = the same)"""
    pp = []
    for it in items:
        kp = np.asarray(it.get("keep", np.ones(len(it["xy"]))), np.float32)
        pp.append(np.array([float(np.mean(kp[np.asarray(p, int)])) if len(p) else 1.0 for p in it["pieces"]], np.float32))
    return pp, [x.copy() for x in pp]


def frame_static(items, part, dev):
    """static tensors of the frames: geo [F,NT,6], tok [F,NT,128] fp16, pp / jp [F,NT], ia / ib [F,NT] (vertex ids, -1 = padding), mask, ego, y, v, curved, row"""
    F = len(items)
    P = L.cache("poses", part)
    rt = L.cache("rt", part)
    pp_all, jp_all = verifier_probs(items, dev)
    geo = np.zeros((F, NT, 6), np.float32)
    tok = np.zeros((F, NT, 128), np.float16)
    pp = np.zeros((F, NT), np.float32)
    jp = np.zeros((F, NT), np.float32)
    ia_ = -np.ones((F, NT), np.int16)
    ib_ = -np.ones((F, NT), np.int16)
    mask = np.zeros((F, NT), bool)
    ego = np.zeros((F, 4), np.float32)
    ys = np.zeros((F, len(SG), 2), np.float32)
    vs = np.zeros((F, len(SG)), bool)
    cu = np.zeros(F, bool)
    rows = np.zeros(F, int)
    scene = np.zeros(F, int)
    for k, it in enumerate(items):
        ia, ib, pid = edge_tokens(it)
        m = len(ia)
        xy = it["xy"].astype(np.float32)
        t = it["tok"].astype(np.float32) * it["has"].astype(np.float32)[:, None]
        if m:
            geo[k, :m] = np.concatenate([xy[ia], 0.5 * (xy[ia] + xy[ib]), xy[ib]], 1) / 50.0
            tok[k, :m] = (0.5 * (t[ia] + t[ib])).astype(np.float16)
            pp[k, :m], jp[k, :m] = pp_all[k][pid], jp_all[k][pid]
            ia_[k, :m], ib_[k, :m] = ia, ib
            mask[k, :m] = True
        r = int(it["row"])
        rows[k] = r
        scene[k] = int(P["scene"][r])
        ego[k] = ego_of(P, r)
        ys[k], vs[k], cu[k] = targets(rt[r])
    return dict(geo=torch.from_numpy(geo), tok=torch.from_numpy(tok), pp=torch.from_numpy(pp), jp=torch.from_numpy(jp), ia=torch.from_numpy(ia_.astype(np.int64)), ib=torch.from_numpy(ib_.astype(np.int64)), mask=torch.from_numpy(mask), ego=torch.from_numpy(ego), y=torch.from_numpy(ys), v=torch.from_numpy(vs), curved=torch.from_numpy(cu), row=torch.from_numpy(rows), scene=torch.from_numpy(scene))


class CellData(torch.utils.data.Dataset):
    """one item = (frame index, cell) -> tokens of the vertices and the SD raster of the cell"""

    def __init__(self, items, part, samples):
        self.items, self.part, self.samples = items, part, samples  # samples: list of (frame k, kind, sigma, drop, salt)
        self.sdl = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        if self.sdl is None:
            self.sdl = L.cache("sd", self.part)
        k, kind, sigma, drop, salt = self.samples[i]
        it = self.items[k]
        row = int(it["row"])
        sd = self.sdl[row] if kind != "w" else self.sdl[(row + 397) % len(self.sdl)]
        if sd is None or kind == "0":
            ras = np.zeros((5, 200, 200), np.float32)
        else:
            sdm, _, _ = L.draw_sd(sd, row, sigma, drop, salt)
            ras = SL.raster_sd(sdm)
        n = min(len(it["xy"]), NMAXV)
        xy = np.zeros((NMAXV, 2), np.float32)
        tok = np.zeros((NMAXV, 128), np.float32)
        cp = np.zeros((NMAXV, 7, 2), np.float32)
        mk = np.zeros(NMAXV, bool)
        xy[:n], tok[:n], cp[:n], mk[:n] = it["xy"][:n], it["tok"][:n], it["cp"][:n], True
        return dict(xy=torch.from_numpy(xy), tok=torch.from_numpy(tok), cp=torch.from_numpy(cp), mask=torch.from_numpy(mk), sd=torch.from_numpy(ras.astype(np.float16)), k=k)


@torch.no_grad()
def route_edge_probs(net, ds, static, dev, bs=16, workers=6):
    """-> [S, NT] route probability per edge (mean of the two end vertices), fp16"""
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)
    out = torch.zeros(len(ds), NT, dtype=torch.float16)
    s0 = 0
    for b in dl:
        bb = {k: v.to(dev) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            lg = net(bb["tok"], bb["xy"], bb["cp"], bb["sd"].float(), bb["mask"])
        p = torch.sigmoid(lg.float()).cpu()  # [B, NMAXV]
        k = b["k"]
        for j in range(len(k)):
            ia, ib = static["ia"][k[j]], static["ib"][k[j]]
            ok = ia >= 0
            ok = ok & (ia < NMAXV) & (ib < NMAXV)
            e = torch.zeros(NT)
            e[ok] = 0.5 * (p[j][ia[ok]] + p[j][ib[ok]])
            out[s0 + j] = e.half()
        s0 += len(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="chain_a")
    ap.add_argument("--tok", required=True, help="token file prefix of the builder run (<tok>_train / _val_clean / _test_clean)")
    ap.add_argument("--route", required=True, help="route head run name (out/loop2/<route>/model.pt, RouteXNet)")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--splits", default="test,val,train", help="loop 16b: 'val,train' keeps the test frames untouched")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    dev = "cuda"
    d = f"{HOME}/exp_gcar_b123/p14"
    os.makedirs(d, exist_ok=True)
    net = PN.RouteXNet().to(dev)
    net.load_state_dict(torch.load(f"{L.OUT}/{a.route}/model.pt", map_location=dev))
    net.eval()
    t0 = time.time()
    for split, files in (("test", [f"{a.tok}_test_clean"]), ("val", [f"{a.tok}_val_clean"]), ("train", [f"{a.tok}_train"])):
        if split not in a.splits.split(","):
            continue
        items = []
        for f in files:  # NOTE: tok_e94_a_oof0/1/2 are NOT out-of-fold for the trace network (only the dense-head inputs were cross-fitted; the trace network saw these rows): the train tokens are in-sample
            z = pickle.load(open(f"{HOME}/exp_gcar_b123/p10/{f}.pkl", "rb"))
            part = z["part"]
            items += z["items"]
        static = frame_static(items, part, dev)
        print(f"{split}: {len(items)} frames, static built ({time.time() - t0:.0f} s); frames with a driven path of >= 20 m {int(static['v'][:, 3].sum())}", flush=True)
        samples, meta = [], []
        if split == "train":
            rng = np.random.RandomState(0)
            for k in range(len(items)):
                if not static["v"][k].any():
                    continue
                for j in range(a.draws):
                    u = rng.rand()
                    kind = "w" if u < 0.15 else ("0" if u < 0.30 else "n")
                    sigma = float(rng.choice([0.0, 1.0, 2.0, 3.0]))
                    drop = 0.3 if rng.rand() < 0.3 else 0.0
                    samples.append((k, kind, sigma, drop, int(rng.randint(1, 1 << 20))))
                    meta.append(dict(cell={"n": f"n{int(sigma)}", "w": "wrong", "0": "none"}[kind]))
        else:
            for k in range(len(items)):
                if not static["v"][k].any():
                    continue
                for cname, (kind, sigma) in CELLS_TEST.items():
                    samples.append((k, kind, sigma, 0.0, 7 + k))
                    meta.append(dict(cell=cname))
        ds = CellData(items, part, samples)
        p = route_edge_probs(net, ds, static, dev, workers=a.workers)
        sample_k = torch.tensor([s[0] for s in samples])
        cell = [m["cell"] for m in meta]
        torch.save(dict(static=static, p=p, k=sample_k, cell=cell, split=split), f"{d}/{a.out}_{split}.pt")
        print(f"{split}: {len(samples)} samples written ({time.time() - t0:.0f} s)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
