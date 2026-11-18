"""Loop 14 / E117: data for a planner-side consumer of the LEARNED chain (system B).
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
from p16_chain_data import edge_tokens, ego_of, targets  # noqa: E402,F401  (module-level functions that were identical copies and use only NT / SG, equal in both files)

HOME = os.path.expanduser("~")
NT = 320
SG = np.arange(5, 41, 5)
NMAXV = 320
CELLS_TEST = {"exact": ("n", 0.0), "s2": ("n", 2.0), "wrong": ("w", 2.0), "none": ("0", 0.0)}


def verifier_probs(items, dev):
    """piece probability and junction probability per piece, for all frames"""
    pnet = PH.PNet().to(dev)
    pnet.load_state_dict(torch.load(f"{L.OUT}/e113_a/pmodel.pt", map_location=dev))
    pnet.eval()
    jnet = JH.JNet().to(dev)
    jnet.load_state_dict(torch.load(f"{L.OUT}/e113_a/jmodel.pt", map_location=dev))
    jnet.eval()
    pp = [np.ones(len(it["pieces"]), np.float32) for it in items]
    jp = [np.ones(len(it["pieces"]), np.float32) for it in items]
    flat = []
    jflat = []
    for k, it in enumerate(items):
        edges = JH.derived_edges(it)
        for i, r in enumerate(PH.piece_feats(it, edges)):
            if r is not None:
                flat.append((k, i, r))
        for r in JH.junction_feats(it, edges):
            jflat.append((k, r))
    X, G = PH.stack([r for _, _, r in flat])
    with torch.no_grad():
        pv = torch.cat([torch.sigmoid(pnet(X[i : i + 4096].to(dev), G[i : i + 4096].to(dev))).cpu() for i in range(0, len(X), 4096)]).numpy()
    for (k, i, _), p_ in zip(flat, pv):
        pp[k][i] = p_
    if jflat:
        Xj, Gj = JH.stack([r for _, r in jflat])
        with torch.no_grad():
            pj = torch.cat([torch.sigmoid(jnet(Xj[i : i + 4096].to(dev), Gj[i : i + 4096].to(dev))).cpu() for i in range(0, len(Xj), 4096)]).numpy()
        for (k, r), p_ in zip(jflat, pj):
            for q in [r["c"]] + list(r["others"]):
                jp[k][q] = min(jp[k][q], float(p_))
    return pp, jp


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
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    dev = "cuda"
    d = f"{HOME}/exp_gcar_b123/p14"
    os.makedirs(d, exist_ok=True)
    net = PN.RouteXNet().to(dev)
    net.load_state_dict(torch.load(f"{L.OUT}/e104b/model.pt", map_location=dev))
    net.eval()
    t0 = time.time()
    for split, files in (("test", ["tok_e94_a_test_clean"]), ("val", ["tok_e94_a_val_clean"]), ("train", ["tok_e94_a_oof0", "tok_e94_a_oof1", "tok_e94_a_oof2"])):
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
