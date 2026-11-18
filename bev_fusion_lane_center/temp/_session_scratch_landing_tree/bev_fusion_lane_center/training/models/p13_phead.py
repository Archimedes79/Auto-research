"""Loop 13 / E116: a learned PIECE verifier for the traced graph of the learned chain (system B): "better short lane graphs than false lane graphs".
Block: for every piece of the traced graph an MLP reads the step tokens of the piece (mean, first and last vertex), the length, the number of vertices, the position and the degree in the graph, and says whether at least half of the piece lies within 1 m of the ground-truth centre lines.
A piece below the threshold is removed from the graph (with its edges). Optionally after the junction verifier of E113c (dissolve at 0.3).
Training on the tokens of the train rows traced by the trace network E94a (the dense-head inputs are out-of-fold, the trace network itself was trained on these rows, so the tokens are cleaner than those of VAL and TEST), threshold sweep on VAL, TEST reported.
Usage (gcar): p13_phead.py --out e113_a"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boot_eval as BE  # noqa: E402
import graph_eval as GE  # noqa: E402
import graph_from_dense as GD  # noqa: E402
import p11_eval as E  # noqa: E402
import p12_calib as C12  # noqa: E402
import p13_jhead as JH  # noqa: E402
import p8_lib as L  # noqa: E402

HOME = os.path.expanduser("~")
OUT = L.OUT
NG = 9


def piece_feats(it, edges, gt_pts=None):
    """one row per piece: token features, geometry, label (share within 1 m of the ground truth >= 0.5)"""
    xy = it["xy"].astype(np.float64)
    tok, has = it["tok"].astype(np.float32), it["has"].astype(np.float32)
    P = [np.asarray(p, int) for p in it["pieces"]]
    deg_o, deg_i = np.zeros(len(P)), np.zeros(len(P))
    for i, j in edges:
        deg_o[i] += 1
        deg_i[j] += 1
    tree = cKDTree(gt_pts) if gt_pts is not None and len(gt_pts) else None
    rows = []
    for k, p in enumerate(P):
        if len(p) == 0:
            rows.append(None)
            continue
        t = tok[p] * has[p][:, None]
        x = np.concatenate([t.mean(0), t[0], t[-1], [has[p].mean()]]).astype(np.float16)
        q = xy[p]
        length = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()) if len(p) > 1 else 0.0
        geo = [length / 50, len(p) / 20, float(has[p].mean()), q[0, 0] / 50, q[0, 1] / 50, q[-1, 0] / 50, q[-1, 1] / 50, deg_o[k] / 3, deg_i[k] / 3]
        share = None
        if gt_pts is not None:
            r = GE.resample(q, 1.0) if len(q) > 1 else q
            share = float((tree.query(r)[0] <= 1.0).mean()) if tree is not None else 0.0
        rows.append(dict(x=x, geo=np.array(geo, np.float32), y=None if share is None else float(share >= 0.5), share=share, length=length))
    return rows


class PNet(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(385 + NG, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x, g):
        return self.net(torch.cat([x, g], -1))[..., 0]


def stack(rows):
    return torch.from_numpy(np.stack([r["x"] for r in rows])).float(), torch.from_numpy(np.stack([r["geo"] for r in rows]))


def prune(g, keep):
    """graph without the pieces where keep is False (indices remapped, edges of removed pieces dropped)"""
    idx = {}
    for i, k in enumerate(keep):
        if k:
            idx[i] = len(idx)
    out = dict(g)
    out["pts"] = [g["pts"][i] for i in range(len(keep)) if keep[i]]
    out["edges"] = [(idx[i], idx[j]) for i, j in g["edges"] if i in idx and j in idx]
    if "nodes" in g and len(g["nodes"]) == len(keep):
        out["nodes"] = [g["nodes"][i] for i in range(len(keep)) if keep[i]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="e113_a")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--thrs", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7")
    ap.add_argument("--jthr", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    od = f"{OUT}/{a.out}"
    t0 = time.time()
    rows_tr = []
    for f in ("tok_e94_a_oof0", "tok_e94_a_oof1", "tok_e94_a_oof2"):
        z = pickle.load(open(f"{HOME}/exp_gcar_b123/p10/{f}.pkl", "rb"))
        gts = GD.load_gt_graphs(z["part"], [int(it["row"]) for it in z["items"]])
        for it, g in zip(z["items"], gts):
            rows_tr += [r for r in piece_feats(it, JH.derived_edges(it), JH_pts(g)) if r is not None]
        print(f"built {f}: {len(rows_tr)} pieces so far ({time.time() - t0:.0f} s)", flush=True)
    y = torch.tensor([r["y"] for r in rows_tr])
    print("train pieces", len(rows_tr), "true share", float(y.mean()), flush=True)
    X, G = stack(rows_tr)
    dev = "cuda"
    torch.manual_seed(a.seed)
    sfx = "" if a.seed == 0 else f"_s{a.seed}"
    net = PNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    n, bs = len(y), 1024
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.epochs * (n // bs))
    for ep in range(a.epochs):
        perm = torch.randperm(n)
        tl = []
        for i in range(0, n - bs + 1, bs):
            ix = perm[i : i + bs]
            loss = nn.functional.binary_cross_entropy_with_logits(net(X[ix].to(dev), G[ix].to(dev)), y[ix].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            tl.append(loss.item())
        if ep % 10 == 9 or ep == a.epochs - 1:
            print(f"epoch {ep + 1} loss {np.mean(tl):.4f}", flush=True)
    torch.save(net.state_dict(), f"{od}/pmodel{sfx}.pt")
    net.eval()
    jnet = JH.JNet().to(dev)
    jnet.load_state_dict(torch.load(f"{od}/jmodel.pt", map_location=dev))
    jnet.eval()
    scene_all = np.load(f"{L.D}/poses_val.npz")["scene"]
    rng = np.random.default_rng(0)
    res = {}
    for st, tokf in (("val_clean", "tok_e94_a_val_clean"), ("test_clean", "tok_e94_a_test_clean")):
        z = pickle.load(open(f"{HOME}/exp_gcar_b123/p10/{tokf}.pkl", "rb"))
        items = z["items"]
        preds = pickle.load(open(f"{OUT}/e94_a/graphs_{st}.pkl", "rb"))["preds"]
        src = L.load_src(st)
        ids = [int(i) for i in src["ids"]]
        gts_l = GD.load_gt_graphs("val", ids)
        frames = [piece_feats(it, [tuple(e) for e in p["edges"]], JH_pts(g)) for it, p, g in zip(items, preds, gts_l)]
        flat = [(k, i, r) for k, fr in enumerate(frames) for i, r in enumerate(fr) if r is not None]
        Xv, Gv = stack([r for _, _, r in flat])
        with torch.no_grad():
            pv = torch.sigmoid(net(Xv.to(dev), Gv.to(dev))).cpu().numpy()
        yv = np.array([r["y"] for _, _, r in flat])
        prob = [np.zeros(len(fr)) for fr in frames]
        for (k, i, _), p_ in zip(flat, pv):
            prob[k][i] = p_
        lens = np.array([r["length"] for _, _, r in flat])
        res[st] = dict(n_pieces=len(flat), true_share=float(yv.mean()), auroc=float(C12.auroc(pv, yv > 0.5)), false_length_share=float(lens[yv < 0.5].sum() / lens.sum()))
        print(st, res[st], flush=True)
        base = E.per_sample(preds, gts_l, a.workers)
        sc = scene_all[np.array(ids)]
        mask = np.ones(len(ids), bool)
        # junction verifier first (edges only), then the piece verifier
        jframes = [JH.junction_feats(it, [tuple(e) for e in p["edges"]], JH.gt_locs_of(g)) for it, p, g in zip(items, preds, gts_l)]
        allj = [r for fr in jframes for r in fr]
        Xj, Gj = JH.stack(allj)
        with torch.no_grad():
            pj = torch.sigmoid(jnet(Xj.to(dev), Gj.to(dev))).cpu().numpy()
        jprobs, k0 = [], 0
        for fr in jframes:
            jprobs.append(pj[k0 : k0 + len(fr)])
            k0 += len(fr)
        dis = JH.dissolve(preds, jframes, jprobs, a.jthr)
        mb = BE.metrics(base.sum(0))
        res[st]["base"] = {k_: float(v) for k_, v in mb.items()}
        for tag, src_graphs in (("pieces", preds), ("junctions+pieces", dis)):
            grid = {}
            for thr in [float(t) for t in a.thrs.split(",")]:
                new = [prune(g, pr >= thr) for g, pr in zip(src_graphs, prob)]
                comp = E.per_sample(new, gts_l, a.workers)
                m = BE.metrics(comp.sum(0))
                d = {k_: E.boot(comp, base, sc, mask, 1000, rng, key=k_) for k_ in ("F1@1m", "P@1m", "R@1m", "split6_F1", "merge6_F1", "route_f", "dir_acc")}
                npc = int(sum(len(g["pts"]) for g in new))
                grid[str(thr)] = dict(metrics={k_: float(v) for k_, v in m.items()}, diff={k_: dict(diff=v[0], lo=v[1], hi=v[2]) for k_, v in d.items()}, n_pieces=npc)
                print(st, tag, thr, {k_: round(float(v), 3) for k_, v in m.items() if k_ in ("F1@1m", "P@1m", "R@1m", "split6_F1", "merge6_F1", "route_f")}, "pieces", npc, flush=True)
            res[st][tag] = grid
        res[st]["base"]["n_pieces"] = int(sum(len(g["pts"]) for g in preds))
        print(st, "base", {k_: round(float(v), 3) for k_, v in mb.items() if k_ in ("F1@1m", "P@1m", "R@1m", "split6_F1", "merge6_F1", "route_f")}, flush=True)
    json.dump(res, open(f"{od}/phead_result{sfx}.json", "w"), indent=1)
    print("DONE", time.time() - t0)


def JH_pts(g):
    xy, _, _, _ = GE.point_graph(g, 1.0)
    return xy


if __name__ == "__main__":
    main()
