"""Loop 13 / E113c: a learned JUNCTION VERIFIER for the traced graph of the learned chain (system B).
Diagnosis (p13_junction.py): the learned graph finds as many ground-truth junctions as the rule graph with key points (56 % of the splits) but predicts about twice as many junctions as exist, and 44 % of its predicted splits lie on a real lane without a ground-truth junction.
Block: for every predicted junction (a piece with >= 2 successors = split, a piece with >= 2 predecessors = merge) an MLP reads the step tokens at the junction (end vertex of the parent, mean of the first vertices of the branches), the kind and 8 geometry numbers and says whether a ground-truth junction of that kind lies within 6 m.
A junction below the threshold is dissolved: only the branch with the smallest turning angle keeps its edge. Training: the tokens of the train rows traced by the trace network E94a (the dense-head inputs are out-of-fold, the trace network itself was trained on these rows, so the tokens are cleaner than those of VAL and TEST); threshold on VAL; TEST reported.
Usage (gcar): p13_jhead.py --out e113_a"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boot_eval as BE  # noqa: E402
import graph_from_dense as GD  # noqa: E402
import p11_eval as E  # noqa: E402
import p13_junction as PJ  # noqa: E402
import p8_lib as L  # noqa: E402

HOME = os.path.expanduser("~")
OUT = L.OUT
NG = 8


def derived_edges(it):
    pof = {}
    for k, p in enumerate(it["pieces"]):
        for v in p:
            pof.setdefault(int(v), k)
    return sorted({(pof[u], pof[v]) for u, v in it["edges"] if u in pof and v in pof and pof[u] != pof[v]})


def junction_feats(it, edges, gt_locs=None):
    """one row per predicted junction: token features, geometry, kind, label (if gt_locs), piece ids"""
    xy = it["xy"].astype(np.float64)
    P = [np.asarray(p, int) for p in it["pieces"]]
    tok, has = it["tok"].astype(np.float32), it["has"]
    succ, pred = {}, {}
    for i, j in edges:
        succ.setdefault(i, []).append(j)
        pred.setdefault(j, []).append(i)
    rows = []

    def tk(v):
        return tok[v] * has[v], float(has[v])

    for kind, grp, par in (("split", succ, None), ("merge", pred, None)):
        for c, others in grp.items():
            if len(others) < 2 or len(P[c]) < 2:
                continue
            if kind == "split":
                loc = xy[P[c][-1]]
                t_par, h_par = tk(P[c][-1])
                bs = [xy[P[o][0]] for o in others if len(P[o]) >= 1]
                t_br = [tk(P[o][0]) for o in others]
                dirs = []
                for o in others:
                    q = xy[P[o]]
                    dirs.append((q[min(3, len(q) - 1)] - q[0]) / max(np.linalg.norm(q[min(3, len(q) - 1)] - q[0]), 1e-6))
                ref = xy[P[c]][-1] - xy[P[c]][max(len(P[c]) - 4, 0)]
            else:
                loc = xy[P[c][0]]
                t_par, h_par = tk(P[c][0])
                bs = [xy[P[o][-1]] for o in others if len(P[o]) >= 1]
                t_br = [tk(P[o][-1]) for o in others]
                dirs = []
                for o in others:
                    q = xy[P[o]]
                    dirs.append((q[-1] - q[max(len(q) - 4, 0)]) / max(np.linalg.norm(q[-1] - q[max(len(q) - 4, 0)]), 1e-6))
                ref = xy[P[c]][min(3, len(P[c]) - 1)] - xy[P[c]][0]
            ref = ref / max(np.linalg.norm(ref), 1e-6)
            ang = [float(np.degrees(np.arccos(np.clip(d @ ref, -1, 1)))) for d in dirs]
            amax = max(float(np.degrees(np.arccos(np.clip(dirs[a_] @ dirs[b_], -1, 1)))) for a_ in range(len(dirs)) for b_ in range(a_ + 1, len(dirs)))
            lens = [float(np.linalg.norm(np.diff(xy[P[o]], axis=0), axis=1).sum()) for o in others if len(P[o]) >= 2]
            gap = [float(np.linalg.norm(b - loc)) for b in bs]
            geo = [len(others) / 4, amax / 90, min(ang) / 90, max(ang) / 90, float(np.mean(gap)) / 5, float(np.mean(lens) if lens else 0) / 30, float(np.linalg.norm(np.diff(xy[P[c]], axis=0), axis=1).sum()) / 30, 1.0 if kind == "split" else 0.0]
            tb = np.mean([t for t, _ in t_br], 0)
            x = np.concatenate([t_par, tb, [h_par, float(np.mean([h for _, h in t_br]))]]).astype(np.float16)
            y = None
            if gt_locs is not None:
                g = gt_locs[kind]
                y = float(len(g) > 0 and np.linalg.norm(g - loc, axis=1).min() <= 6.0)
            rows.append(dict(x=x, geo=np.array(geo, np.float32), kind=kind, c=c, others=others, ang=ang, y=y))
    return rows


def gt_locs_of(g):
    out = {"split": [], "merge": []}
    for kind, loc, _ in PJ.kinds(g):
        out[kind].append(loc)
    return {k: np.array(v).reshape(-1, 2) for k, v in out.items()}


class JNet(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(258 + NG, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x, g):
        return self.net(torch.cat([x, g], -1))[..., 0]


def stack(rows):
    return torch.from_numpy(np.stack([r["x"] for r in rows])).float(), torch.from_numpy(np.stack([r["geo"] for r in rows]))


def dissolve(preds, per_frame_rows, probs, thr):
    """graphs with every junction of probability < thr dissolved (only the branch with the smallest turning angle keeps its edge)"""
    new = [dict(p, edges=list(p["edges"])) for p in preds]
    for k, (rows, pr) in enumerate(zip(per_frame_rows, probs)):
        drop = set()
        for r, p in zip(rows, pr):
            if p >= thr:
                continue
            best = int(np.argmin(r["ang"]))
            for n_, o in enumerate(r["others"]):
                if n_ == best:
                    continue
                drop.add((r["c"], o) if r["kind"] == "split" else (o, r["c"]))
        if drop:
            new[k]["edges"] = [e for e in new[k]["edges"] if tuple(e) not in drop]
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="e113_a")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--thrs", default="0.2,0.3,0.4,0.5,0.6")
    a = ap.parse_args()
    od = f"{OUT}/{a.out}"
    t0 = time.time()
    rows_tr = []
    for f in ("tok_e94_a_oof0", "tok_e94_a_oof1", "tok_e94_a_oof2"):
        z = pickle.load(open(f"{HOME}/exp_gcar_b123/p10/{f}.pkl", "rb"))
        gts = GD.load_gt_graphs(z["part"], [int(it["row"]) for it in z["items"]])
        for it, g in zip(z["items"], gts):
            rows_tr += junction_feats(it, derived_edges(it), gt_locs_of(g))
        print(f"built {f}: {len(rows_tr)} junctions so far ({time.time() - t0:.0f} s)", flush=True)
    y = torch.tensor([r["y"] for r in rows_tr])
    print("train junctions", len(rows_tr), "true share", float(y.mean()), "split share", float(np.mean([r["kind"] == "split" for r in rows_tr])), flush=True)
    X, G = stack(rows_tr)
    dev = "cuda"
    torch.manual_seed(a.seed)
    sfx = "" if a.seed == 0 else f"_s{a.seed}"
    net = JNet().to(dev)
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
    torch.save(net.state_dict(), f"{od}/jmodel{sfx}.pt")
    net.eval()
    scene_all = np.load(f"{L.D}/poses_val.npz")["scene"]
    rng = np.random.default_rng(0)
    res = {}
    import p12_calib as C12

    for st, tokf in (("val_clean", "tok_e94_a_val_clean"), ("test_clean", "tok_e94_a_test_clean")):
        z = pickle.load(open(f"{HOME}/exp_gcar_b123/p10/{tokf}.pkl", "rb"))
        items = z["items"]
        preds = pickle.load(open(f"{OUT}/e94_a/graphs_{st}.pkl", "rb"))["preds"]
        src = L.load_src(st)
        ids = [int(i) for i in src["ids"]]
        gts_l = GD.load_gt_graphs("val", ids)
        frames = [junction_feats(it, [tuple(e) for e in p["edges"]], gt_locs_of(g)) for it, p, g in zip(items, preds, gts_l)]
        allr = [r for fr in frames for r in fr]
        Xv, Gv = stack(allr)
        with torch.no_grad():
            pv = torch.sigmoid(net(Xv.to(dev), Gv.to(dev))).cpu().numpy()
        yv = np.array([r["y"] for r in allr])
        res[st] = dict(n_junctions=len(allr), true_share=float(yv.mean()), auroc=float(C12.auroc(pv, yv > 0.5)))
        print(st, res[st], flush=True)
        probs, k0 = [], 0
        for fr in frames:
            probs.append(pv[k0 : k0 + len(fr)])
            k0 += len(fr)
        base = E.per_sample(preds, gts_l, a.workers)
        sc = scene_all[np.array(ids)]
        mask = np.ones(len(ids), bool)
        grid = {}
        for thr in [float(t) for t in a.thrs.split(",")]:
            new = dissolve(preds, frames, probs, thr)
            comp = E.per_sample(new, gts_l, a.workers)
            m = BE.metrics(comp.sum(0))
            d = {k_: E.boot(comp, base, sc, mask, 1000, rng, key=k_) for k_ in ("F1@1m", "split6_F1", "merge6_F1", "route_f", "dir_acc")}
            nj = sum(len(PJ.kinds(p)) for p in new)
            grid[str(thr)] = dict(metrics={k_: float(v) for k_, v in m.items()}, diff={k_: dict(diff=v[0], lo=v[1], hi=v[2]) for k_, v in d.items()}, n_junctions=int(nj))
            print(st, thr, {k_: round(float(v), 3) for k_, v in m.items() if k_ in ("F1@1m", "P@1m", "R@1m", "split6_F1", "merge6_F1", "route_f")}, "junctions", nj, flush=True)
        mb = BE.metrics(base.sum(0))
        res[st]["base"] = {k_: float(v) for k_, v in mb.items()}
        res[st]["base"]["n_junctions"] = int(sum(len(PJ.kinds(p)) for p in preds))
        res[st]["grid"] = grid
        print(st, "base", {k_: round(float(v), 3) for k_, v in mb.items() if k_ in ("F1@1m", "split6_F1", "merge6_F1", "route_f")}, "junctions", res[st]["base"]["n_junctions"], flush=True)
    json.dump(res, open(f"{od}/jhead_result{sfx}.json", "w"), indent=1)
    print("DONE", time.time() - t0)


if __name__ == "__main__":
    main()
