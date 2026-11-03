"""Scoring of the grid chain (E272) on VAL: vertex metrics of the FUSED map (hold on) and of the same weights with the hold off, and the assignment of the read-out vertices to the SD map.

    wsl bash tools/env/_py14.sh chain_eval.py --run e272_j_s0 [--budget 11.2 8 14.3] [--n 0]

Frames = the VAL split through `BuilderData.rows` (never the stored TEST rows), chains of T = 3 frames ending at every VAL frame (the predecessors are VAL frames of the same scene; a
missing one repeats). The read-out threshold of each variant is the exact k-th largest 3 x 3-maximum weight over 300 TRAIN chains at the budget (as in vertex_eval: in-sample, named).
Cards: runs/e272/<run>/card.json (hold on) and runs/e272/<run>_nohold/card.json (hold off), with `counts` per frame (paired bootstrap against the E271 cards: same frames, same order),
and in the first card `assign`: for the vertices of the hold-on read-out, the TRAINED assigner and the START-value assigner side by side:
  road_acc  share of the vertices within 1 m of a GT lane (heading within 30 degrees) whose most probable road is a plausible road of that lane's nearest point (the SD map holds duplicate
            roads: a plausible set, not one id), dust_off  share of the vertices more than 2 m from every lane that the assigner puts in the dustbin.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training/grid", "training/data"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import assign as AS  # noqa: E402
import chain as CH  # noqa: E402
import chain_data as CD  # noqa: E402
import vertex_eval as VE  # noqa: E402
from lane_task import measures as M  # noqa: E402


def fit_thresholds(model, budgets, n=300, seed=0, device="cuda"):
    """-> {variant: {budget: (thr, vertices per 100 m reached)}} for "hold" (fused map) and "nohold" (the last sparse map), from `n` TRAIN chains"""
    ds0 = CD.ChainData("train")
    ks = sorted(np.random.RandomState(seed).choice(len(ds0), n, replace=False).tolist())
    _, dl = CD.loader("train", bs=4, workers=2, shuffle=False, ks=ks)
    model.eval().to(device)
    ws = {"hold": [], "nohold": []}
    glen = 0.0
    with torch.no_grad():
        for b in dl:
            out = model(b["x"].to(device), b["step"].to(device), b["has"].to(device))
            for i in range(out["fused"].shape[0]):
                ws["hold"].append(VE.peak_weights(out["fused"][i].cpu()))
                ws["nohold"].append(VE.peak_weights(out["sparse"][-1][i].cpu()))
            glen += float(b["m05"][:, -1].sum()) * 0.5
    res = {}
    for v, lst in ws.items():
        a = np.sort(np.concatenate(lst))[::-1]
        res[v] = {}
        for bud in budgets:
            k = min(max(int(round(bud * glen / 100.0)), 1), len(a))
            thr = float(a[k - 1])
            res[v][bud] = (thr, 100.0 * float((a >= thr).sum()) / glen)
    return res


def assign_counts(assign_fn, V, H, w, mp, G, T, gmask):
    """-> (hits, labelled, dust_off, off) of the vertices (metric frame, inside the box) against the GT points G (metric frame), their tangents T and the plausible-road mask gmask (numpy bool, (len G, R + 1))"""
    from scipy.spatial import cKDTree
    if len(V) == 0 or mp is None or len(G) == 0:
        return 0, 0, 0, 0
    d, j = cKDTree(G).query(V)
    lab = (d < 1.0) & ((H * T[j]).sum(1) > np.cos(np.deg2rad(30.0)))
    off = d > 2.0
    with torch.no_grad():
        out = assign_fn(torch.from_numpy(V.astype(np.float32)), torch.from_numpy(H.astype(np.float32)), torch.from_numpy(w.astype(np.float32)), mp)
    best = out["P_road"].cpu().numpy().argmax(1)
    R = int(mp["n_ids"])
    hit = gmask[j[lab], best[lab]]
    return int(hit.sum()), int(lab.sum()), int((best[off] == R).sum()), int(off.sum())


def eval_chain(model, name, seed, budgets=(11.2, 8.0, 14.3), device="cuda", n=0, extra=None):
    import p16_data as PD
    fit = fit_thresholds(model, list(budgets), device=device)
    nval = len(PD.BuilderData("val_clean", train=False).rows)
    ks = list(range(nval))[: n or None]
    _, dl = CD.loader("val_clean", bs=4, workers=2, shuffle=False, ks=ks)
    model.eval().to(device)
    start = AS.Assigner().to("cpu")
    cs = {v: {b: [] for b in budgets} for v in ("hold", "nohold")}
    acc = {"trained": np.zeros(4), "start": np.zeros(4)}
    rows = []
    with torch.no_grad():
        for b in dl:
            out = model(b["x"].to(device), b["step"].to(device), b["has"].to(device))
            maps = {"hold": out["fused"].cpu(), "nohold": out["sparse"][-1].cpu()}
            for i in range(maps["hold"].shape[0]):
                m = b["m05"][i, -1] > 0
                G, T = b["g05"][i, -1][m].numpy().astype(np.float64), b["t05"][i, -1][m].numpy().astype(np.float64)
                rows.append(int(b["row"][i]))
                for v in ("hold", "nohold"):
                    for bud in budgets:
                        V, H, w = VE.read_out(maps[v][i], fit[v][bud][0])
                        cs[v][bud].append(VE.frame_counts(V, H, (G, T)))
                        if v == "hold" and bud == budgets[0] and b["mp"][i] is not None:
                            Vi, Hi, wi = VE.inside(V, H, w)
                            gm = b["gmask"][i].numpy()
                            for key, fn in (("trained", lambda p, h, ww, mp_: _trained(model, device, p, h, ww, mp_)), ("start", lambda p, h, ww, mp_: _start(start, p, h, ww, mp_))):
                                acc[key] += np.array(assign_counts(fn, Vi, Hi, wi, b["mp"][i], G, T, gm))
    pz = np.load(f"{PD.D}/poses_val.npz")
    scenes = [int(pz["scene"][r]) for r in rows]
    cards = {}
    for v, nm in (("hold", name), ("nohold", name + "_nohold")):
        front = {str(bud): dict(thr=fit[v][bud][0], train_vgt100=fit[v][bud][1], reached=bool(abs(fit[v][bud][1] - bud) <= 0.5), **{k: float(x) for k, x in M.vertex_numbers(VE.total(cs[v][bud])).items()}) for bud in budgets}
        ex = dict(budget=budgets[0], thr=fit[v][budgets[0]][0], frontier=front, variant=v, **(extra or {}))
        if v == "hold":
            ex["assign"] = {k: dict(road_acc=a[0] / max(a[1], 1), labelled=int(a[1]), dust_off=a[2] / max(a[3], 1), off=int(a[3])) for k, a in acc.items()}
        cards[v], d = VE.write_card("j", cs[v][budgets[0]], scenes, seed=seed, extra=ex, name=nm, exp="e272")
    return cards, d


def _start(start, p, h, w, mp):
    pos = torch.stack([p[:, 1], -p[:, 0]], 1)
    hdg = torch.stack([h[:, 1], -h[:, 0]], 1)
    return start(pos, hdg, w, AS.to_torch(mp))


def _trained(model, device, p, h, w, mp):
    return model.assign_vertices(p.to(device), h.to(device), w.to(device), AS.to_torch(mp, device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--budget", type=float, nargs="*", default=[11.2, 8.0, 14.3])
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    run_dir = os.path.join(ROOT, "runs", a.run)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    model = CH.GridChain(width=cfg.get("width", 48))
    model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu")["state"])
    cards, d = eval_chain(model, a.run, int(cfg.get("seed", 0)), budgets=tuple(a.budget), device="cuda" if torch.cuda.is_available() else "cpu", n=a.n, extra=dict(rescored=True))
    for v, c in cards.items():
        print(a.run, v, {k: round(x, 4) for k, x in c["metrics"].items()}, c.get("assign", ""))


if __name__ == "__main__":
    main()
