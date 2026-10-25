"""Loop 16 diagnostics of a trained design (plan section 6 diagnosis table): drift per level, along / across vertex error, duplicate rate, recall by lane length, split / merge landmark P/R at 1 / 2 / 4 / 6 m,
vertices per 100 m and chord lengths, validity calibration, hypothesis survival per level; then one rule-based recommendation for the next refinement.
Usage (gcar, GPU): p16_diag.py --design D3 --name e123_D3_r0_00 [--tau 0.3] [--n 125]  -> out/loop2/e123_ladder/diag/<name>.json"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_train as PT  # noqa: E402


def landmark_pr(preds, gts, tol):
    out = {}
    for key, fn in (("split", lambda g: GE.junctions(g)[0]), ("merge", lambda g: GE.junctions(g)[1])):
        m = n_p = n_g = 0
        for p, g in zip(preds, gts):
            a, b = fn(p), fn(g)
            m += GE._match_points(a, b, tol)
            n_p += len(a)
            n_g += len(b)
        P, R = m / max(n_p, 1), m / max(n_g, 1)
        out[key] = dict(P=P, R=R, F1=2 * P * R / max(P + R, 1e-9), n_pred=n_p / len(preds), n_gt=n_g / len(preds))
    return out


def vertex_errors(preds, gts):
    """along / across error of the vertices within 2 m of a lane (relative to the lane tangent at the nearest ground-truth point)"""
    al, ac = [], []
    for p, g in zip(preds, gts):
        G, Dd, _ = GE.dense_points(g, 0.5)
        if len(G) == 0:
            continue
        V = np.concatenate([np.asarray(q) for q in p["pts"] if len(q)], 0) if p["pts"] else np.zeros((0, 2))
        if len(V) == 0:
            continue
        d, ix = cKDTree(G).query(V)
        m = d <= 2.0
        rel = V[m] - G[ix[m]]
        t = Dd[ix[m]]
        n = np.stack([-t[:, 1], t[:, 0]], 1)
        al.append(np.abs((rel * t).sum(1)))
        ac.append(np.abs((rel * n).sum(1)))
    al, ac = np.concatenate(al), np.concatenate(ac)
    return dict(along_mean=float(al.mean()), across_mean=float(ac.mean()), across_median=float(np.median(ac)), n=int(len(al)))


def recall_by_length(preds, gts):
    short, long_ = [], []
    for p, g in zip(preds, gts):
        P, _, _ = GE.dense_points(p, 0.5)
        tP = cKDTree(P) if len(P) else None
        for i, q in enumerate(g["pts"]):
            q = GE.resample(np.asarray(q), 0.5)
            if len(q) < 2:
                continue
            L = 0.5 * len(q)
            cov = float((tP.query(q)[0] <= 1.0).mean()) if tP is not None else 0.0
            (short if L < 15 else long_).append(cov)
    return dict(recall_short=float(np.mean(short)) if short else None, n_short=len(short), recall_long=float(np.mean(long_)) if long_ else None, n_long=len(long_))


def node_stats(raw, gts, tau):
    """validity calibration and hypothesis survival per level from the raw node arrays"""
    lv = {}
    conf, hit = [], []
    for r, g in zip(raw, gts):
        G, _, _ = GE.dense_points(g, 0.5)
        tree = cKDTree(G) if len(G) else None
        d = tree.query(r["pos"])[0] if tree is not None else np.full(len(r["pos"]), 99.0)
        for l in np.unique(r["level"]):
            m = (r["level"] == l) & (r["valid"] > 0)
            e = lv.setdefault(int(l), dict(n=0, alive=0, near=0))
            e["n"] += int(m.sum())
            a = m & (r["pi"] >= tau)
            e["alive"] += int(a.sum())
            e["near"] += int((a & (d < 1.0)).sum())
        m = r["valid"] > 0
        conf += list(r["pi"][m])
        hit += list((d[m] < 1.0).astype(float))
    conf, hit = np.array(conf), np.array(hit)
    bins = np.linspace(0, 1, 6)
    ece, cal = 0.0, []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (conf >= a) & (conf < b + (1e-9 if b == 1 else 0))
        if m.sum():
            ece += m.mean() * abs(conf[m].mean() - hit[m].mean())
            cal.append((round(a, 1), round(float(conf[m].mean()), 3), round(float(hit[m].mean()), 3), int(m.sum())))
    return dict(levels={l: dict(valid=v["n"], alive=v["alive"], alive_near_lane=v["near"], alive_precision=v["near"] / max(v["alive"], 1)) for l, v in lv.items()}, ece=float(ece), calibration=cal)


def recommend(m, d):
    """rule-based next refinement (plan section 6 diagnosis table, first rule that applies)"""
    rec = []
    if m["F1o2o@1.0"] < 0.75 * m["F1@1.0"]:
        rec.append(("D", "duplicates: F1 one-to-one %.3f vs F1 %.3f" % (m["F1o2o@1.0"], m["F1@1.0"]), "density / repulsion weight up, or learned pair head (dedup flip)"))
    lm = d["landmarks"]["6.0"]
    if lm["split"]["P"] < 0.2 and lm["split"]["n_pred"] > 3 * max(lm["split"]["n_gt"], 0.5):
        rec.append(("D", "spurious forks: %.1f predicted vs %.1f true splits per frame" % (lm["split"]["n_pred"], lm["split"]["n_gt"]), "NMS radius up, n children down, coverage merge, best-per-GT duplicate label"))
    if lm["split"]["R"] < 0.4 and lm["split"]["n_pred"] <= 1.5 * max(lm["split"]["n_gt"], 0.5):
        rec.append(("D", "forks missed: split recall %.2f" % lm["split"]["R"], "b = 3 instead of 2, lower prune threshold"))
    if m["vert100"] > 17:
        rec.append(("H", "vertices per 100 m %.1f" % m["vert100"], "step-length target, vertex-count penalty (SPF1)"))
    ve = d["vertex_error"]
    if ve["along_mean"] > 2.0 * ve["across_mean"]:
        rec.append(("D", "along-lane error %.2f >> across %.2f" % (ve["along_mean"], ve["across_mean"]), "step-length target (Douglas-Peucker rule), vertex-count penalty"))
    if d["recall_by_length"]["recall_short"] is not None and d["recall_by_length"]["recall_short"] < 0.6 * (d["recall_by_length"]["recall_long"] or 1):
        rec.append(("D", "short lanes missed: recall %.2f vs long %.2f" % (d["recall_by_length"]["recall_short"], d["recall_by_length"]["recall_long"]), "more seeds or lower validity threshold, focal gamma up"))
    if d["nodes"]["ece"] > 0.15:
        rec.append(("H", "validity calibration ECE %.2f" % d["nodes"]["ece"], "focal loss, temperature"))
    if m["false_len_frac"] > 0.1:
        rec.append(("D", "false length %.3f" % m["false_len_frac"], "stage C rescore, density weight up"))
    if not rec:
        rec.append(("H", "no diagnostic below its threshold", "sweep the most sensitive weight one step (position loss)"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--n", type=int, default=125)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfgx = PT.parse_set(a.set)
    c = json.load(open(f"{C.LADDER}/cards/{a.name}.json", encoding="utf-8-sig"))
    tau = a.tau if a.tau is not None else c["metrics"]["tau"]
    cfgx.update({k: v for k, v in c["cfg"].items() if k not in cfgx})  # the run's own configuration (flips and switches included)
    net, cfg = PT.build(a.design, cfgx)
    net = net.cuda()
    net.load_state_dict(torch.load(f"{C.OUT}/e123/{a.name}/model.pt", map_location="cuda"), strict=False)
    ks = C.val_subset()[:: max(1, 250 // a.n)]
    ds = PD.BuilderData("val_clean", ks=ks, train=False, with_prev=cfg.temporal)
    raw = []
    graphs, gts, _ = PI.predict(net, ds, ks, taus=(tau,), raw=raw)
    g = graphs[tau]
    m = C.card(g, gts)
    d = dict(tau=tau, n=len(g), metrics={k: m[k] for k in ("F1@1.0", "F1@0.5", "F1o2o@1.0", "J", "Sp", "S", "vert100", "false_len_frac", "lanes_found", "dir_acc")})
    d["landmarks"] = {str(t): landmark_pr(g, gts, t) for t in (1.0, 2.0, 4.0, 6.0)}
    d["vertex_error"] = vertex_errors(g, gts)
    d["recall_by_length"] = recall_by_length(g, gts)
    d["nodes"] = node_stats(raw, gts, tau)
    chord = [np.linalg.norm(np.diff(np.asarray(p), axis=0), axis=1) for gg in g for p in gg["pts"] if len(p) > 1]
    chord = np.concatenate(chord) if chord else np.zeros(1)
    d["chord_length"] = dict(mean=float(chord.mean()), median=float(np.median(chord)), p90=float(np.percentile(chord, 90)))
    d["recommendation"] = recommend(m, d)
    os.makedirs(f"{C.LADDER}/diag", exist_ok=True)
    json.dump(d, open(f"{C.LADDER}/diag/{a.name}.json", "w"), indent=1, default=float)
    lm = d["landmarks"]["6.0"]
    print(a.name, "tau", tau, "| F1o2o %.3f | splits pred %.1f gt %.1f P %.3f R %.3f | merges pred %.1f gt %.1f | v100 %.1f chord %.1f m | along %.2f across %.2f | ece %.2f" % (
        m["F1o2o@1.0"], lm["split"]["n_pred"], lm["split"]["n_gt"], lm["split"]["P"], lm["split"]["R"], lm["merge"]["n_pred"], lm["merge"]["n_gt"], m["vert100"], d["chord_length"]["mean"], d["vertex_error"]["along_mean"], d["vertex_error"]["across_mean"], d["nodes"]["ece"]))
    for r in d["recommendation"]:
        print("  ->", r)


if __name__ == "__main__":
    main()
