"""Loop 12 / E109: calibration of the probabilities that are handed to the planner (system A: keep and route probability per edge, seed ensemble of the assignment network).
Labels as in training: keep = overlap with the ground-truth lane >= 0.6; route = overlap with the ground-truth route lane >= 0.5 (right map and a route in the frame), 0 for every edge of a wrong or missing map (abstention target).
Weights = edge length (m). Metrics per (source, SD-map cell): expected calibration error (10 equal-width bins), Brier score, log loss, mean probability vs frequency, expected vs true length per frame.
Temperature (one scalar per output) fitted on the VAL rows with the clean source (cells s1, s2) and applied to the TEST rows (all sources).
Usage (gcar, CPU): p12_calib.py --out e109_calib"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p8_lib as L  # noqa: E402

OUT = L.OUT
NB = 10


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def collect(samples, P):
    """-> dict group -> dict(keep=(p, y, w, frame), route=(p, y, w, frame))"""
    g = {}
    for i, s in enumerate(samples):
        m = len(s["tl"])
        w = s["length"].astype(np.float64)
        key = (s["src"], s["cell"])
        d = g.setdefault(key, dict(keep=[[], [], [], []], route=[[], [], [], []]))
        pk, pr = P[i, :m, 0].astype(np.float64), P[i, :m, 1].astype(np.float64)
        yk = (s["tl"] >= 0.6).astype(np.float64)
        for nm, p, y in (("keep", pk, yk),):
            for k_, v in enumerate((p, y, w, np.full(m, i))):
                d[nm][k_].append(v)
        if s["kind"] == "n" and s["has_route"]:
            yr = (s["rl"] >= 0.5).astype(np.float64)
        elif s["kind"] in ("w", "0"):
            yr = np.zeros(m)
        else:
            continue
        for k_, v in enumerate((pr, yr, w, np.full(m, i))):
            d["route"][k_].append(v)
    return {k: {nm: tuple(np.concatenate(a) for a in v[nm]) if len(v[nm][0]) else None for nm in v} for k, v in g.items()}


def metrics(p, y, w, fr):
    W = w.sum()
    b = np.minimum((p * NB).astype(int), NB - 1)
    ece, rel = 0.0, []
    for k in range(NB):
        mk = b == k
        wk = w[mk].sum()
        if wk > 0:
            conf, acc = (p[mk] * w[mk]).sum() / wk, (y[mk] * w[mk]).sum() / wk
            ece += wk / W * abs(conf - acc)
            rel.append([float(conf), float(acc), float(wk / W)])
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    nll = float(-(w * (y * np.log(pc) + (1 - y) * np.log(1 - pc))).sum() / W)
    brier = float((w * (p - y) ** 2).sum() / W)
    fs, inv = np.unique(fr, return_inverse=True)
    exp_len, true_len = np.bincount(inv, p * w), np.bincount(inv, y * w)
    return dict(ece=float(ece), brier=brier, nll=nll, mean_p=float((p * w).sum() / W), freq=float((y * w).sum() / W), exp_len=float(exp_len.mean()), true_len=float(true_len.mean()), n_edges=int(len(p)), length=float(W), rel=rel)


def auroc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ss = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def frame_conf(samples, P, tk=0.3, tr=0.4):
    """frame level: is the route confidence (length-weighted mean route probability of the selected route edges) a usable trust signal? -> dict group -> (confidence, success)"""
    import p8_assign as A

    srcs = A.srcs_for(samples)
    g = {}
    for i, s in enumerate(samples):
        if not (s["kind"] == "n" and s["has_route"]):
            continue
        m = len(s["tl"])
        keep = P[i, :m, 0] >= tk
        rt = (P[i, :m, 1] >= tr) & keep
        part, row = srcs[s["src"]]["part"], s["row"]
        r = L.route_success(A.seg_points(s, rt), L.cache("rpts", part)[int(row)])
        if r is None:
            continue
        w = s["length"][rt].astype(np.float64)
        c = float((P[i, :m, 1][rt] * w).sum() / w.sum()) if w.sum() > 0 else 0.0
        pk_ = P[i, :m, 0]
        g.setdefault((s["src"], s["cell"]), []).append((c, float(r["succ"]), float(w.sum()), float(pk_[keep].mean()) if keep.any() else 0.0,
                                                        float(s["length"][keep].sum()), float(pk_.mean()) if m else 0.0, float((P[i, :m, 1] * s["length"]).sum()), float(m)))
    return {k: np.array(v) for k, v in g.items()}


def fit_temp(p, y, w):
    z = logit(p)
    best = None
    for T in np.exp(np.linspace(np.log(0.3), np.log(5.0), 120)):
        q = np.clip(sig(z / T), 1e-6, 1 - 1e-6)
        nll = -(w * (y * np.log(q) + (1 - y) * np.log(1 - q))).sum() / w.sum()
        if best is None or nll < best[0]:
            best = (nll, T)
    return float(best[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="e109_calib")
    ap.add_argument("--runs", default="e87_w3,e87_w3s1,e87_w3s2")
    ap.add_argument("--tag", default="_seg")
    a = ap.parse_args()
    od = f"{OUT}/{a.out}"
    os.makedirs(od, exist_ok=True)
    dev = pickle.load(open(f"{L.D}/p8/ds_dev_tmpl{a.tag}.pkl", "rb"))
    Pd = np.mean([np.load(f"{OUT}/{r}/dev_pred.npy").astype(np.float32) for r in a.runs.split(",")], 0)
    gd = collect(dev, Pd)
    del dev
    te = pickle.load(open(f"{L.D}/p8/ds_test_tmpl{a.tag}.pkl", "rb"))
    Pt = np.load(f"{OUT}/e87_ens/test_pred.npy").astype(np.float32)
    gt = collect(te, Pt)
    fc = frame_conf(te, Pt)
    del te
    res = dict(temp={}, val={}, test={}, frame={})
    for (src, cell), v in sorted(fc.items()):
        if cell not in ("exact", "s2"):
            continue
        c, succ = v[:, 0], v[:, 1]
        o = np.argsort(c)
        thirds = [float(succ[idx].mean()) for idx in np.array_split(o, 3)]
        res["frame"][f"{src}/{cell}"] = dict(n=int(len(c)), success=float(succ.mean()), auroc_conf=auroc(c, succ >= 0.5), auroc_len=auroc(v[:, 2], succ >= 0.5), auroc_keep=auroc(v[:, 3], succ >= 0.5), success_by_conf_tercile=thirds, conf_mean=float(c.mean()))
        print(f"frame {src:20s} {cell:6s} n {len(c):4d} success {succ.mean():.3f}  AUROC route confidence {res['frame'][f'{src}/{cell}']['auroc_conf']:.3f} (route length {res['frame'][f'{src}/{cell}']['auroc_len']:.3f}, mean keep {res['frame'][f'{src}/{cell}']['auroc_keep']:.3f})  success by confidence tercile {thirds[0]:.2f} {thirds[1]:.2f} {thirds[2]:.2f}", flush=True)
    for nm in ("keep", "route"):  # temperature on the VAL clean rows (cells s1, s2)
        parts = [gd[("val_clean", c)][nm] for c in ("s1", "s2") if ("val_clean", c) in gd and gd[("val_clean", c)][nm] is not None]
        p, y, w = (np.concatenate([x[k] for x in parts]) for k in range(3))
        res["temp"][nm] = fit_temp(p, y, w)
        print(f"temperature {nm}: {res['temp'][nm]:.2f}", flush=True)
    # failure-conditioned temperature: fitted on the VAL rows of the failure source (cell s1) -> applied to the TEST rows of the same source (the failure type is known) or of the held-out source (T pooled over the other VAL failure sources)
    res["cond"] = {}
    fails = sorted({s_.split("_", 1)[1] for (s_, c_) in gd if s_ != "val_clean"})
    for nm in ("keep", "route"):
        for f_ in fails:
            gk = ("val_" + f_, "s1")
            if gk not in gd or gd[gk][nm] is None:
                continue
            Tf = fit_temp(*gd[gk][nm][:3])
            oth = [gd[("val_" + o, "s1")][nm][:3] for o in fails if o != f_ and ("val_" + o, "s1") in gd and gd[("val_" + o, "s1")][nm] is not None]
            Tl = fit_temp(*(np.concatenate([x[k] for x in oth]) for k in range(3)))
            tk_ = ("test_" + f_, "s2")
            if tk_ not in gt or gt[tk_][nm] is None:
                continue
            p, y, w, fr = gt[tk_][nm]
            row = dict(T_clean=res["temp"][nm], T_source=Tf, T_other_sources=Tl)
            for lab, T_ in (("raw", None), ("clean_T", res["temp"][nm]), ("source_T", Tf), ("other_sources_T", Tl)):
                q = p if T_ is None else sig(logit(p) / T_)
                row[lab] = metrics(q, y, w, fr)["ece"]
            res["cond"][f"{f_}/{nm}"] = row
            print(f"cond {f_:15s} {nm:5s} ECE raw {row['raw']:.3f} clean-T {row['clean_T']:.3f} ({row['T_clean']:.2f}) source-T {row['source_T']:.3f} ({Tf:.2f}) other-sources-T {row['other_sources_T']:.3f} ({Tl:.2f})", flush=True)
    for split, G in (("val", gd), ("test", gt)):
        for (src, cell), d in sorted(G.items()):
            for nm in ("keep", "route"):
                if d[nm] is None:
                    continue
                p, y, w, fr = d[nm]
                r0 = metrics(p, y, w, fr)
                T = res["temp"][nm]
                r1 = metrics(sig(logit(p) / T), y, w, fr)
                res[split][f"{src}/{cell}/{nm}"] = dict(raw=r0, scaled={k: v for k, v in r1.items() if k != "rel"})
                print(f"{split:4s} {src:20s} {cell:6s} {nm:5s} ECE {r0['ece']:.3f} -> {r1['ece']:.3f} Brier {r0['brier']:.3f} -> {r1['brier']:.3f} mean p {r0['mean_p']:.3f} freq {r0['freq']:.3f} | length per frame expected {r0['exp_len']:.1f} true {r0['true_len']:.1f}", flush=True)
    json.dump(res, open(f"{od}/calib_a.json", "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
