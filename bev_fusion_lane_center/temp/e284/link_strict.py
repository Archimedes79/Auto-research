"""scratch (continuation of ends_quality.py: the delivered long-link ENDS are mostly 2 - 4 m off along the lane (flagged ends within 2 m of a GT entry / exit: 49 %, within 4 m: 73 %), while the pair rule of E276 P9 / E281 matches the ends of a delivered link to GT NODES within 2 m,
so a delivered long link with correct curve but ends 2 - 4 m off counts as 'wrong' or 'unmatched': long-link precision 0.17 - 0.19). A STRICT CURVE-LEVEL label that does not depend on where the ends lie along the lane:
a delivered long link (baseline B + the hard step, kept, long flag) is RIGHT when >= 80 % of its curve points (0.5 m samples) lie within 1.0 m of ONE GT connector run (a contiguous run of connector points of a piece of the lane GT, >= 3 points) and the link runs
in the direction of the run (the nearest run point of the link's last point has a larger index than that of its first point). A GT connector run is FOUND when a delivered long link is right for it (>= 80 % of the LINK's points within 1.0 m of the run AND >= 80 % of the RUN's
points within 1.0 m of the link). Reported on the full dump, VAL 873: strict precision of the delivered long links, against the pair-rule precision of the same links (from `temp/e281/info_s<seed>.pkl`); strict recall of the connector runs >= 8 m;
the strict precision by link probability bin; AUC of the link probability for the strict label (all candidate long links with both ends kept, p >= 0.05).
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 bash tools/env/_py14.sh temp/e284/link_strict.py"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as QR  # noqa: E402
import qtargets as QT  # noqa: E402

SPEC = QR.parse("close:1.5@0.5")
frames = QE.val_frames()


def runs_of(g):
    out = []
    for pc in g.get("pieces", []):
        q = np.asarray(pc["pts"], float).reshape(-1, 2)
        if len(q) < 4:
            continue
        conn = np.zeros(len(q), bool)
        for s in pc["segments"]:
            if s["kind"] == "connector":
                conn[s["i0"]: s["i1"] + 1] = True
        k = 0
        while k < len(q):
            if conn[k]:
                j = k
                while j + 1 < len(q) and conn[j + 1]:
                    j += 1
                if j - k + 1 >= 3:
                    out.append(q[k: j + 1])
                k = j + 1
            else:
                k += 1
    return out


res = {}
for SEED in (0, 1):
    dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{SEED}.pkl"), "rb"))
    INFO = pickle.load(open(os.path.join(ROOT, "temp", "e281", f"info_s{SEED}.pkl"), "rb"))
    n_del = n_strict = n_pair = n_pair_and_strict = 0
    n_runs = n_runs_found = 0
    all_p, all_y = [], []
    bins = {b: [0, 0] for b in ((0.05, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.01))}
    for n, (f, d) in enumerate(zip(frames, dump)):
        runs = runs_of(f.lane())
        runs_in = [r for r in runs if (np.abs(r) < 48).all(1).any()]
        trees = [cKDTree(r) for r in runs_in]
        pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                             links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), SPEC)
        keep = QE.kept(pred, 0.5, 0.5)[0]
        lg = np.asarray(d["link_long"], bool)
        pl = np.asarray(d["link_p"], float)
        sel = np.flatnonzero(lg & keep[d["link_src"]] & keep[d["link_dst"]] & (d["link_src"] != d["link_dst"]) & (pl >= 0.05))
        found = np.zeros(len(runs_in), bool)
        for l in sel:
            a, b = int(d["link_src"][l]), int(d["link_dst"][l])
            c = QT.hermite(d["pos"][a], d["dirv"][a], d["pos"][b], d["dirv"][b], float(d["link_a"][l]), float(d["link_b"][l]), 0.5)
            inb = (np.abs(c) < 48).all(1)
            right = False
            for ri, (r, tr) in enumerate(zip(runs_in, trees)):
                dd, idx = tr.query(c)
                if (dd[inb] <= 1.0).mean() >= 0.8 and inb.any() and idx[-1] > idx[0]:
                    right = True
                    rr = cKDTree(c).query(r)[0]
                    if (rr[(np.abs(r) < 48).all(1)] <= 1.0).mean() >= 0.8:
                        found[ri] = True
            all_p.append(pl[l])
            all_y.append(right)
            for (lo, hi), v in bins.items():
                if lo <= pl[l] < hi:
                    v[0] += int(right)
                    v[1] += 1
            if pl[l] >= 0.5:
                n_del += 1
                n_strict += int(right)
                pr = bool(INFO[n]["right"][l]) if len(INFO[n]["right"]) > l else False
                n_pair += int(pr)
                n_pair_and_strict += int(pr and right)
        for ri, r in enumerate(runs_in):
            if 0.5 * len(r) >= 8.0:
                n_runs += 1
                n_runs_found += int(found[ri])
        f._lane = f._ref = None
        if (n + 1) % 200 == 0:
            print(f"  seed {SEED} frame {n + 1}/{len(frames)}", flush=True)
    all_p, all_y = np.array(all_p), np.array(all_y)
    r = dict(delivered_long_links=n_del, strict_precision=n_strict / max(n_del, 1), pair_rule_precision_all_delivered=n_pair / max(n_del, 1), pair_rule_right_that_are_strict_right=n_pair_and_strict / max(n_pair, 1),
             runs_ge_8m=n_runs, runs_found_strict=n_runs_found / max(n_runs, 1), auc_p_for_strict=float(roc_auc_score(all_y, all_p)), candidates=int(len(all_y)), strict_share_of_candidates=float(all_y.mean()),
             by_p={f"{lo}-{hi}": dict(n=v[1], strict_precision=v[0] / max(v[1], 1)) for (lo, hi), v in bins.items()})
    res[f"seed {SEED}"] = r
    print(f"seed {SEED}: delivered long links {n_del}; STRICT curve-level precision {100 * r['strict_precision']:.1f} %  (pair-rule label on the same links, all of them as denominator: {100 * r['pair_rule_precision_all_delivered']:.1f} %; of the pair-rule right ones {100 * r['pair_rule_right_that_are_strict_right']:.0f} % are strict right)")
    print(f"   connector runs >= 8 m found strictly: {100 * r['runs_found_strict']:.1f} % of {n_runs};  AUC of p for the strict label {r['auc_p_for_strict']:.3f} on {r['candidates']} candidates ({100 * r['strict_share_of_candidates']:.1f} % strict right)")
    print("   strict precision by p:", {k: (v["n"], round(v["strict_precision"], 3)) for k, v in r["by_p"].items()})
json.dump(res, open(os.path.join(ROOT, "temp", "e284", "link_strict.json"), "w"), indent=1)
