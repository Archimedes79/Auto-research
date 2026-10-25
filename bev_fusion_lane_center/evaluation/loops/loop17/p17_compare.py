"""Loop 17: table of failure-series results (clean phase / failure phase) of builder runs and of the E94 control graphs of loops 13 / 14, with paired scene bootstraps of F1@1 m between entries.
Entries: label=builder:<out>/<pert>_<variant>   (p17_eval.py result)   or   label=control:<variant>  (runs/e116_valfail/graphs_<pert>_<variant>.pkl; variant as in the file name, e.g. maxdecay_0.95)
Usage (gcar): p17_compare.py --pert dark --entries champ=builder:e130_champion/dark_single ctrl=control:single --boot champ,ctrl [--md tables/e130.md]"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boot_eval as BE  # noqa: E402
import p11_eval as E  # noqa: E402
import p16_common as C  # noqa: E402
import p8_lib as L  # noqa: E402

E17 = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e17"
OUT = L.OUT
_GT = None


def gts_of(rows):
    global _GT
    if _GT is None:
        _GT = L.cache("gt", "val")
    return [_GT[int(r)] for r in rows]


def load(entry, pert):
    kind, ref = entry.split(":", 1)
    if kind == "builder":
        r = pickle.load(open(f"{E17}/{ref}.pkl", "rb"))
        return dict(rows=np.asarray(r["rows"]), scene=np.asarray(r["scene"]), fail=np.asarray(r["phase"]) == 1, graphs=r["graphs"], comp=r["comp"])
    r = pickle.load(open(f"{OUT}/e116_valfail/graphs_{pert}_{ref}.pkl", "rb"))
    rows = np.asarray(r["rows"])
    return dict(rows=rows, scene=np.asarray(r["scene"]), fail=np.asarray([p == "fail" for p in r["phase"]]), graphs=r["graphs"], comp=E.per_sample(r["graphs"], gts_of(rows)))


def cards(d):
    g = gts_of(d["rows"])
    return {nm: C.card([d["graphs"][i] for i in np.where(m)[0]], [g[i] for i in np.where(m)[0]]) for nm, m in (("cl", ~d["fail"]), ("fail", d["fail"]))}


def boot(a, b, draws=2000, seed=0):
    """paired scene bootstrap of the failure-phase F1@1 m difference a - b (rows matched by frame id)"""
    ia = {int(r): i for i, r in enumerate(a["rows"])}
    ib = {int(r): i for i, r in enumerate(b["rows"])}
    common = [r for r in a["rows"] if int(r) in ib and a["fail"][ia[int(r)]]]
    sc = np.asarray([a["scene"][ia[int(r)]] for r in common])
    ca = np.stack([a["comp"][ia[int(r)]] for r in common])
    cb = np.stack([b["comp"][ib[int(r)]] for r in common])
    u = np.unique(sc)
    rng = np.random.RandomState(seed)
    f = lambda c: BE.metrics(c.sum(0))["F1@1m"]
    base = f(ca) - f(cb)
    ds = []
    for _ in range(draws):
        pick = rng.choice(u, len(u))
        idx = np.concatenate([np.where(sc == s)[0] for s in pick])
        ds.append(f(ca[idx]) - f(cb[idx]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return base, lo, hi, len(common), len(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pert", required=True)
    ap.add_argument("--entries", nargs="+", required=True)
    ap.add_argument("--boot", default="", help="a,b: bootstrap of F1@1 m (a - b) in the failure phase; several pairs separated by ;")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    ent = {}
    for s in a.entries:
        k, v = s.split("=", 1)
        ent[k] = load(v, a.pert)
    lines = [f"| {a.pert} | phase | n | S | F1@1 | F1@0.5 | J | vertices / 100 m | false length | lanes found |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, d in ent.items():
        cs = cards(d)
        for nm in ("cl", "fail"):
            c = cs[nm]
            n = int((d["fail"] if nm == "fail" else ~d["fail"]).sum())
            lines.append(f"| {k} | {nm} | {n} | {c['S']:.3f} | {c['F1@1.0']:.3f} | {c['F1@0.5']:.3f} | {c['J']:.3f} | {c['vert100']:.1f} | {c['false_len_frac']:.3f} | {c['lanes_found']:.2f} |")
    for pair in [p for p in a.boot.split(";") if p]:
        x, y = pair.split(",")
        b0, lo, hi, n, ns = boot(ent[x], ent[y])
        lines.append(f"| boot {x} - {y} | fail | {n} frames, {ns} scenes | F1@1 {b0:+.3f} [{lo:+.3f}, {hi:+.3f}] | | | | | | |")
    txt = "\n".join(lines)
    print(txt)
    if a.md:
        os.makedirs(os.path.dirname(f"{E17}/{a.md}"), exist_ok=True)
        open(f"{E17}/{a.md}", "a", encoding="utf-8").write(txt + "\n\n")


if __name__ == "__main__":
    main()
