"""Histograms and sensitivity tables of the footprint share f (D38 b) from temp/coverage/footprint_share.csv. CPU, stdlib + numpy.
Run from bev_fusion (WSL): python3 temp/coverage/analyse_footprint_share.py [--scenes eval]  -> prints markdown and writes temp/coverage/footprint_share_report.md
Objects counted: those with a part inside the 60 x 52 m grid (in_grid > 0), official validation scenes of the cache."""
import argparse
import csv
import pickle

import numpy as np

a = argparse.ArgumentParser()
a.add_argument("--scenes", default="val", help="val = every cached validation frame; eval = only the 82 scenes of the cold VAL 1000 evaluation")
a.add_argument("--var", default="f_fus", choices=["f_fus", "f_fusnoocc", "f_cam"])
args = a.parse_args()

rows = list(csv.DictReader(open("temp/coverage/footprint_share.csv")))
eval_scenes = None
if args.scenes == "eval":
    recs = pickle.load(open("runs/t10_dual_s0/eval/scene_records_n1000.pkl", "rb"))
    eval_scenes = set(next(iter(recs.values())).keys())
for r in rows:
    for k in ("val", "observed", "vel_known", "c_cam", "c_fus"):
        r[k] = int(r[k])
    for k in ("cx", "cy", "rng", "length", "width", "in_grid", "f_cam", "f_fusnoocc", "f_fus"):
        r[k] = float(r[k])
rows = [r for r in rows if r["val"] == 1 and r["in_grid"] > 0 and (eval_scenes is None or r["scene"] in eval_scenes)]
f = np.array([r[args.var] for r in rows])
grp = np.array([r["group"] for r in rows])
rng = np.array([r["rng"] for r in rows])
cfus = np.array([r["c_fus"] for r in rows])
ccam = np.array([r["c_cam"] for r in rows])
obs = np.array([r["observed"] for r in rows])
ingrid = np.array([r["in_grid"] for r in rows])
fno = np.array([r["f_fusnoocc"] for r in rows])
out = []


def P(s=""):
    out.append(s)
    print(s)


groups = sorted(set(grp))
P(f"# Footprint share f ({args.var}) of the GT objects in the fusion area, official validation scenes ({args.scenes}), objects with a part inside the grid")
P(f"\n{len(rows)} objects: " + ", ".join(f"{g} {int((grp == g).sum())}" for g in groups) + f"; frames in the csv: {len({(r['scene'], r['sample']) for r in rows})}")
P(f"objects fully inside the grid (in_grid = 1): {int((ingrid >= 0.999).sum())}, cut by the grid edge: {int((ingrid < 0.999).sum())}")
P("\n## Histogram of f (share of the footprint inside the fusion area; samples outside the grid count as not observed)\n")
edges = [0.0, 1e-9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 - 1e-9, 1.0 + 1e-9]
labels = ["0", "(0, .1)", "[.1, .2)", "[.2, .3)", "[.3, .4)", "[.4, .5)", "[.5, .6)", "[.6, .7)", "[.7, .8)", "[.8, .9)", "[.9, 1)", "1"]
P("| f | all | " + " | ".join(groups) + " |")
P("|---|---|" + "---|" * len(groups))
for k, lab in enumerate(labels):
    lo, hi = edges[k], edges[k + 1]
    sel = (f >= lo) & (f < hi) if k > 0 else (f < hi)
    P(f"| {lab} | {int(sel.sum())} | " + " | ".join(str(int((sel & (grp == g)).sum())) for g in groups) + " |")
binary = ((f <= 0.0) | (f >= 1.0)).sum()
P(f"\nexactly 0 or exactly 1: {int(binary)} of {len(f)} ({100 * binary / len(f):.1f} %); strictly between: {len(f) - int(binary)} ({100 * (len(f) - binary) / len(f):.1f} %)")
P("\n## One threshold t (the hard rule: care when f >= t, else don't care): objects kept\n")
ts = [0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99]
P("| t | all | " + " | ".join(groups) + " | change vs the neighbour t |")
P("|---|---|" + "---|" * len(groups) + "---|")
prev = None
for t in ts:
    kept = f >= t
    ch = "" if prev is None else f"{int(prev.sum() - kept.sum())} objects leave"
    P(f"| {t} | {int(kept.sum())} ({100 * kept.mean():.1f} %) | " + " | ".join(f"{int((kept & (grp == g)).sum())}" for g in groups) + f" | {ch} |")
    prev = kept
P("\n## Two thresholds: don't care below t_low, partial between, care from t_high (counts of all objects)\n")
P("| t_low | t_high | don't care | partial | care |")
P("|---|---|---|---|---|")
for lo in (0.1, 0.2, 0.3, 0.5):
    for hi in (0.5, 0.7, 0.8, 0.9, 0.95):
        if hi <= lo:
            continue
        P(f"| {lo} | {hi} | {int((f < lo).sum())} | {int(((f >= lo) & (f < hi)).sum())} | {int((f >= hi).sum())} |")
P("\n## The rules in use today against f\n")
P(f"- centre cell in the fusion area (c_fus = 1): {int(cfus.sum())} objects; of them with f < 0.5: {int(((cfus == 1) & (f < 0.5)).sum())}, with f < 0.1: {int(((cfus == 1) & (f < 0.1)).sum())}")
P(f"- centre cell NOT in the fusion area: {int((cfus == 0).sum())} objects; of them with f >= 0.5: {int(((cfus == 0) & (f >= 0.5)).sum())}, with f >= 0.9: {int(((cfus == 0) & (f >= 0.9)).sum())}")
P(f"- centre cell in the camera cells (c_cam = 1): {int(ccam.sum())}; f counts: {int(((ccam == 1) & (f < 0.5)).sum())} below 0.5")
P(f"- `observed` flag (a lidar or radar point in the box): {int(obs.sum())} observed, {int((obs == 0).sum())} unobserved; unobserved with f >= 0.5: {int(((obs == 0) & (f >= 0.5)).sum())}; observed with f < 0.5: {int(((obs == 1) & (f < 0.5)).sum())}")
P(f"- hidden behind a box (f_fusnoocc - f_fus): objects that lose at least 0.1 of f by the occlusion: {int(((fno - f) >= 0.1).sum())}; at least 0.5: {int(((fno - f) >= 0.5).sum())}")
P("\n## By range (care at t = 0.5, share of the objects of the bin)\n")
P("| range m | objects | f >= 0.5 | f = 0 | 0 < f < 0.5 | 0.5 <= f < 1 | f = 1 |")
P("|---|---|---|---|---|---|---|")
for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 200)):
    s = (rng >= lo) & (rng < hi)
    n = int(s.sum())
    if n:
        P(f"| {lo}-{hi} | {n} | {100 * (s & (f >= 0.5)).sum() / n:.0f} % | {100 * (s & (f <= 0)).sum() / n:.0f} % | {100 * (s & (f > 0) & (f < 0.5)).sum() / n:.0f} % | {100 * (s & (f >= 0.5) & (f < 1)).sum() / n:.0f} % | {100 * (s & (f >= 1)).sum() / n:.0f} % |")
open(f"temp/coverage/footprint_share_report_{args.scenes}_{args.var}.md", "w").write("\n".join(out) + "\n")
