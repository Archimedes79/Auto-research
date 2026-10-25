"""Loop 18 / R2.5 (K190 fork pruning): the junction head decides which forks of the assembled graph stay linked (p16_infer.prune_forks: a fork whose 3 x 3 junction heat is below the threshold keeps one branch linked, the
other branches start / end at a copy of the vertex - geometry unchanged). Evaluation on all validation frames with the shared metric code (sparse_lanes) and the paired scene bootstrap against the unpruned champion.
The raw node outputs of the 873 VAL frames are cached by p18_asm_check.py (~/exp_gcar_b123/p18/raw_val873.pkl; champion f1, M 288 / K0 144). Nothing is selected on TEST.
Usage (gcar): p18_fork_prune_eval.py [--thr 0.1,0.2,0.3,0.4,0.5] [--pairs 0.3:0.2,0.2:0.3] [--out fork_prune_val]   -> runs/e18/<out>/table.md, results.json"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap as SB  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_timing as TM  # noqa: E402
import sparse_metrics as SM  # noqa: E402

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
ap = argparse.ArgumentParser()
ap.add_argument("--thr", default="0.1,0.2,0.3,0.4,0.5")
ap.add_argument("--pairs", default="0.3:0.2,0.2:0.3,0.4:0.3")
ap.add_argument("--out", default="fork_prune_val")
ap.add_argument("--n", type=int, default=873)
a = ap.parse_args()
raw, gts = pickle.load(open(os.path.expanduser(f"~/exp_gcar_b123/p18/raw_val{a.n}.pkl"), "rb"))
sp = json.load(open(f"{R}/data/splits.json"))
scene = np.load(os.path.expanduser("~/exp_gcar_b123/poses_val.npz"))["scene"][np.array(sp["val"])][: len(gts)]
net, cfg = TM.load_builder(288, 144)
variants = [("champion (no pruning)", {})]
for t in a.thr.split(","):
    variants.append((f"prune {t}", dict(fork_prune=float(t))))
for p in [x for x in a.pairs.split(",") if x]:
    s, m = p.split(":")
    variants.append((f"prune split {s} / merge {m}", dict(fork_prune=float(s), fork_prune_m=float(m))))
recs, cards, graphs = {}, {}, {}
for name, kw in variants:
    asm = dict(TM.ASM)
    asm.update(kw)
    gs = [PI.assemble(r, None, 0.7, cfg, fast=2, **asm) for r in raw]
    graphs[name] = gs
    recs[name] = SB.frame_records(gs, gts)
    m = SB.pooled(recs[name])
    cards[name] = {k: float(m[k]) for k in SB.KEYS if k in m}
    cards[name].update({k: float(m[k]) for k in ("split6_P", "split6_R", "merge6_P", "merge6_R") if k in m})
    print(name, {k: round(v, 3) for k, v in cards[name].items() if k in ("S", "Q", "J", "vert100", "F1@1.0", "lanes_found")}, flush=True)
base = variants[0][0]
lines = ["| variant | S | Q | J | split P / R | merge P / R | vert100 | F1@1 | lanes found | S vs champion [95 %] | Q vs champion [95 %] | J vs champion [95 %] | verdict on S |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
res = {}
for name, _ in variants:
    c = cards[name]
    if name == base:
        d = ["", "", "", ""]
    else:
        pr = SB.paired(recs[name], recs[base], scene, B=1000)
        res[name] = {k: pr[k] for k in ("S", "Q", "J") if k in pr}
        d = [f"{pr[k]['diff']:+.3f} [{pr[k]['lo']:+.3f}, {pr[k]['hi']:+.3f}]" if isinstance(pr.get(k), dict) and "diff" in pr[k] else str(pr.get(k)) for k in ("S", "Q", "J")] + [str(pr["S"].get("verdict", ""))]
    lines.append(f"| {name} | {c['S']:.3f} | {c['Q']:.3f} | {c['J']:.3f} | {c['split6_P']:.2f} / {c['split6_R']:.2f} | {c['merge6_P']:.2f} / {c['merge6_R']:.2f} | {c['vert100']:.1f} | {c['F1@1.0']:.3f} | {c['lanes_found']:.3f} | {d[0]} | {d[1]} | {d[2]} | {d[3]} |")
od = f"{R}/runs/e18/{a.out}"
os.makedirs(od, exist_ok=True)
open(f"{od}/table.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
json.dump(dict(cards=cards, paired=res), open(f"{od}/results.json", "w"), indent=1, default=float)
print("\n".join(lines))
