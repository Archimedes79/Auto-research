"""Does the metric respond the way it should?  Controlled errors are applied to ground-truth graphs (they play the 'prediction'); every metric of the card is measured.
Checks (assertions) + a table: how much S changes per kind and size of error, and how large an error is worth the noise band of 0.010.
Writes evaluation/METRIC_RESPONSE.md.   Run: tools/env/_py14.sh test_metric_response.py [--n 150]"""
import argparse
import json
import os
import pickle

import numpy as np

import graph_eval as GE
import sparse_metrics as SM
import bootstrap as SB

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
HOME = os.path.expanduser("~")
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=150)
a = ap.parse_args()
sp = json.load(open(f"{R}/data/splits.json"))
allgt = pickle.load(open(f"{HOME}/exp_gcar_b123/graph_gt_val.pkl", "rb"))
idx = sp["val"][:: max(len(sp["val"]) // a.n, 1)][: a.n]
gts = [dict(pts=[np.asarray(p, np.float64) for p in allgt[i]["pts"]], edges=[tuple(e) for e in allgt[i]["edges"]], nodes=None) for i in idx]
rng = np.random.RandomState(0)


def sub(g, keep):
    """graph restricted to the pieces `keep`, edges remapped"""
    new = {k: j for j, k in enumerate(keep)}
    return dict(pts=[g["pts"][k] for k in keep], edges=[(new[i], new[j]) for i, j in g["edges"] if i in new and j in new], nodes=None)


def jitter(g, s):
    return dict(pts=[p + rng.normal(0, s, p.shape) for p in g["pts"]], edges=list(g["edges"]), nodes=None)


def offset(g, d):
    return dict(pts=[p + np.array([0.0, d]) for p in g["pts"]], edges=list(g["edges"]), nodes=None)


def drop(g, p):
    keep = [k for k in range(len(g["pts"])) if rng.rand() >= p]
    return sub(g, keep)


def false_lines(g, p):
    n = len(g["pts"])
    extra = [g["pts"][k] + np.array([5.0, 0.0]) for k in range(n) if rng.rand() < p]  # copies 5 m to the side: not near any lane (mostly)
    return dict(pts=list(g["pts"]) + extra, edges=list(g["edges"]), nodes=None)


def duplicate(g, p):
    extra = [g["pts"][k] + np.array([0.3, 0.0]) for k in range(len(g["pts"])) if rng.rand() < p]  # parallel duplicate within 1 m
    return dict(pts=list(g["pts"]) + extra, edges=list(g["edges"]), nodes=None)


def cut_edges(g, p):
    return dict(pts=list(g["pts"]), edges=[e for e in g["edges"] if rng.rand() >= p], nodes=None)


def simplify(g, eps):
    out = []
    for q in g["pts"]:
        q = np.asarray(q, np.float64)
        k = SM._dp_keep(q, eps) if len(q) > 2 else list(range(len(q)))
        out.append(q[k])
    return dict(pts=out, edges=list(g["edges"]), nodes=None)


def reverse(g):
    return dict(pts=[p[::-1].copy() for p in g["pts"]], edges=[(j, i) for i, j in g["edges"]], nodes=None)


def breaks(g, p, gap=2.0):
    """a share p of the pieces is cut in the middle into two unconnected parts with a gap of `gap` m (geometry nearly unchanged, connectivity broken)"""
    pts, edges, n = list(g["pts"]), list(g["edges"]), len(g["pts"])
    for k in range(n):
        q = np.asarray(g["pts"][k], np.float64)
        if rng.rand() >= p or len(q) < 4:
            continue
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
        if s[-1] < 3 * gap:
            continue
        mid = s[-1] / 2
        a, b = q[s <= mid - gap / 2], q[s >= mid + gap / 2]
        if len(a) < 2 or len(b) < 2:
            continue
        pts[k] = a
        pts.append(b)
        j = len(pts) - 1
        edges = [(j, e[1]) if e[0] == k else e for e in edges]  # successors now follow the second part
    return dict(pts=pts, edges=edges, nodes=None)


tests = [("identity", 0, lambda g: g), ("jitter sigma m", 0.25, lambda g: jitter(g, 0.25)), ("jitter sigma m", 0.5, lambda g: jitter(g, 0.5)), ("jitter sigma m", 1.0, lambda g: jitter(g, 1.0)),
         ("offset m", 0.5, lambda g: offset(g, 0.5)), ("offset m", 1.0, lambda g: offset(g, 1.0)), ("offset m", 2.0, lambda g: offset(g, 2.0)),
         ("drop share of lanes", 0.05, lambda g: drop(g, 0.05)), ("drop share of lanes", 0.10, lambda g: drop(g, 0.10)), ("drop share of lanes", 0.20, lambda g: drop(g, 0.20)), ("drop share of lanes", 0.40, lambda g: drop(g, 0.40)),
         ("false lanes (share added)", 0.05, lambda g: false_lines(g, 0.05)), ("false lanes (share added)", 0.10, lambda g: false_lines(g, 0.10)), ("false lanes (share added)", 0.20, lambda g: false_lines(g, 0.20)),
         ("parallel duplicates (share)", 0.10, lambda g: duplicate(g, 0.10)), ("parallel duplicates (share)", 0.30, lambda g: duplicate(g, 0.30)),
         ("fork/merge links cut (share)", 0.10, lambda g: cut_edges(g, 0.10)), ("fork/merge links cut (share)", 0.30, lambda g: cut_edges(g, 0.30)), ("fork/merge links cut (share)", 0.60, lambda g: cut_edges(g, 0.60)),
         ("Douglas-Peucker eps m", 0.3, lambda g: simplify(g, 0.3)), ("Douglas-Peucker eps m", 1.0, lambda g: simplify(g, 1.0)), ("Douglas-Peucker eps m", 2.0, lambda g: simplify(g, 2.0)),
         ("travel direction reversed", 1, reverse),
         ("lanes broken, 2 m gap (share)", 0.3, lambda g: breaks(g, 0.3)), ("lanes broken, 2 m gap (share)", 0.6, lambda g: breaks(g, 0.6))]
rows = []
for name, par, fn in tests:
    preds = [fn(g) for g in gts]
    m = SM.card_any(preds, gts, extra=True)
    m["len_ratio"] = SB.pooled(SB.frame_records(preds, gts))["len_ratio"]
    m["dir_acc"] = m.get("dir_acc", float("nan"))
    rows.append((name, par, m))
base = rows[0][2]
print(f"{len(gts)} validation ground-truth graphs; identity: F1@1 {base['F1@1.0']:.3f}, J {base['J']:.3f}, Sp {base['Sp']:.3f} (vert100 {base['vert100']:.1f}), S {base['S']:.3f}")
by = {}
for name, par, m in rows:
    by.setdefault(name, []).append((par, m))
# assertions: the metric must respond in the right direction
assert base["F1@1.0"] > 0.999 and base["J"] > 0.999 and base["false_len_frac"] < 1e-9
for name in ("jitter sigma m", "offset m", "drop share of lanes", "false lanes (share added)", "fork/merge links cut (share)"):
    s = [m["S"] for _, m in by[name]]
    assert all(s[i] >= s[i + 1] - 1e-9 for i in range(len(s) - 1)) and s[-1] < base["S"] - 1e-3, (name, s)
dup = by["parallel duplicates (share)"][-1][1]
assert dup["F1o2o@1.0"] < dup["F1@1.0"] - 0.02, "1:1 F1 must punish parallel duplicates, point F1 must not"
assert by["fork/merge links cut (share)"][-1][1]["J"] < 0.7 and by["fork/merge links cut (share)"][-1][1]["F1@1.0"] > 0.99, "cutting links must hit J only"
dpv = [m["vert100"] for _, m in by["Douglas-Peucker eps m"]]
assert dpv[0] > dpv[1] > dpv[2] and by["Douglas-Peucker eps m"][0][1]["F1@1.0"] > 0.99
assert by["travel direction reversed"][0][1].get("dir_acc", 1.0) < 0.05
# the metrics of topo_metrics close the blind spots
assert base["topo_F1"] > 0.99 and base["dup_frac"] < 0.02, (base["topo_F1"], base["dup_frac"])
brk = by["lanes broken, 2 m gap (share)"][-1][1]
assert brk["F1@1.0"] > 0.97 and brk["topo_R"] < base["topo_R"] - 0.10, "a broken lane must cost connectivity, not geometry"
assert dup["dup_frac"] > base["dup_frac"] + 0.10, "parallel duplicates must show in dup_frac"
assert by["travel direction reversed"][0][1]["topo_F1"] < 0.10, "reversed travel direction must kill connectivity"
assert by["fork/merge links cut (share)"][-1][1]["topo_R"] < base["topo_R"] - 0.05
assert by["drop share of lanes"][-1][1]["topo_R"] < base["topo_R"] - 0.10
print("assertions passed")

# table + the price of the noise band
lines = ["# Metric response (generated by evaluation/tests/test_metric_response.py)", "",
         f"Controlled errors on {len(gts)} validation ground-truth graphs (seeded; the errors are the 'prediction'). Identity: F1@1 {base['F1@1.0']:.3f}, J {base['J']:.3f}, FL {base['false_len_frac']:.3f}, vertices per 100 m {base['vert100']:.1f} (Douglas-Peucker count, no vertex ids), S {base['S']:.3f}.",
         "Question: when an experiment moves S by 0.010 (the noise band), what has happened to the graph?", "",
         "| error | size | F1@1 | F1@0.5 | 1:1 F1@1 | J | FL | vert/100 m | pred/GT length | dir acc | S | dS | dS / 0.010 | Q | topo P | topo R | topo F1 | dup | R@1 far |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for name, par, m in rows:
    lines.append(f"| {name} | {par} | {m['F1@1.0']:.3f} | {m['F1@0.5']:.3f} | {m['F1o2o@1.0']:.3f} | {m['J']:.3f} | {m['false_len_frac']:.3f} | {m['vert100']:.1f} | {m['len_ratio']:.2f} | {m.get('dir_acc', float('nan')):.2f} | {m['S']:.3f} | {m['S'] - base['S']:+.3f} | {(m['S'] - base['S']) / 0.010:+.1f} | {m['Q']:.3f} | {m['topo_P']:.3f} | {m['topo_R']:.3f} | {m['topo_F1']:.3f} | {m['dup_frac']:.3f} | {m['R@1_far']:.3f} |")
lines += ["", "## What a change of 0.010 in S is worth (linear interpolation over the sizes above)", ""]
for name in ("offset m", "drop share of lanes", "false lanes (share added)", "fork/merge links cut (share)"):
    pts = [(0.0, 0.0)] + [(float(p), m["S"] - base["S"]) for p, m in by[name]]
    pts.sort()
    x = None
    for (p0, d0), (p1, d1) in zip(pts[:-1], pts[1:]):
        if d0 > -0.010 >= d1:
            x = p0 + (p1 - p0) * (-0.010 - d0) / (d1 - d0)
            break
    lines.append(f"- {name}: dS = -0.010 at about {x:.3f}" if x is not None else f"- {name}: more than the largest size tested")
# local sensitivity of the composite to its parts
lines += ["", "## Composite S = 0.30 F1@1 + 0.20 F1@0.5 + 0.20 J + 0.15 Sp + 0.15 (1 - FL): reading it", "",
          "- A gain of 0.010 in S equals +0.033 in F1@1, or +0.050 in J, or +0.067 in (1 - FL), or +0.067 in Sp (about +0.8 vertices per 100 m saved at 13.5 vertices per 100 m: Sp = 14.3 / vert100, dSp/dvert = -14.3 / vert100^2 = -0.078 per vertex).",
          "- J is the part with most room: at J around 0.27 - 0.40 a +0.10 in J is worth +0.020 in S; F1@1 is at 0.84 - 0.88 and nearly saturated by comparison.",
          "- Sp is capped at 1.0 (vert100 <= 14.3): below that, fewer vertices no longer raise S; only F1, J and FL can.",
          "- **Blind spots of S** (all by construction, found by this test): parallel duplicates leave S unchanged (point F1 and FL cannot see them, vertices per 100 m is normalised by the predicted length); read 1:1 F1 and pred / GT length next to S. Travel direction enters S only through J (reversal: dir acc 0, J 0.18). Sparser than 14.3 vertices per 100 m is not rewarded (Sp capped at 1): the champion (13.5) is at the cap, so further sparsity can only cost.",
          "- **Jitter is expensive for DP-counted sparsity**: sigma 0.25 m on every sample multiplies the Douglas-Peucker vertex count by 5 (63 vs 13 per 100 m) and costs 0.12 in S although F1@1 stays 1.0 - for graphs without vertex ids the sparsity term measures smoothness, not vertex economy. Builders that emit vertices (node ids) are not affected.",
          "- Differences below 0.010 in S are ties by protocol; the paired scene bootstrap (`sparse_lanes/bootstrap.py`) says whether a smaller difference is still resolvable.",
          "",
          "## The added metrics (sparse_lanes/topo_metrics.py, 2026-10-25)",
          "",
          "- **topo P / R / F1** (directed reachability within 30 m from seeds every 5 m, 1:1 matching at 1 m): the connectivity a planner needs. It falls with broken lanes and cut links where F1 does not, and reversed travel direction drives it to zero.",
          "- **dup** (share of the predicted length running parallel within 0.75 m of a piece that is neither its neighbour nor its sibling): the parallel duplicates that S cannot see.",
          "- **R@1 far** (recall at 1 m beyond 40 m from the ego, with near / mid in the card): where along the range the misses are.",
          "- Reading rule: S and Q stay for comparison with the history; judge new work by Q at vert100 <= 14.3, topo F1, lanes found and R@1 far (with dup as a check), plus route success and ms per frame."]
open(f"{R}/evaluation/METRIC_RESPONSE.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("written evaluation/METRIC_RESPONSE.md")
