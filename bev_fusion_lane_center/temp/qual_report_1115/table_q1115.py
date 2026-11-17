"""scratch (2026-11-15): per-frame table from the stored counts pickles of E324 s (head) / E312 (head) / B (base), seed 0, VAL 873."""
import os, sys, json, pickle
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training/grid", "training", "training/data", "evaluation/sparse_lanes", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import qgraph_eval as QE
fr = QE.val_frames()
c324 = pickle.load(open(f"{ROOT}/runs/e324_s_s0/ends_card_snap_thr0.1.counts.pkl", "rb"))
c312 = pickle.load(open(f"{ROOT}/runs/e312_p_s0/ends_card_snap_thr0.1.counts.pkl", "rb"))
assert c324["scenes"] == [f.scene for f in fr] == c312["scenes"]
def f1(c):
    t = c["topo"]["topo"]; d = t[1] + t[2]
    return float(2 * t[0] / d) if d > 0 else float("nan")
def far_r(c):
    r = c["topo"]["rng"][2]; return float(r[0] / r[1]) if r[1] > 0 else float("nan")
def own_r(c):
    r = c["tier"]["rec"][0] if c.get("tier") is not None else None
    return float(r[0] / r[1]) if r is not None and r[1] > 0 else float("nan")
def own_p(c):
    p = c["tier"]["prec"][1] if c.get("tier") is not None else None
    return float(p[0] / p[1]) if p is not None and p[1] > 0 else float("nan")
def own_n(c):
    r = c["tier"]["rec"][0] if c.get("tier") is not None else None
    return 0.5 * float(r[1]) if r is not None else float("nan")
def dup(c):
    u = c["topo"]["dup"]; return float(u[0] / u[1]) if u[1] > 0 else float("nan")
def jf(c):
    n_p, n_g, tp = c["lane"]["junc"]; return float(2 * tp / (n_p + n_g)) if n_p + n_g > 0 else float("nan")
def lf(c):
    rs = c["rec"]["rs"]; return int(rs["lanes_found"]), int(rs["lanes_total"])
rows = []
for k, f in enumerate(fr):
    r = dict(k=k, row=f.row, scene=f.scene, token=f.token)
    for nm, cc in (("B", c324["base"][k]), ("E312", c312["head"][k]), ("E324s", c324["head"][k])):
        r[nm + "_topo"] = f1(cc); r[nm + "_far"] = far_r(cc); r[nm + "_ownR"] = own_r(cc); r[nm + "_ownP"] = own_p(cc)
        r[nm + "_dup"] = dup(cc); r[nm + "_jf"] = jf(cc); r[nm + "_lf"], r[nm + "_lt"] = lf(cc)
        r[nm + "_ngt"] = float(cc["topo"]["topo"][2]); r[nm + "_npred"] = float(cc["topo"]["topo"][1])
        r[nm + "_dupn"] = float(cc["topo"]["dup"][0]); r[nm + "_predlen"] = float(cc["rec"]["rs"]["pred_len"])
    r["ownlen"] = own_n(c324["head"][k])
    r["n_junc_gt"] = float(c324["head"][k]["lane"]["junc"][1])
    rows.append(r)
json.dump(rows, open(f"{ROOT}/temp/qual_report_1115/perframe_table.json", "w"))
import statistics as S
for nm in ("B", "E312", "E324s"):
    v = [r[nm + "_topo"] for r in rows if r[nm + "_topo"] == r[nm + "_topo"]]
    print(nm, "frame-mean topo F1", round(S.mean(v), 4), "n", len(v))
# pooled check against the cards
for nm, c in (("B", [x for x in c324["base"]]), ("E312", c312["head"]), ("E324s", c324["head"])):
    t = sum(x["topo"]["topo"] for x in c); P, R = t[0] / t[1], t[0] / t[2]
    print(nm, "pooled topo F1", round(2 * P * R / (P + R), 4))
print("scene counts", sorted({r['scene'] for r in rows}))
