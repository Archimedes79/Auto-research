"""scratch (Mirko 2026-11-08: 'work alone, improve the extraction systematically through error analysis and fixes'): WHERE does the delivered lane graph of the current best (baseline B = E273 head + the hard close-node step 1.5 m of E275,
read out at 0.5) lose length? CPU, no model, from the full prediction dump (`temp/e276/predfull_e273_p_s<seed>.pkl`: nodes e >= 0.2, links p >= 0.05), VAL 873, lane GT of autogt.
RECALL side: every GT lane point (every 0.5 m, |x|, |y| < 48; kind lane / connector, zone junction = a connector point inside an intersection, mouth = within 8 m of one, road = the rest) is COVERED when a delivered polyline point lies within 1.0 m
(the R@1.0 of the scorer). A missed point gets ONE cause, first match of:
  R1 offset            a delivered point lies within 2.5 m (the graph is there, shifted by 1 - 2.5 m)
  R2 link below thr    a kept node lies within 4 m and a link with p in [0.05, 0.5) between dump nodes has its Hermite curve within 1.0 m of the point (the link head knew it, the read-out cut it)
  R3 link not proposed a kept node lies within 4 m, no such curve (nodes there, nothing connects them along this GT line)
  R4 node below thr    no kept node within 4 m but a dump node (e in [0.2, 0.5) or removed by the close-node step) within 4 m
  R5 no node           no dump node within 4 m
PRECISION side: every delivered polyline point (every 0.5 m) is FALSE when no GT lane point lies within 1.0 m (the P@1.0): P1 offset (within 2.5 m of GT), P2 invented (farther); by piece kind (section chain / short link / long link) and by zone
(within 8 m of a GT connector point inside an intersection or not). Doubles: a point of a polyline with a point of ANOTHER polyline within 0.7 m (counted for the polyline with the higher index only).
Also per GT connector run: found when >= 80 % of its points are covered.
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 bash tools/env/_py14.sh temp/e284/err_taxonomy.py <seed> [--rows N] [--spec close:1.5@0.5]"""
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as QR  # noqa: E402
import qtargets as QT  # noqa: E402

SEED = int(sys.argv[1])
N_ROWS = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 0
SPEC = QR.parse(sys.argv[sys.argv.index("--spec") + 1] if "--spec" in sys.argv else "close:1.5@0.5")
BOX = 48.0
frames = QE.val_frames()
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{SEED}.pkl"), "rb"))
if N_ROWS:
    frames, dump = frames[:N_ROWS], dump[:N_ROWS]
ZONES = ("junction", "mouth", "road")
RC = ("R1 offset 1-2.5 m", "R2 link below threshold", "R3 link not proposed", "R4 node below threshold / removed", "R5 no node proposed")
PC = ("P1 offset 1-2.5 m", "P2 invented")
rec = {(z, k, c): 0.0 for z in ZONES for k in ("lane", "connector") for c in ("covered",) + RC}
prec = {(z, pk, c): 0.0 for z in ("junction zone", "elsewhere") for pk in ("section", "short link", "long link") for c in ("true",) + PC}
dup_len = {pk: 0.0 for pk in ("section", "short link", "long link")}
conn_found = conn_total = 0
t0 = time.time()
for n, (f, d) in enumerate(zip(frames, dump)):
    g = f.lane()
    # ---- GT points with kind and zone
    P, K, J = [], [], []
    runs = []
    for pc in g.get("pieces", []):
        q = np.asarray(pc["pts"], float).reshape(-1, 2)
        if len(q) < 2:
            continue
        conn = np.zeros(len(q), bool)
        inter = np.zeros(len(q), bool)
        for s in pc["segments"]:
            if s["kind"] == "connector":
                conn[s["i0"]: s["i1"] + 1] = True
                if s["in_intersection"]:
                    inter[s["i0"]: s["i1"] + 1] = True
        P.append(q)
        K.append(conn)
        J.append(inter)
        k = 0
        while k < len(q):
            if conn[k]:
                j = k
                while j + 1 < len(q) and conn[j + 1]:
                    j += 1
                if j - k + 1 >= 3:
                    runs.append(q[k: j + 1])
                k = j + 1
            else:
                k += 1
    P, K, J = np.concatenate(P), np.concatenate(K), np.concatenate(J)
    inb = (np.abs(P) < BOX).all(1)
    jt = cKDTree(P[J]) if J.any() else None
    dj = jt.query(P)[0] if jt is not None else np.full(len(P), 1e9)
    zone = np.where(J, 0, np.where(dj <= 8.0, 1, 2))
    gt_tree = cKDTree(P)
    # ---- the delivered graph
    pred = dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"]))
    q_ = QR.apply(pred, SPEC)
    graph, bk = QE.readout(q_, 0.5, 0.5)
    keep = bk["keep"]
    lines = [np.asarray(l, float) for l in graph.lines]
    if lines:
        D = np.concatenate(lines)
        lid = np.concatenate([np.full(len(l), i) for i, l in enumerate(lines)])
        pk_of = []
        for i, kd in enumerate(bk["kind"]):
            if kd == "section":
                pk_of.append("section")
            else:
                pk_of.append("long link" if bool(np.asarray(d["link_long"])[bk["piece_links"][i][0]]) else "short link")
        pk = np.array([pk_of[i] for i in lid])
        dtree = cKDTree(D)
    else:
        D, lid, pk, dtree = np.zeros((0, 2)), np.zeros(0, int), np.zeros(0, object), None
    # ---- recall
    dd = dtree.query(P)[0] if dtree is not None else np.full(len(P), 1e9)
    miss = inb & (dd > 1.0)
    cov = inb & ~miss
    for zi, z in enumerate(ZONES):
        for kn, kv in (("lane", False), ("connector", True)):
            rec[(z, kn, "covered")] += 0.5 * float((cov & (zone == zi) & (K == kv)).sum())
    if miss.any():
        kept_pos = d["pos"][keep]
        ktree = cKDTree(kept_pos) if len(kept_pos) else None
        cand_mask = ~keep
        ctree = cKDTree(d["pos"][cand_mask]) if cand_mask.any() else None
        mi = np.flatnonzero(miss)
        dk = ktree.query(P[mi])[0] if ktree is not None else np.full(len(mi), 1e9)
        dc = ctree.query(P[mi])[0] if ctree is not None else np.full(len(mi), 1e9)
        need_links = np.flatnonzero((dd[mi] > 2.5) & (dk <= 4.0))
        sub_tree = None
        if len(need_links):
            pl = np.asarray(d["link_p"], float)
            sel = np.flatnonzero((pl >= 0.05) & (pl < 0.5) & (np.asarray(d["link_src"]) != np.asarray(d["link_dst"])))
            pts_ = []
            for l in sel:
                a, b = int(d["link_src"][l]), int(d["link_dst"][l])
                pts_.append(QT.hermite(d["pos"][a], d["dirv"][a], d["pos"][b], d["dirv"][b], float(d["link_a"][l]), float(d["link_b"][l]), 0.5))
            if pts_:
                sub_tree = cKDTree(np.concatenate(pts_))
        for j, i in enumerate(mi):
            if dd[i] <= 2.5:
                c = RC[0]
            elif dk[j] <= 4.0:
                c = RC[1] if (sub_tree is not None and sub_tree.query(P[i])[0] <= 1.0) else RC[2]
            elif dc[j] <= 4.0:
                c = RC[3]
            else:
                c = RC[4]
            rec[(ZONES[zone[i]], "connector" if K[i] else "lane", c)] += 0.5
    for r in runs:
        r = r[(np.abs(r) < BOX).all(1)]
        if len(r) >= 3:
            conn_total += 1
            conn_found += int((dtree.query(r)[0] <= 1.0).mean() >= 0.8) if dtree is not None else 0
    # ---- precision
    if len(D):
        dg, _ = gt_tree.query(D)
        inD = (np.abs(D) < BOX).all(1)
        zD = np.where(jt.query(D)[0] <= 8.0, "junction zone", "elsewhere") if jt is not None else np.full(len(D), "elsewhere")
        for i in np.flatnonzero(inD):
            c = "true" if dg[i] <= 1.0 else (PC[0] if dg[i] <= 2.5 else PC[1])
            prec[(zD[i], pk[i], c)] += 0.5
        # doubles
        for li in range(len(lines)):
            m = np.flatnonzero((lid == li) & inD)
            if not len(m):
                continue
            nb = dtree.query_ball_point(D[m], 0.7)
            hit = sum(any(lid[x] > li for x in nbs) for nbs in nb)           # a point that another polyline of HIGHER index also covers: counted once per pair
            dup_len[pk[m[0]]] += 0.5 * hit
    f._lane = f._ref = None
    if (n + 1) % 100 == 0:
        print(f"  frame {n + 1}/{len(frames)}  {time.time() - t0:.0f} s", flush=True)
gt_total = sum(rec.values())
del_total = sum(prec.values())
out = dict(seed=SEED, spec=sys.argv[sys.argv.index("--spec") + 1] if "--spec" in sys.argv else "close:1.5@0.5", frames=len(frames), gt_length_m=gt_total, delivered_length_m=del_total,
           recall={f"{z} | {k} | {c}": v for (z, k, c), v in rec.items()}, precision={f"{z} | {pk} | {c}": v for (z, pk, c), v in prec.items()}, double_length_m=dup_len,
           connectors_found=conn_found, connectors_total=conn_total)
json.dump(out, open(os.path.join(ROOT, "temp", "e284", f"err_taxonomy_s{SEED}{'_rows' + str(N_ROWS) if N_ROWS else ''}.json"), "w"), indent=1)
print(f"GT length {gt_total:.0f} m, delivered {del_total:.0f} m, connectors found {conn_found} / {conn_total}")
print("RECALL (share of the GT length), by zone | kind")
for z in ZONES:
    for k in ("lane", "connector"):
        tot = sum(rec[(z, k, c)] for c in ("covered",) + RC)
        if tot < 1:
            continue
        print(f"  {z:9s} {k:9s} {tot:8.0f} m = {100 * tot / gt_total:5.1f} % of GT | covered {100 * rec[(z, k, 'covered')] / tot:5.1f} % | " + " | ".join(f"{c[:2]} {100 * rec[(z, k, c)] / tot:4.1f} %" for c in RC))
print("losses by cause, all zones (share of the GT length):", " | ".join(f"{c} {100 * sum(rec[(z, k, c)] for z in ZONES for k in ('lane', 'connector')) / gt_total:4.1f} %" for c in RC))
print("PRECISION (share of the delivered length), by zone | piece kind")
for z in ("junction zone", "elsewhere"):
    for pk_ in ("section", "short link", "long link"):
        tot = sum(prec[(z, pk_, c)] for c in ("true",) + PC)
        if tot < 1:
            continue
        print(f"  {z:13s} {pk_:10s} {tot:8.0f} m = {100 * tot / del_total:5.1f} % of delivered | true {100 * prec[(z, pk_, 'true')] / tot:5.1f} % | " + " | ".join(f"{c[:2]} {100 * prec[(z, pk_, c)] / tot:4.1f} %" for c in PC))
print("doubles (m, share of delivered):", {k: (round(v), round(100 * v / del_total, 1)) for k, v in dup_len.items()})
