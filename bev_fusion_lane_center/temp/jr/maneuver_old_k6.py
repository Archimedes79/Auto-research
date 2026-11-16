"""maneuver coverage: the number that asks what the planner needs and what the pictures show (scratch, CPU).
A GT MANEUVER is one entry -> exit connection of a junction (the connector runs of the lane GT grouped by their entry / exit cluster, `qends`); it is REALISED in a delivered graph when at least 90 % of the points of its GT run (every 1 m, inside the 100 m box)
have a delivered polyline point within 1.5 m whose direction differs by at most 40 degrees (the opposite lane or a crossing lane does not count). No pairing, no tolerance of 4 m / 30 degrees at the ends: only 'is the lane drawn there, in that direction'.
Also: the junction PRECISION = delivered points within 8 m of a GT connector run that have a GT lane point within 1.5 m (any direction) / all such delivered points.
Variants on the same frames: plain (B + hard close-node step), the connectors of a pair-head run at p >= 0.5 / 0.2 / 0.1, and the ORACLE pair classifier on its keypoints (logit +20 for a candidate whose two keypoints match the two ends of a GT pair, -20 else).
usage (through tools/env/_py14.sh): maneuver.py <pair run> <seed>   -> temp/jr/media/errors/maneuver_<run>.json + maneuver.png"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("BFLC_ROOT", ROOT)
for sub in ("evaluation", "training", "training/grid", "training/data", "tools/viz", ".."):
    sys.path.insert(0, os.path.join(ROOT, sub))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import jgt as J  # noqa: E402
import jrule_control as JR  # noqa: E402
import qends as QENDS  # noqa: E402
import qends_eval as QEV  # noqa: E402
import qgraph_eval as QGE  # noqa: E402
import qgraph_readout as QR  # noqa: E402
import qgraph_view as QV  # noqa: E402
import qtargets as QT  # noqa: E402
import qjunction as QJ  # noqa: E402
import snapconn as SC  # noqa: E402

run, seed = sys.argv[1], int(sys.argv[2])
OUT = os.path.join(ROOT, "temp", "jr", "media", "errors")
os.makedirs(OUT, exist_ok=True)
BOX, TOL, ANG, NEED = 50.0, 1.5, 40.0, 0.9
ALL = ["plain", "p>=0.5", "p>=0.2", "p>=0.1", "oracle", "oracle+fit", "p>=0.5 snap", "p>=0.1 snap", "oracle snap", "oracle+fit snap"]
VARIANTS = [v for v in ALL if len(sys.argv) <= 3 or v in sys.argv[3:]]         # optional: the variants to compute (default all)

frames = QGE.val_frames()
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{seed}.pkl"), "rb"))
ends = {f["row"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "rb"))}
spec = QR.parse("close:1.5@0.5")


def maneuvers(g):
    """[(entry cluster, exit cluster, run points (n, 2) sampled every ~1 m, tangents)] of the GT connector runs, the longest run per (entry, exit)"""
    runs = []
    for pc in g.get("pieces", []):
        q = np.asarray(pc["pts"], float).reshape(-1, 2)
        if len(q) < 2:
            continue
        tan = QT._tangents(q)
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
                    runs.append((q[k], tan[k], q[j], tan[j], q[k: j + 1], tan[k: j + 1]))
                k = j + 1
            else:
                k += 1
    if not runs:
        return []
    _, eid = QENDS.cluster([(r[0], r[1]) for r in runs])
    _, xid = QENDS.cluster([(r[2], r[3]) for r in runs])
    best = {}
    for r, a, b in zip(runs, eid, xid):
        if (a, b) not in best or len(r[4]) > len(best[(a, b)][4]):
            best[(a, b)] = r
    out = []
    for (a, b), r in best.items():
        P = QGE._densify(r[4], 1.0)
        T = QT._tangents(P)
        m = (np.abs(P) < BOX).all(1)
        if m.sum() >= 3:
            out.append((a, b, P[m], T[m]))
    return out


def delivered(raw):
    """densified delivered points (0.5 m) with their tangents"""
    Ps, Ts = [], []
    for p in raw["pts"]:
        if len(p) < 2:
            continue
        d = QGE._densify(p)
        Ps.append(d)
        Ts.append(QT._tangents(d))
    if not Ps:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(Ps), np.concatenate(Ts)


def realised(P, T, tree, run_pts, run_tan):
    """share of the GT run points that have a delivered point within TOL and a direction within ANG"""
    if tree is None:
        return 0.0
    d, ix = tree.query(run_pts, k=6, distance_upper_bound=TOL)
    ok = np.zeros(len(run_pts), bool)
    cosmin = np.cos(np.radians(ANG))
    for c in range(d.shape[1]):
        v = np.isfinite(d[:, c])
        if not v.any():
            continue
        t2 = T[ix[v, c]]
        ok[v] |= (t2 * run_tan[v]).sum(1) >= cosmin
    return float(ok.mean())


def graph_variants(f, d, e, jg):
    pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                         links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
    graph, bk = QGE.readout(pred, 0.5, 0.5)
    out = {"plain": QGE.to_raw(graph)}
    k = e["kp_pos"].shape[0] // 2
    logit_or = np.full_like(e["logit"], -20.0)
    scales_or = {}
    if jg is not None:
        K = e["logit"].shape[0]
        ve = e["valid"][0] & (e["kp_p"][:K] >= 0.5)
        vx = e["valid"][1] & (e["kp_p"][K:] >= 0.5)
        ma, mb = JR._match(e["kp_pos"][:K], e["kp_dir"][:K], ve, jg["ent"]), JR._match(e["kp_pos"][K:], e["kp_dir"][K:], vx, jg["ext"])
        sc_all = QJ.fit_pair_scales(jg)                                                  # (npair, 2): the handle scales fitted to the GT connector of every true pair
        for ip, pr in enumerate(jg["pairs"]):
            a, b = int(pr[0]), int(pr[1])
            if ma[a] >= 0 and mb[b] >= 0 and e["allowed"][ma[a], mb[b]]:
                logit_or[ma[a], mb[b]] = 20.0
                scales_or[(int(ma[a]), int(mb[b]))] = tuple(float(x) if np.isfinite(x) else 1.0 for x in sc_all[ip])
    table = (("p>=0.5", e["logit"], 0.5), ("p>=0.2", e["logit"], 0.2), ("p>=0.1", e["logit"], 0.1), ("oracle", logit_or, 0.5), ("oracle+fit", logit_or, 0.5),
             ("p>=0.5 snap", e["logit"], 0.5), ("p>=0.1 snap", e["logit"], 0.1), ("oracle snap", logit_or, 0.5), ("oracle+fit snap", logit_or, 0.5))
    for name, lg, thr in [x for x in table if x[0] in VARIANTS]:
        snap = name.endswith(" snap")
        base = name[:-5] if snap else name
        p = 1.0 / (1.0 + np.exp(-lg)) * e["allowed"]
        ia, ib = np.nonzero(p >= thr)
        prs = [(e["kp_pos"][i], e["kp_dir"][i], e["kp_pos"][k + j], e["kp_dir"][k + j]) + (scales_or[(int(i), int(j))] if base == "oracle+fit" else (tuple(e["scale"][i, j]) if (e.get("scale") is not None and base.startswith("p")) else ())) for i, j in zip(ia, ib)]
        add = SC.add_connectors_snap if snap else QEV.add_connectors
        raw, _, _ = add(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), prs)
        out[name] = raw
    return out


per = {v: [] for v in VARIANTS}
rows = []                                  # one row per GT maneuver: ends found by the keypoints, length, turn, realised per variant          # per junction frame: (realised, total, precision hits, precision total)
tok = []
t0 = __import__("time").time()
for b, (f, d) in enumerate(zip(frames, dump)):
    g = QV.load_gt(f.token)
    mans = maneuvers(g)
    if not mans:
        continue
    jg = J.junction_gt(g)
    e = ends[f.row]
    gv = graph_variants(f, d, e, jg)
    tok.append(f.token)
    junction_pts = np.concatenate([m[2] for m in mans])
    jtree = cKDTree(junction_pts)
    ref = f.ref()
    for v in VARIANTS:
        P, T = delivered(gv[v])
        tree = cKDTree(P) if len(P) else None
        real = [realised(P, T, tree, m[2], m[3]) >= NEED for m in mans]
        if v == "plain":
            base_idx = len(rows)
            kp, kd = e["kp_pos"], e["kp_dir"]
            K = e["logit"].shape[0]
            ve = e["valid"][0] & (e["kp_p"][:K] >= 0.5)
            vx = e["valid"][1] & (e["kp_p"][K:] >= 0.5)

            def near_kp(kpos, kdir, valid, pt, tn):
                if not valid.any():
                    return False
                d_ = np.linalg.norm(kpos[valid] - pt, axis=1)
                c_ = (kdir[valid] * tn).sum(1)
                return bool(((d_ <= 4.0) & (c_ >= np.cos(np.radians(30)))).any())
            for m in mans:
                ang = float(np.degrees(np.arctan2(m[3][0][0] * m[3][-1][1] - m[3][0][1] * m[3][-1][0], (m[3][0] * m[3][-1]).sum())))
                rows.append(dict(token=f.token, length=float(np.linalg.norm(m[2][-1] - m[2][0])), run_len=float(len(m[2])), turn=ang, range=float(np.linalg.norm(m[2][0])),
                                 found_e=near_kp(kp[:K], kd[:K], ve, m[2][0], m[3][0]), found_x=near_kp(kp[K:], kd[K:], vx, m[2][-1], m[3][-1]), real={}))
        for i_, r_ in enumerate(real):
            rows[base_idx + i_]["real"][v] = bool(r_)
        near = P[(np.abs(P) < BOX).all(1)]
        near = near[jtree.query(near)[0] <= 8.0] if len(near) else near
        hit = int((ref.tree.query(near)[0] <= TOL).sum()) if len(near) else 0
        per[v].append((int(sum(real)), len(mans), hit, len(near)))
    if (b + 1) % 100 == 0:
        print(f"  {b + 1}/{len(frames)}  {__import__('time').time() - t0:.0f} s", flush=True)

res = {}
for v in VARIANTS:
    a = np.array(per[v])
    res[v] = dict(maneuvers=int(a[:, 1].sum()), realised=int(a[:, 0].sum()), recall=float(a[:, 0].sum() / a[:, 1].sum()), frame_recall_mean=float(np.mean(a[:, 0] / a[:, 1])),
                  frames_none=int((a[:, 0] == 0).sum()), frames_all=int((a[:, 0] == a[:, 1]).sum()), junction_precision=float(a[:, 2].sum() / max(1, a[:, 3].sum())), frames=len(a))
json.dump(dict(run=run, seed=seed, tol=TOL, ang=ANG, need=NEED, rows=rows, variants=res, per_frame={v: per[v] for v in VARIANTS}, tokens=tok), open(os.path.join(OUT, f"maneuver_{run}.json"), "w"))
for v in VARIANTS:
    r = res[v]
    print(f"{v:8s} maneuvers realised {r['realised']}/{r['maneuvers']} = {r['recall']:.3f}; per-frame mean {r['frame_recall_mean']:.3f}; frames with none {r['frames_none']}, with all {r['frames_all']} of {r['frames']}; junction precision {r['junction_precision']:.3f}")

fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
cols = {"plain": "#7f7f7f", "p>=0.5": "#2ca02c", "p>=0.2": "#17becf", "p>=0.1": "#1f77b4", "oracle": "#ff7f0e", "oracle+fit": "#d62728", "p>=0.5 snap": "#006400", "p>=0.1 snap": "#00008b", "oracle snap": "#b8860b", "oracle+fit snap": "#8b0000"}
cols = {k_: v_ for k_, v_ in cols.items() if k_ in VARIANTS}
ax = axs[0]
for v in VARIANTS:
    a = np.array(per[v])
    ax.hist(a[:, 0] / a[:, 1], bins=np.linspace(0, 1, 21), histtype="step", lw=1.8, color=cols[v], label=f"{v}: {res[v]['recall']:.2f}")
ax.set_xlabel("share of the GT maneuvers of a frame that are realised in the delivered graph")
ax.set_ylabel(f"junction frames ({res['plain']['frames']})")
ax.set_title("maneuver coverage per frame (legend: pooled over all maneuvers)")
ax.legend(fontsize=8)
ax = axs[1]
ax.bar(range(len(VARIANTS)), [res[v]["recall"] for v in VARIANTS], color=[cols[v] for v in VARIANTS])
ax.plot(range(len(VARIANTS)), [res[v]["junction_precision"] for v in VARIANTS], "ko-", label="precision of the delivered points near junctions")
ax.set_xticks(range(len(VARIANTS)))
ax.set_xticklabels(VARIANTS)
ax.set_ylim(0, 1)
ax.set_title("maneuver recall (bars) and junction precision (line)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"maneuver_{run}.png"), dpi=80)
print("done")
