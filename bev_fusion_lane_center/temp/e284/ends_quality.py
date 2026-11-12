"""scratch (continuation of pair_geometry.py / p14_geometry_links.py, Mirko 2026-11-08): the pair classifier on exact GT ends gives AUC 0.995, but on the delivered long links geometry adds nothing (p14, gate failed): the delivered ENDS are not clean. How clean are they?
GT entries = starts, GT exits = ends of the connector runs of the lane GT (clusters of 1.5 m / 40 deg; the oracle_junction.py definition; frames with >= 1 run, |x|, |y| < 48). Delivered: baseline B + the hard close-node step 1.5 m (E275 read-out), full dump, VAL 873.
 RECALL    share of the GT entries (exits) with a kept node within 2.0 m whose heading is within 30 deg of the GT heading: (any) any kept node, (flag) a kept node with the flag stop (entries) / exit (exits) >= 0.5
 PRECISION share of the kept nodes with the flag stop (exit) >= 0.5, and of the SOURCES (TARGETS) of the delivered long links, that lie within 2.0 m / 30 deg of a GT entry (exit)
 COUNTS    per frame: GT entries / exits against the flagged nodes and the distinct sources / targets of the long links
    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 bash tools/env/_py14.sh temp/e284/ends_quality.py"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as QR  # noqa: E402
import qtargets as QT  # noqa: E402

SPEC = QR.parse("close:1.5@0.5")
ANGTOL = float(os.environ.get("ANGTOL", "30"))               # heading tolerance in degrees; 181 = position only
POSTOL = float(os.environ.get("POSTOL", "2.0"))              # position tolerance in metres
FI = QT.FI
frames = QE.val_frames()


def cluster(items, tol_pos=1.5, tol_ang=40.0):
    reps = []
    for pos, tan in items:
        for rp, rt in reps:
            if np.linalg.norm(pos - rp) <= tol_pos and np.degrees(np.arccos(np.clip(tan @ rt, -1, 1))) <= tol_ang:
                break
        else:
            reps.append((pos, tan))
    return reps


def near(P, H, gp, gh, tol=None, ang=None):
    ang = ANGTOL if ang is None else ang
    tol = POSTOL if tol is None else tol
    """for every row of P (positions) / H (unit headings): is there a GT end within tol and heading within ang"""
    if not len(gp) or not len(P):
        return np.zeros(len(P), bool)
    t = cKDTree(gp)
    out = np.zeros(len(P), bool)
    for i, c in enumerate(t.query_ball_point(P, tol)):
        out[i] = bool(c) and bool((gh[c] @ H[i] >= np.cos(np.radians(ang))).any())
    return out


res = {}
for SEED in (0, 1):
    dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{SEED}.pkl"), "rb"))
    A = dict(gt_ent=0, gt_ext=0, ent_any=0, ent_flag=0, ext_any=0, ext_flag=0, n_stop=0, stop_true=0, n_exit=0, exit_true=0, n_src=0, src_true=0, n_dst=0, dst_true=0, frames=0)
    for n, (f, d) in enumerate(zip(frames, dump)):
        g = f.lane()
        ent, ext = [], []
        for pc in g.get("pieces", []):
            q = np.asarray(pc["pts"], float).reshape(-1, 2)
            if len(q) < 4:
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
                        if (np.abs(q[k]) < 48).all():
                            ent.append((q[k], tan[k]))
                        if (np.abs(q[j]) < 48).all():
                            ext.append((q[j], tan[j]))
                    k = j + 1
                else:
                    k += 1
        ent, ext = cluster(ent), cluster(ext)
        if not ent and not ext:
            f._lane = f._ref = None
            continue
        A["frames"] += 1
        ep, eh = np.array([e[0] for e in ent]).reshape(-1, 2), np.array([e[1] for e in ent]).reshape(-1, 2)
        xp, xh = np.array([e[0] for e in ext]).reshape(-1, 2), np.array([e[1] for e in ext]).reshape(-1, 2)
        pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                             links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), SPEC)
        graph, bk = QE.readout(pred, 0.5, 0.5)
        keep = bk["keep"]
        pos = d["pos"]
        u = d["dirv"] / (np.linalg.norm(d["dirv"], axis=1, keepdims=True) + 1e-9)
        fl = d["flags"]
        kp = np.flatnonzero(keep)
        stop = kp[fl[kp, FI["stop"]] >= 0.5]
        exi = kp[fl[kp, FI["exit"]] >= 0.5]
        L = np.asarray(bk["links"], int)
        lg = np.asarray(d["link_long"], bool)
        Ll = L[lg[L]] if len(L) else L
        src, dst = np.unique(d["link_src"][Ll]), np.unique(d["link_dst"][Ll])
        # recall
        for gp_, gh_, key in ((ep, eh, "ent"), (xp, xh, "ext")):
            if not len(gp_):
                continue
            A[f"gt_{key}"] += len(gp_)
            tr = cKDTree(gp_)
            for pool, nm in ((kp, "any"), (stop if key == "ent" else exi, "flag")):
                hit = np.zeros(len(gp_), bool)
                if len(pool):
                    nodes_t = cKDTree(pos[pool])
                    for i, c in enumerate(nodes_t.query_ball_point(gp_, POSTOL)):
                        if c and (u[pool][c] @ gh_[i] >= np.cos(np.radians(ANGTOL))).any():
                            hit[i] = True
                A[f"{key}_{nm}"] += int(hit.sum())
        # precision
        for nodes, gp_, gh_, nk, tk in ((stop, ep, eh, "n_stop", "stop_true"), (exi, xp, xh, "n_exit", "exit_true"), (src, ep, eh, "n_src", "src_true"), (dst, xp, xh, "n_dst", "dst_true")):
            if not len(nodes):
                continue
            A[nk] += len(nodes)
            A[tk] += int(near(pos[nodes], u[nodes], gp_, gh_).sum())
        f._lane = f._ref = None
        if (n + 1) % 200 == 0:
            print(f"  seed {SEED} frame {n + 1}/{len(frames)}", flush=True)
    r = dict(frames_with_junction=A["frames"], gt_entries=A["gt_ent"], gt_exits=A["gt_ext"], entry_recall_any_node=A["ent_any"] / max(A["gt_ent"], 1), entry_recall_stop_flag=A["ent_flag"] / max(A["gt_ent"], 1),
             exit_recall_any_node=A["ext_any"] / max(A["gt_ext"], 1), exit_recall_exit_flag=A["ext_flag"] / max(A["gt_ext"], 1),
             stop_flag_precision=A["stop_true"] / max(A["n_stop"], 1), exit_flag_precision=A["exit_true"] / max(A["n_exit"], 1),
             long_link_source_precision=A["src_true"] / max(A["n_src"], 1), long_link_target_precision=A["dst_true"] / max(A["n_dst"], 1),
             per_frame=dict(gt_entries=A["gt_ent"] / A["frames"], gt_exits=A["gt_ext"] / A["frames"], stop_nodes=A["n_stop"] / A["frames"], exit_nodes=A["n_exit"] / A["frames"], long_sources=A["n_src"] / A["frames"], long_targets=A["n_dst"] / A["frames"]))
    res[f"seed {SEED}"] = r
    print(f"seed {SEED}: {r['frames_with_junction']} frames with a junction, GT entries {r['gt_entries']}, exits {r['gt_exits']}")
    print(f"   entry recall: any kept node {100 * r['entry_recall_any_node']:.1f} %, with the stop flag {100 * r['entry_recall_stop_flag']:.1f} %   |   exit recall: any kept node {100 * r['exit_recall_any_node']:.1f} %, with the exit flag {100 * r['exit_recall_exit_flag']:.1f} %")
    print(f"   precision: stop-flag nodes {100 * r['stop_flag_precision']:.1f} %, exit-flag nodes {100 * r['exit_flag_precision']:.1f} %, long-link sources {100 * r['long_link_source_precision']:.1f} %, long-link targets {100 * r['long_link_target_precision']:.1f} %")
    print("   per frame:", {k: round(v, 1) for k, v in r["per_frame"].items()})
json.dump(res, open(os.path.join(ROOT, "temp", "e284", "ends_quality" + ("" if ANGTOL == 30 else f"_ang{ANGTOL:g}") + ("" if POSTOL == 2.0 else f"_pos{POSTOL:g}") + ".json"), "w"), indent=1)
