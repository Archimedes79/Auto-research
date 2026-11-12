"""Maneuver coverage of a delivered lane graph (E310): the number that asks what the planner needs and what the pictures show, next to the identity-strict pair numbers (`jpair_pr.py`) and the pooled topology scores (`qends_eval.py score`).

    wsl bash tools/env/_py14.sh jmaneuver.py score --run e308_t_s0 --seed 0 [--variants plain p0.5 p0.1 oracle oracle+fit] [--rows] [--chord] [--max N]     CPU, VAL 873, from runs/<run>/ends_val.pkl (`jends_eval.py dump`) and B's full dump of the seed
                                                                                                                                      -> runs/<run>/maneuver_val.json
    wsl bash tools/env/_py14.sh jmaneuver.py selftest

A GT MANEUVER is one entry -> exit connection of a junction: the connector runs of the lane GT grouped by their entry / exit cluster (`qends.cluster`, as `qends.end_table`), the longest run per (entry, exit). It is REALISED in a delivered graph when at least 90 % of the
points of its GT run (the GT points, spacing 0.5 m, inside the 100 m box; at least 3 points) have a delivered polyline point within 1.5 m whose direction differs by at most 40 degrees: 'is the lane drawn there, in that direction', no pairing and no tolerance at the ends. The junction PRECISION is the share of the
delivered points within 8 m of a GT connector run that lie within 1.5 m of a GT lane point.
Variants of the delivered graph (all on B's graph + the hard close-node step of the E275 spec): `plain` (no connectors), `p<t>` (written `p0.5`, shell-safe; `p0.5s`: with snapped connector ends, E311; `p>=0.5` is accepted when quoted: the connectors of the run's pair logits at probability t, with the run's handle scales when its dump has them, `--chord`: always the chord tangent), `oracle` (a perfect pair classifier on the
run's keypoints (every `valid` one, like the `p<t>` variants): logit +20 for a candidate whose two keypoints are matched, strict one-to-one, to the two ends of a GT pair, -20 for every other; chord-tangent curves), `oracle+fit` (the same with the handle scales fitted to the GT connector, `qjunction.fit_pair_scales`; a pair whose fit is NaN, a GT chord under 1 m, keeps the chord curve). An oracle is a
sanity check of where the loss is, not a bound of the system. With `--rows` the json also carries one row per GT maneuver (ends found by a keypoint with p >= 0.5 within 4 m / 30 degrees, length, turn, range, realised per variant) for the decompositions of the E310 card. The maneuver denominator (826 frames, 10991 maneuvers on VAL) is smaller than the 11083 GT pairs of the pair numbers: the box and the 3-point rule drop 0.8 %. `--max N`: the first N frames only (a smoke test of the pipeline).
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "tools/viz", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))

BOX, TOL, ANG, NEED, JUNC = 50.0, 1.5, 40.0, 0.9, 8.0
COS = np.cos(np.radians(ANG))


def _dense(p, step):
    """a polyline resampled so that no two neighbouring points are farther apart than `step` (the vertices stay)"""
    p = np.asarray(p, float).reshape(-1, 2)
    if len(p) < 2:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    n = np.maximum(np.ceil(seg / step - 1e-9).astype(int), 1)
    return np.concatenate([p[:1]] + [p[i] + (p[i + 1] - p[i]) * (np.arange(1, n[i] + 1) / n[i])[:, None] for i in range(len(seg))])


def _tangents(p):
    """unit tangents of a polyline (central differences, the ends one-sided)"""
    p = np.asarray(p, float)
    t = np.zeros_like(p)
    if len(p) < 2:
        t[:] = [0.0, 1.0]
        return t
    t[1:-1] = p[2:] - p[:-2]
    t[0], t[-1] = p[1] - p[0], p[-1] - p[-2]
    n = np.linalg.norm(t, axis=1, keepdims=True)
    return t / np.maximum(n, 1e-9)


def maneuvers(g, cluster):
    """[(entry cluster, exit cluster, run points every 1 m inside the box (n, 2), tangents (n, 2))] of the connector runs of a lane GT graph, the longest run per (entry, exit); `cluster` = `qends.cluster`"""
    runs = []
    for pc in g.get("pieces", []):
        q = np.asarray(pc["pts"], float).reshape(-1, 2)
        if len(q) < 2:
            continue
        tan = _tangents(q)
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
                    runs.append((q[k], tan[k], q[j], tan[j], q[k: j + 1]))
                k = j + 1
            else:
                k += 1
    if not runs:
        return []
    _, eid = cluster([(r[0], r[1]) for r in runs])
    _, xid = cluster([(r[2], r[3]) for r in runs])
    best = {}
    for r, a, b in zip(runs, eid, xid):
        if (a, b) not in best or len(r[4]) > len(best[(a, b)][4]):
            best[(a, b)] = r
    out = []
    for (a, b), r in best.items():
        P = _dense(r[4], 1.0)
        m = (np.abs(P) < BOX).all(1)
        if m.sum() >= 3:
            out.append((a, b, P[m], _tangents(P)[m]))
    return out


def delivered(raw):
    """the densified points (0.5 m) of a delivered graph (dict with `pts`) and their tangents"""
    Ps, Ts = [], []
    for p in raw["pts"]:
        if len(p) < 2:
            continue
        d = _dense(p, 0.5)
        Ps.append(d)
        Ts.append(_tangents(d))
    if not Ps:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(Ps), np.concatenate(Ts)


def realised_share(tree, tang, run_pts, run_tan):
    """share of the GT run points that have a delivered point within TOL (ALL of them, not the nearest few: a crowded graph must not hide the right line) whose direction differs by at most ANG"""
    if tree is None:
        return 0.0
    ok = np.zeros(len(run_pts), bool)
    for i, nb in enumerate(tree.query_ball_point(run_pts, TOL)):
        if nb:
            ok[i] = bool(((tang[nb] * run_tan[i]).sum(1) >= COS).any())
    return float(ok.mean())


def frame_counts(raw, mans, ref_tree):
    """(maneuvers realised, maneuvers, delivered points near the junction that are on a GT lane, delivered points near the junction, realised flags) of one delivered graph"""
    P, T = delivered(raw)
    tree = cKDTree(P) if len(P) else None
    real = [realised_share(tree, T, m[2], m[3]) >= NEED for m in mans]
    near = P[(np.abs(P) < BOX).all(1)] if len(P) else P
    if len(near):
        near = near[cKDTree(np.concatenate([m[2] for m in mans])).query(near)[0] <= JUNC]
    hit = int((ref_tree.query(near)[0] <= TOL).sum()) if len(near) else 0
    return int(sum(real)), len(mans), hit, len(near), real


def score(a):
    import jgt as J
    import jrule_control as JR
    import qends as QE
    import qends_eval as QEV
    import qgraph_eval as QGE
    import qgraph_readout as QR
    import qgraph_view as QV
    import qjunction as QJ
    names = a.variants
    names = [("p" + v[3:]) if v.startswith("p>=") else v for v in names]                      # p>=0.5 (quoted) = p0.5
    snapped = {v for v in names if v.startswith("p") and v.endswith("s") and v[1:-1].replace(".", "", 1).isdigit()}                    # p0.1s = connectors at p >= 0.1 with snapped ends (E311)
    thr = {v: float(v[1:-1] if v in snapped else v[1:]) for v in names if v in snapped or (v.startswith("p") and v[1:].replace(".", "", 1).isdigit())}
    unknown = [v for v in names if v not in ("plain", "oracle", "oracle+fit") and v not in thr]
    assert not unknown, f"unknown variants {unknown} (plain, p<t> such as p0.5, oracle, oracle+fit)"
    frames = QGE.val_frames()
    dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{a.seed}.pkl"), "rb"))
    ends = {f["row"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", a.run, "ends_val.pkl"), "rb"))}
    assert [d["row"] for d in dump] == [f.row for f in frames], "the full dump of B is not in the order of the VAL frames"
    cfg = json.load(open(os.path.join(ROOT, "runs", a.run, "config.json")))
    assert cfg.get("seed", a.seed) == a.seed, f"--seed {a.seed} does not match the run (seed {cfg.get('seed')}): its keypoints would sit on another seed's graph"
    spec = QR.parse("close:1.5@0.5")
    per = {v: [] for v in names}
    rows = []
    t0 = time.time()
    for b, (f, d) in enumerate(zip(frames, dump)):
        if a.max and b >= a.max:
            break
        g = QV.load_gt(f.token)
        mans = maneuvers(g, QE.cluster)
        if not mans:
            continue
        jg = J.junction_gt(g)
        e = ends[f.row]
        pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                             links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
        graph, bk = QGE.readout(pred, 0.5, 0.5)
        long_mask = np.asarray(pred["links"]["long"], bool)
        K = e["logit"].shape[0]
        ve = e["valid"][0] & (e["kp_p"][:K] >= 0.5)
        vx = e["valid"][1] & (e["kp_p"][K:] >= 0.5)
        logit_or = np.full_like(e["logit"], -20.0)
        scales_or = {}
        if jg is not None and any(v.startswith("oracle") for v in names):
            ma, mb = JR._match(e["kp_pos"][:K], e["kp_dir"][:K], e["valid"][0], jg["ent"]), JR._match(e["kp_pos"][K:], e["kp_dir"][K:], e["valid"][1], jg["ext"])
            sc_all = QJ.fit_pair_scales(jg)
            for ip, pr in enumerate(jg["pairs"]):
                i, j = int(ma[int(pr[0])]), int(mb[int(pr[1])])
                if i >= 0 and j >= 0 and e["allowed"][i, j]:
                    logit_or[i, j] = 20.0
                    if np.isfinite(sc_all[ip]).all():
                        scales_or[(i, j)] = tuple(float(x) for x in sc_all[ip])
        ref_tree = f.ref().tree
        base = len(rows)
        for v in names:
            if v == "plain":
                raw = QGE.to_raw(graph)
            else:
                lg, th = (logit_or, 0.5) if v.startswith("oracle") else (e["logit"], thr[v])
                p = 1.0 / (1.0 + np.exp(-lg)) * e["allowed"]
                ia, ib = np.nonzero(p >= th)
                prs = [(e["kp_pos"][i], e["kp_dir"][i], e["kp_pos"][K + j], e["kp_dir"][K + j]) + (scales_or.get((int(i), int(j)), ()) if v == "oracle+fit" else (tuple(e["scale"][i, j]) if (e.get("scale") is not None and not a.chord and v != "oracle") else ()))
                       for i, j in zip(ia, ib)]
                raw, _, _ = QEV.add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], long_mask, prs, snap=v in snapped)
            n_ok, n, hit, n_near, real = frame_counts(raw, mans, ref_tree)
            per[v].append((n_ok, n, hit, n_near))
            if a.rows:
                if v == names[0]:
                    kp, kd = e["kp_pos"], e["kp_dir"]
                    for m in mans:
                        ang = float(np.degrees(np.arctan2(m[3][0][0] * m[3][-1][1] - m[3][0][1] * m[3][-1][0], (m[3][0] * m[3][-1]).sum())))
                        rows.append(dict(token=f.token, length=float(np.linalg.norm(m[2][-1] - m[2][0])), turn=ang, range=float(np.linalg.norm(m[2][0])),
                                         found_e=_near(kp[:K], kd[:K], ve, m[2][0], m[3][0]), found_x=_near(kp[K:], kd[K:], vx, m[2][-1], m[3][-1]), real={}))
                for i_, r_ in enumerate(real):
                    rows[base + i_]["real"][v] = bool(r_)
        if (b + 1) % 100 == 0:
            print(f"  {b + 1}/{len(frames)}  {time.time() - t0:.0f} s", flush=True)
    res = {}
    for v in names:
        x = np.array(per[v])
        res[v] = dict(maneuvers=int(x[:, 1].sum()), realised=int(x[:, 0].sum()), recall=float(x[:, 0].sum() / x[:, 1].sum()), frame_recall_mean=float(np.mean(x[:, 0] / x[:, 1])),
                      frames_none=int((x[:, 0] == 0).sum()), frames_all=int((x[:, 0] == x[:, 1]).sum()), junction_precision=float(x[:, 2].sum() / max(1, x[:, 3].sum())), frames=len(x))
    out = dict(run=a.run, seed=a.seed, tol=TOL, angle=ANG, need=NEED, variants=res)
    if a.rows:
        out["rows"] = rows
    json.dump(out, open(os.path.join(ROOT, "runs", a.run, "maneuver_val.json"), "w"), indent=1)
    for v in names:
        r = res[v]
        print(f"{v:12s} maneuvers realised {r['realised']}/{r['maneuvers']} = {r['recall']:.3f}; per-frame mean {r['frame_recall_mean']:.3f}; frames with none {r['frames_none']}, with all {r['frames_all']} of {r['frames']}; junction precision {r['junction_precision']:.3f}")


def _near(kpos, kdir, valid, pt, tn):
    """a keypoint of the kind within 4 m and 30 degrees of a GT end (not one-to-one: 'is an end found there')"""
    if not valid.any():
        return False
    d = np.linalg.norm(kpos[valid] - pt, axis=1)
    return bool(((d <= 4.0) & ((kdir[valid] * tn).sum(1) >= np.cos(np.radians(30)))).any())


def selftest():
    # a GT run: a quarter circle of radius 15 m (a left turn) from (0, -15) heading +y to (-15, 0) heading -x
    th = np.linspace(0, np.pi / 2, 60)
    arc = np.stack([-(15 - 15 * np.cos(th)), -15 + 15 * np.sin(th)], 1)
    g = dict(pieces=[dict(pts=arc, segments=[dict(kind="connector", i0=0, i1=len(arc) - 1)])])
    import qends as QE
    mans = maneuvers(g, QE.cluster)
    assert len(mans) == 1 and len(mans[0][2]) >= 20, mans
    ref_tree = cKDTree(arc)
    ok = dict(pts=[arc])
    c = np.array([-15.0, -15.0])                                                       # the centre of the circle: radial offsets keep the direction and move the lane sideways everywhere
    shifted = dict(pts=[c + (arc - c) * (19.0 / 15.0)])                               # 4 m aside: not realised
    reverse = dict(pts=[arc[::-1]])                                                    # the same lane the other way round: the direction rule rejects it
    half = dict(pts=[arc[: len(arc) // 2]])                                            # half of it: 50 % < 90 %
    near = dict(pts=[c + (arc - c) * (15.6 / 15.0)])                                   # 0.6 m aside: realised
    for raw, want in ((ok, 1), (shifted, 0), (reverse, 0), (half, 0), (near, 1), (dict(pts=[]), 0)):
        n_ok, n, hit, n_near, _ = frame_counts(raw, mans, ref_tree)
        assert n == 1 and n_ok == want, (n_ok, want)
    # precision: the delivered points near the junction that lie on a GT lane (the shifted copy lies 4 m away: none)
    assert frame_counts(ok, mans, ref_tree)[2] == frame_counts(ok, mans, ref_tree)[3] > 0
    assert frame_counts(shifted, mans, ref_tree)[2] == 0
    # a frame without a connector run has no maneuver
    assert maneuvers(dict(pieces=[dict(pts=arc, segments=[dict(kind="lane", i0=0, i1=len(arc) - 1)])]), QE.cluster) == []
    # the thresholds are pinned: a straight GT run along x (101 points, 0.5 m apart), a delivered line parallel to it
    x = np.arange(-25.0, 25.01, 0.5)
    run = np.stack([x, np.zeros_like(x)], 1)
    gs = dict(pieces=[dict(pts=run, segments=[dict(kind="connector", i0=0, i1=len(run) - 1)])])
    ms = maneuvers(gs, QE.cluster)
    assert len(ms) == 1 and len(ms[0][2]) == 101
    tr = cKDTree(run)

    def line(y, x0=-25.0, x1=25.0, rev=False):
        q = np.stack([np.arange(x0, x1 + 1e-9, 0.5), np.full(int((x1 - x0) / 0.5) + 1, float(y))], 1)
        return q[::-1] if rev else q

    for y, want in ((1.4, 1), (1.6, 0)):                                                  # TOL = 1.5 m
        assert frame_counts(dict(pts=[line(y)]), ms, tr)[0] == want, y
    for x1, want in ((19.0, 1), (17.0, 0)):                                               # NEED = 0.9: 92 % of the run covered, 87 % covered
        assert frame_counts(dict(pts=[line(0.0, -25.0, x1)]), ms, tr)[0] == want, x1
    t = _tangents(run)
    rot = lambda deg: np.stack([np.cos(np.radians(deg)) * t[:, 0] - np.sin(np.radians(deg)) * t[:, 1], np.sin(np.radians(deg)) * t[:, 0] + np.cos(np.radians(deg)) * t[:, 1]], 1)
    for deg, want in ((35, 1.0), (45, 0.0)):                                              # ANG = 40 degrees
        assert realised_share(tr, rot(deg), run, t) == want, deg
    # a crowded graph does not hide the right line: the correct line 1.2 m aside plus eight reversed duplicates and a crossing line
    crowd = [line(0.05 * k, rev=True) for k in range(1, 9)] + [np.stack([np.zeros(41), np.linspace(-10, 10, 41)], 1)] + [line(1.2)]          # the right line comes LAST in the point order
    assert frame_counts(dict(pts=crowd), ms, tr)[0] == 1
    # the junction precision looks only at delivered points within 8 m of a GT run: a far line is not counted
    assert frame_counts(dict(pts=[line(30.0)]), ms, tr)[3] == 0 and frame_counts(dict(pts=[line(5.0)]), ms, tr)[3] > 0
    # two runs of one (entry, exit) pair: the longer one is the maneuver, a second pair is a second maneuver
    short, long_ = run[40:61], run
    g2 = dict(pieces=[dict(pts=short, segments=[dict(kind="connector", i0=0, i1=len(short) - 1)]), dict(pts=long_, segments=[dict(kind="connector", i0=0, i1=len(long_) - 1)])])
    m2 = maneuvers(g2, QE.cluster)
    assert len(m2) <= 2 and max(len(m[2]) for m in m2) == 101, [len(m[2]) for m in m2]
    print("jmaneuver selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--variants", nargs="+", default=["plain", "p0.5", "p0.1", "oracle", "oracle+fit"])
    s.add_argument("--rows", action="store_true")
    s.add_argument("--chord", action="store_true", help="the chord tangent for the p<t> variants even when the dump has learned handle scales (the control)")
    s.add_argument("--max", type=int, default=0, help="only the first N frames (a smoke test of the pipeline)")
    sub.add_parser("selftest")
    a = ap.parse_args()
    selftest() if a.cmd == "selftest" else score(a)
