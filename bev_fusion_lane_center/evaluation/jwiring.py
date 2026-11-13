"""Wiring of the connectors of the junction pair head to the delivered pieces (E322, gate G0): how much is there to gain from CHOOSING the delivered piece a connector end is joined to?

    wsl bash tools/env/_py14.sh jwiring.py gt-ends                                   the GT ends of the 873 VAL frames (position + heading, mask, true pairs) -> temp/jwiring/gt_ends_val.pkl (CPU, the loader of the dumps; needs the VAL lane GT store)
    wsl bash tools/env/_py14.sh jwiring.py score --run e312_p_s0 --seed 0 [--thr 0.1] [--attach] [--max N]
                                                                                       -> runs/<run>/wiring_g0[_attach].json + a printed table (CPU, VAL; reads runs/<run>/ends_val.pkl, B's full dump temp/e276/predfull_e273_p_s<seed>.pkl and the GT ends above)
    wsl bash tools/env/_py14.sh jwiring.py reach --run e312_p_s0 --seed 0 [--attach]  how many delivered piece ends lie in reach (4 m / 45 degrees) of the connector ends: how often the one-piece rule has a choice
    wsl bash tools/env/_py14.sh jwiring.py selftest

The snapped read-out of a run (connectors p >= thr, each end wired to ONE delivered piece end and drawn from it: `qends_eval.add_connectors(snap=True)`) with the piece chosen
  c       by the rule: the nearest piece end within 4 m / 45 degrees of the keypoint end (the card of record of E312, `ends_card_snap_thr0.1.json`; the sanity check: it must reproduce that card)
  o       by the rule applied at the matched GT END (an oracle selection: the piece the rule would pick if the keypoint were exactly on the GT end; an end that matches no GT end or finds no piece at its GT end keeps the keypoint rule)
  o_drop  as o, and a pair with an end that matches no GT end is not delivered (an oracle for the false connectors; separates the choice of the piece from the filter for false pairs)
  r2 r3 r6  the rule with the reach 2 / 3 / 6 m instead of 4 m (is a learned gate worth anything: the reach is the one parameter of the rule)
The headroom of a learned selection is o - c (the oracle cannot be beaten by a head that is trained against it); `stat` counts the ends, the ends that match a GT end, those that find a piece at the GT end, and how often the rule at the keypoint
picks the same piece end as the rule at the GT end. VAL only, TEST never. The matching of keypoints to GT ends is the one of `qendnet.match_keypoints` (strict one-to-one, 4 m, 30 degrees), copied here in numpy.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
from qends_eval import WIRE_ANG, WIRE_POS, KEYS, _ang, _tan, add_connectors, make_pairs  # noqa: E402

MATCH_POS, MATCH_ANG = 4.0, 30.0
GT_PATH = os.path.join(ROOT, "temp", "jwiring", "gt_ends_val.pkl")
VARIANTS = (("c", None), ("o", None), ("o_drop", None), ("r2", 2.0), ("r3", 3.0), ("r6", 6.0))


def match(kp, kd, valid, G, m):
    """numpy copy of qendnet.match_keypoints for one frame and one kind: kp (n, 2), kd (n, 2), valid (n,), GT ends G (g, 4: x, y, tx, ty) with mask m (g,) -> (n,) index of the GT end the keypoint is
    (within 4 m and 30 degrees, the nearest, one-to-one: a duplicate keypoint of the same end is no positive), -1 = none"""
    res = np.full(len(kp), -1, int)
    if not m.any() or not len(kp):
        return res
    idx = np.where(m)[0]
    G = G[idx]
    dk = np.linalg.norm(kp[:, None, :] - G[None, :, :2], axis=-1)
    ang = (kd[:, None, :] * G[None, :, 2:]).sum(-1)
    ok = (dk <= MATCH_POS) & (ang >= np.cos(np.radians(MATCH_ANG))) & valid[:, None]
    dk = np.where(ok, dk, 1e6)
    a = dk.argmin(1)
    best = dk.argmin(0)
    mine = best[a] == np.arange(len(kp))
    return np.where(ok.any(1) & mine, idx[a], -1)


def piece_ends(lines):
    """[(last point, its tangent, first point, its tangent)] of the delivered lines"""
    L = [np.asarray(P, float) for P in lines if len(P)]
    return [(P[-1], _tan(P, True), P[0], _tan(P, False)) for P in L]


def rule_end(pe, pos, tan, entry, reach=WIRE_POS):
    """the delivered piece end the nearest-piece rule picks for an end at pos with heading tan (entry: the last point of a piece, exit: the first point; within `reach` m and 45 degrees), (point, tangent) or None"""
    best, bd = None, np.inf
    for en in pe:
        q, t = (en[0], en[1]) if entry else (en[2], en[3])
        d = np.linalg.norm(q - pos)
        if d <= reach and _ang(t, tan) <= WIRE_ANG and d < bd:
            best, bd = (q, t), d
    return best


def oracle_pairs(pairs, ia, ib, pe, m_in, m_out, gt_ent, gt_ext, qp, qd, k, stat=None):
    """pairs of `make_pairs` with every end that matches a GT end and finds a piece at that GT end replaced by that piece end (position, tangent): add_connectors(snap=True) then wires exactly that piece.
    Returns (o_pairs, o_drop_pairs); stat (dict) is incremented: ends, matched, found, both (the rule at the keypoint also finds a piece), same (and it is the same piece end)"""
    o_pairs, od_pairs = [], []
    for n, (i, j) in enumerate(zip(ia, ib)):
        new = list(pairs[n])
        for slot, kidx, mt, G, entry in ((0, i, m_in, gt_ent, True), (2, j, m_out, gt_ext, False)):
            gi = mt[kidx]
            if stat is not None:
                stat["ends"] += 1
                stat["matched"] += int(gi >= 0)
            if gi < 0:
                continue
            oe = rule_end(pe, G[gi, :2], G[gi, 2:], entry)
            if oe is None:
                continue
            if stat is not None:
                stat["found"] += 1
                ke = rule_end(pe, qp[kidx + (0 if entry else k)], qd[kidx + (0 if entry else k)], entry)
                if ke is not None:
                    stat["both"] += 1
                    stat["same"] += int(np.linalg.norm(ke[0] - oe[0]) < 1e-9)
            new[slot], new[slot + 1] = oe[0], oe[1]
        o_pairs.append(tuple(new))
        if m_in[i] >= 0 and m_out[j] >= 0:
            od_pairs.append(tuple(new))
    return o_pairs, od_pairs


def gt_ends(a=None):
    import lane_gt_data as LD
    LD.QData.DEFAULT_JUNCTION = True
    ds, dl = LD.loader("val_clean", bs=8, workers=0, shuffle=False, map_mode="error", pixels=False)
    assert ds.missing == 0, f"{ds.missing} VAL frames have no stored lane GT"
    out = {}
    for b in dl:
        for i in range(len(b["token"])):
            out[int(b["row"][i])] = dict(ent=b["ent"][i].numpy(), ent_m=b["ent_m"][i].numpy(), ext=b["ext"][i].numpy(), ext_m=b["ext_m"][i].numpy(), pairs=b["pairs"][i].numpy())
    os.makedirs(os.path.dirname(GT_PATH), exist_ok=True)
    pickle.dump(out, open(GT_PATH, "wb"))
    print(f"{len(out)} frames -> {os.path.relpath(GT_PATH, ROOT)}")


def _load(a):
    import qgraph_eval as QE
    import qgraph_readout as QR
    dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{a.seed}.pkl"), "rb"))
    ends = {f["row"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", a.run, "ends_val.pkl"), "rb"))}
    frames = QE.val_frames()
    spec = QR.parse("close:1.5@0.5")
    return QE, QR, dump, ends, frames, spec


def _graph(QE, QR, d, spec):
    pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                         links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
    graph, bk = QE.readout(pred, 0.5, 0.5)
    return pred, graph, bk


def score(a):
    QE, QR, dump, ends, frames, spec = _load(a)
    gt = pickle.load(open(GT_PATH, "rb"))
    rec = {v: [] for v, _ in VARIANTS}
    stat = dict(ends=0, matched=0, found=0, both=0, same=0)
    t0 = time.time()
    for b, (f, d) in enumerate(zip(frames, dump)):
        if a.max and b >= a.max:
            break
        pred, graph, bk = _graph(QE, QR, d, spec)
        pe = piece_ends(graph.lines)
        e, g = ends[f.row], gt[f.row]
        k = e["kp_pos"].shape[0] // 2
        p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
        ia, ib = np.nonzero(p >= a.thr)
        pairs = make_pairs(e, ia, ib, k, attach=a.attach)
        qp, qd = (e["att_pos"], e["att_dir"]) if a.attach else (e["kp_pos"], e["kp_dir"])
        valid = np.asarray(e["valid"], bool)
        m_in = match(e["kp_pos"][:k], e["kp_dir"][:k], valid[0], g["ent"], g["ent_m"])
        m_out = match(e["kp_pos"][k:], e["kp_dir"][k:], valid[1], g["ext"], g["ext_m"])
        o_pairs, od_pairs = oracle_pairs(pairs, ia, ib, pe, m_in, m_out, g["ent"], g["ext"], qp, qd, k, stat)
        use = {"c": pairs, "o": o_pairs, "o_drop": od_pairs, "r2": pairs, "r3": pairs, "r6": pairs}
        for name, reach in VARIANTS:
            raw, _, _ = add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), use[name], snap=True, wire_pos=reach)
            rec[name].append(QE.graph_counts(raw, f))
        f._lane = f._ref = None
        if (b + 1) % 200 == 0:
            print(f"  {b + 1}/{len(frames)}  {time.time() - t0:.0f} s", flush=True)
    num = {v: QE.graph_numbers(rec[v]) for v, _ in VARIANTS}
    out = dict(run=a.run, seed=a.seed, thr=a.thr, attach=a.attach, frames=min(len(frames), a.max or len(frames)), stat=stat, numbers={v: {kk: float(num[v][kk]) for kk in KEYS if kk in num[v]} for v in num})
    name = f"wiring_g0{'_attach' if a.attach else ''}{'_max' + str(a.max) if a.max else ''}.json"
    json.dump(out, open(os.path.join(ROOT, "runs", a.run, name), "w"), indent=1)
    if not a.max:                                                                                                       # the per-frame counts of every variant: `jdiag.py boot <run> wiring_g0_o <run> wiring_g0_c` (paired scene bootstrap)
        for v, _ in VARIANTS:
            pickle.dump(dict(scenes=[f.scene for f in frames], base=rec["c"], head=rec[v]), open(os.path.join(ROOT, "runs", a.run, f"wiring_g0{'_attach' if a.attach else ''}_{v}.counts.pkl"), "wb"))
    print(f"{'':8s}" + "".join(f"{kk:>12s}" for kk in KEYS))
    for v, _ in VARIANTS:
        print(f"{v:8s}" + "".join(f"{num[v][kk]:12.4f}" for kk in KEYS))
    print(stat)


def reach(a):
    QE, QR, dump, ends, frames, spec = _load(a)
    hist = {"in": np.zeros(6, int), "out": np.zeros(6, int)}
    gaps = []
    for f, d in zip(frames, dump):
        pred, graph, bk = _graph(QE, QR, d, spec)
        pe = piece_ends(graph.lines)
        e = ends[f.row]
        k = e["kp_pos"].shape[0] // 2
        p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
        ia, ib = np.nonzero(p >= a.thr)
        qp, qd = (e["att_pos"], e["att_dir"]) if a.attach else (e["kp_pos"], e["kp_dir"])
        for i, j in zip(ia, ib):
            for key, idx, pos_i, tan_i in (("in", i, 0, 1), ("out", k + j, 2, 3)):
                ds = sorted(float(np.linalg.norm(en[pos_i] - qp[idx])) for en in pe if np.linalg.norm(en[pos_i] - qp[idx]) <= WIRE_POS and _ang(en[tan_i], qd[idx]) <= WIRE_ANG)
                hist[key][min(len(ds), 5)] += 1
                if len(ds) >= 2:
                    gaps.append(ds[1] - ds[0])
    n = {kk: hist[kk].sum() for kk in hist}
    print(a.run, "attached ends" if a.attach else "keypoint ends", "pairs", n["in"])
    for kk in hist:
        print(f"  {kk:3s} candidates in reach 0 / 1 / 2 / 3 / 4 / 5+: " + " / ".join(f"{100 * h / n[kk]:.1f} %" for h in hist[kk]))
    g = np.asarray(gaps)
    print(f"  ends with >= 2 candidates: {len(g)}; second minus nearest distance: median {np.median(g):.2f} m, share under 0.5 m {100 * np.mean(g < 0.5):.0f} %, under 1 m {100 * np.mean(g < 1.0):.0f} %")


def selftest():
    up, up2 = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    # two parallel lanes that end at the junction: piece 0 at x = 0, piece 1 at x = 2.5, both heading +y; two pieces that start after it (2, 3) at x = 8 and 10.5
    lines = [np.c_[np.zeros(10), np.linspace(-20, -2, 10)], np.c_[np.zeros(10) + 2.5, np.linspace(-20, -2, 10)], np.c_[np.zeros(10) + 8, np.linspace(2, 20, 10)], np.c_[np.zeros(10) + 10.5, np.linspace(2, 20, 10)]]
    pe = piece_ends(lines)
    # the rule at a keypoint 1.0 m right of piece 0 picks piece 0 (1.0 m) not piece 1 (1.5 m); the GT end 0.1 m from piece 1 makes the oracle pick piece 1
    kp = np.array([1.0, -2.0])
    r = rule_end(pe, kp, up, True)
    assert r is not None and np.allclose(r[0], [0.0, -2.0]), r
    o = rule_end(pe, np.array([2.4, -2.0]), up, True)
    assert o is not None and np.allclose(o[0], [2.5, -2.0]), o
    assert rule_end(pe, np.array([30.0, 0.0]), up, True) is None and rule_end(pe, kp, np.array([1.0, 0.0]), True) is None, "reach and the 45 degree gate"
    assert rule_end(pe, kp, up, True, reach=0.5) is None and rule_end(pe, kp, up, True, reach=1.2) is not None
    # matching: one keypoint per GT end, the nearer duplicate wins, a keypoint with the wrong heading or too far is no match
    G = np.array([[2.4, -2.0, 0.0, 1.0], [30.0, 0.0, 0.0, 1.0]])
    kps = np.array([[1.0, -2.0], [2.0, -2.0], [2.4, -2.0]])
    kds = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    mt = match(kps, kds, np.array([True, True, True]), G, np.array([True, True]))
    assert mt.tolist() == [-1, 0, -1], mt                                                              # keypoint 1 is the nearer one of the GT end 0; keypoint 0 is a duplicate; keypoint 2 runs the wrong way
    # the oracle pair: the entry end of the keypoint is replaced by piece 1's end, the exit keypoint matches nothing and stays
    pairs = [(kp, up, np.array([8.0, 2.0]), up2)]
    m_in, m_out = np.array([0]), np.array([-1])
    stat = dict(ends=0, matched=0, found=0, both=0, same=0)
    op, od = oracle_pairs(pairs, np.array([0]), np.array([0]), pe, m_in, m_out, np.array([[2.4, -2.0, 0.0, 1.0]]), np.array([[0.0, 0.0, 0.0, 1.0]]), np.array([kp, np.array([8.0, 2.0])]), np.array([up, up2]), 1, stat)
    assert np.allclose(op[0][0], [2.5, -2.0]) and np.allclose(op[0][2], [8.0, 2.0]) and od == [], (op, od)
    assert stat == dict(ends=2, matched=1, found=1, both=1, same=0), stat
    # add_connectors(snap=True) wires exactly the chosen piece: the rule's pair joins piece 0, the oracle's pair joins piece 1
    kinds, links, long_mask = ["lane"] * 4, [(0,), (1,), (2,), (3,)], np.array([False] * 4)
    r_rule = add_connectors(lines, [], [[0], [1], [2], [3]], kinds, links, long_mask, pairs, snap=True)[0]
    r_orc = add_connectors(lines, [], [[0], [1], [2], [3]], kinds, links, long_mask, op, snap=True)[0]
    assert (0, 4) in r_rule["edges"] and (1, 4) not in r_rule["edges"] and (1, 4) in r_orc["edges"] and (0, 4) not in r_orc["edges"], (r_rule["edges"], r_orc["edges"])
    print("jwiring selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gt-ends")
    for c in ("score", "reach"):
        s = sub.add_parser(c)
        s.add_argument("--run", required=True)
        s.add_argument("--seed", type=int, required=True)
        s.add_argument("--thr", type=float, default=0.1)
        s.add_argument("--attach", action="store_true")
        s.add_argument("--max", type=int, default=0)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    else:
        {"gt-ends": gt_ends, "score": score, "reach": reach}[a.cmd](a)
