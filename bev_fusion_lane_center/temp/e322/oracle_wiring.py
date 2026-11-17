"""E322 scratch, gate G0: the headroom of the wiring. The snapped read-out of a run (connectors p >= thr, every end wired to ONE delivered piece end and drawn from it) with the piece chosen
  c        : by the rule (nearest piece end within 4 m / 45 degrees of the KEYPOINT end)        = the card of record of E312 (sanity: must reproduce ends_card_snap_thr0.1.json)
  o        : by the rule applied at the matched GT END (oracle selection; an end that matches no GT end or finds no piece at the GT end keeps the keypoint rule)
  o_drop   : as o, and a pair with an end that matches no GT end is not delivered (an oracle for the false connectors; separates the selection from the false-pair filter)
usage: oracle_wiring.py <run> <seed> [--thr 0.1] [--attach]  -> temp/e322/oracle_<run>.json. CPU."""
import argparse, json, os, pickle, sys, time
import numpy as np
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import qgraph_eval as QE
import qgraph_readout as QR
from qends_eval import add_connectors, make_pairs, _tan, _ang, WIRE_POS, WIRE_ANG, KEYS
ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("seed", type=int); ap.add_argument("--thr", type=float, default=0.1); ap.add_argument("--attach", action="store_true"); ap.add_argument("--max", type=int, default=0)
a = ap.parse_args()
MATCH_POS, MATCH_ANG = 4.0, 30.0
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{a.seed}.pkl"), "rb"))
ends = {f["row"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", a.run, "ends_val.pkl"), "rb"))}
gt = pickle.load(open(os.path.join(ROOT, "temp", "e322", "gt_ends_val.pkl"), "rb"))
frames = QE.val_frames()
spec = QR.parse("close:1.5@0.5")


def match(kp, kd, valid, G, m):
    """numpy copy of qendnet.match_keypoints for one frame and one kind: the index of the GT end each keypoint is (within 4 m and 30 degrees, nearest, one-to-one), -1 = none"""
    res = np.full(len(kp), -1, int)
    if not m.any() or not len(kp):
        return res
    G = G[m]
    dk = np.linalg.norm(kp[:, None, :] - G[None, :, :2], axis=-1)
    ang = (kd[:, None, :] * G[None, :, 2:]).sum(-1)
    ok = (dk <= MATCH_POS) & (ang >= np.cos(np.radians(MATCH_ANG))) & valid[:, None]
    dk = np.where(ok, dk, 1e6)
    aa = dk.argmin(1)
    best = dk.argmin(0)
    mine = best[aa] == np.arange(len(kp))
    idx = np.where(m)[0]
    res = np.where(ok.any(1) & mine, idx[aa], -1)
    return res


rec = {v: [] for v in ("c", "o", "o_drop", "r2", "r3", "r6")}
stat = dict(ends=0, matched=0, found=0, both=0, same=0)
t0 = time.time()
for b, (f, d) in enumerate(zip(frames, dump)):
    if a.max and b >= a.max:
        break
    pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                         links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
    graph, bk = QE.readout(pred, 0.5, 0.5)
    L = [np.asarray(P, float) for P in graph.lines if len(P)]
    pe = [(P[-1], _tan(P, True), P[0], _tan(P, False)) for P in L]
    e = ends[f.row]
    g = gt[f.row]
    k = e["kp_pos"].shape[0] // 2
    p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
    ia, ib = np.nonzero(p >= a.thr)
    pairs = make_pairs(e, ia, ib, k, attach=a.attach)
    qp, qd = (e["att_pos"], e["att_dir"]) if a.attach else (e["kp_pos"], e["kp_dir"])
    valid = np.asarray(e["valid"], bool)
    m_in = match(e["kp_pos"][:k], e["kp_dir"][:k], valid[0], g["ent"], g["ent_m"])
    m_out = match(e["kp_pos"][k:], e["kp_dir"][k:], valid[1], g["ext"], g["ext_m"])

    def oracle_end(G, gi, entry):
        """the delivered piece end the rule picks at the GT end (entry: the last point of a piece, exit: the first point); None when there is none in reach"""
        pos, tan = G[gi, :2], G[gi, 2:]
        best, bd = None, np.inf
        for en in pe:
            q, t = (en[0], en[1]) if entry else (en[2], en[3])
            dd = np.linalg.norm(q - pos)
            if dd <= WIRE_POS and _ang(t, tan) <= WIRE_ANG and dd < bd:
                best, bd = (q, t), dd
        return best
    o_pairs, od_pairs = [], []
    for n, (i, j) in enumerate(zip(ia, ib)):
        pr = list(pairs[n])
        new = list(pr)
        for slot, kidx, mt, G, entry in ((0, i, m_in, g["ent"], True), (2, j, m_out, g["ext"], False)):
            stat["ends"] += 1
            gi = mt[kidx]
            if gi >= 0:
                stat["matched"] += 1
                oe = oracle_end(G, gi, entry)
                if oe is not None:
                    stat["found"] += 1
                    ke = oracle_end(np.concatenate([qp[kidx + (0 if entry else k)], qd[kidx + (0 if entry else k)]])[None, :], 0, entry)      # the rule at the end the pair uses (keypoint, or the attached end)
                    if ke is not None:
                        stat["both"] += 1
                        stat["same"] += int(np.linalg.norm(ke[0] - oe[0]) < 1e-9)
                    new[slot], new[slot + 1] = oe[0], oe[1]
        o_pairs.append(tuple(new))
        if m_in[i] >= 0 and m_out[j] >= 0:
            od_pairs.append(tuple(new))
    for name, pp, rr in (("c", pairs, None), ("o", o_pairs, None), ("o_drop", od_pairs, None), ("r2", pairs, 2.0), ("r3", pairs, 3.0), ("r6", pairs, 6.0)):
        raw, n_c, n_w = add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), pp, snap=True, wire_pos=rr)
        rec[name].append(QE.graph_counts(raw, f))
    f._lane = f._ref = None
    if (b + 1) % 200 == 0:
        print(f"  {b + 1}/{len(frames)}  {time.time() - t0:.0f} s", flush=True)
num = {v: QE.graph_numbers(rec[v]) for v in rec}
out = dict(run=a.run, seed=a.seed, thr=a.thr, attach=a.attach, stat=stat, numbers={v: {kk: float(num[v][kk]) for kk in KEYS if kk in num[v]} for v in num})
json.dump(out, open(os.path.join(ROOT, "temp", "e322", f"oracle_{a.run}{'_attach' if a.attach else ''}.json"), "w"), indent=1)
print(f"{'':8s}" + "".join(f"{kk:>12s}" for kk in KEYS))
for v in num:
    print(f"{v:8s}" + "".join(f"{num[v][kk]:12.4f}" for kk in KEYS))
print(stat)
