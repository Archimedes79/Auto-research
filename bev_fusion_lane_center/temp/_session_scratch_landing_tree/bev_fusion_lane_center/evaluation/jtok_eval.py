"""Controls, read-out and scoring of E357 (the GT nodes as tokens, one pair head for every edge; `training/grid/jtok.py`). The delivered graph of every arm = the GT nodes of the lattice + the arm's links.

    wsl bash tools/env/_py14.sh jtok_eval.py ctl --lattice 2.5 [--max N] [--drop 0.17 --drop-seed 0]      # CPU: the arms lo, lo_near, lo_all, hi -> runs/eval/e357_<arm>_<readout>/val.json (never overwritten)
    wsl bash tools/env/_py14.sh jtok_eval.py ctl --lattice 2.0 --check [--max N --ref CARD]              # the kill check: arm hi (GT-section read-out) against runs/eval/e346_l2/val.json to 1e-6 -> PASS / FAIL, writes nothing
    wsl bash tools/env/_py14.sh jtok_eval.py dump --run R --seed S [--drop 0.17 --drop-seed 0] [--cpu]  # the head on the 873 VAL frames -> runs/R/tok_val[_drop..].pkl (qformat dicts; links = allowed pairs with p >= 0.05)
    wsl bash tools/env/_py14.sh jtok_eval.py score --run R --seed S --thr 0.5 [--drop 0.17 --drop-seed 0] # CPU -> runs/R/tok_card_thr0.5[_drop..]_<readout>.json

Arms (controls on the same nodes, no learning; handle scales 1/3):
  lo        every node ONE link to the nearest node ahead: within 45 degrees of its heading, at most 8 m away, heading difference at most 60 degrees
  lo_near   lo + every node WITHOUT a successor (in lo) gets one LONG link (chord above 8 m) to the nearest node WITHOUT a predecessor (in lo) inside the allowed gate of `jtok.pair_geometry`, at most 40 m away
  lo_all    lo + every node without a successor gets a long link to EVERY node without a predecessor inside the allowed gate
  hi        the GT links with their fitted handle scales: `qformat.oracle_prediction` on the lattice
Read-outs (`qgraph_eval.readout`, unchanged; every card exists for both):
  sec0      OF RECORD: every node carries section 0, so no GT piece id reaches the read-out. All kept links are then 'inside' one section and `graph_from_vertices` cuts them into maximal chains between nodes with
            in-degree != 1 or out-degree != 1: every fork, merge, dead end and every false extra link cuts a chain, each chain is one piece, a piece is joined to every piece that starts at its last node
  gtsec     the GT piece id as section: links inside a piece are chained, a link across pieces is a connector piece of its own. A false link at a mid-piece node is inert there (flattering); the hi check uses it
Numbers per arm and read-out: the graph numbers of the standard scorer (`readout` -> `graph_counts` -> `graph_numbers`, what `score_frames` reports as 'graph'; scored inside the lattice block because `LaneRef` builds a node
table; the scorer's 'nodes' block is not reported) and the LINK TABLES (the same for both read-outs), predicted links with p >= thr against the GT links of the table:
  classes of a GT link (E346): junction = long (chord > 8 m) and more than half of the lane GT points of its path are connector points; lane_long = long otherwise; short = chord <= 8 m (conn_short = its part on connector paths)
  true positive of a class = a predicted link that is a GT link of the class; false positive of junction AND of lane_long = a predicted link with chord > 8 m that is no GT link (a false long link has no class of its own);
  false positive of short = a predicted link with chord <= 8 m that is no GT link; by chord (below 8, 8 - 20, above 20 m) the same with the chord of the pair. `link_junction_P / _R / _F1` is the second gate metric of the card.
  `_gate_out` = share of the GT links of the group outside the allowed gate (the head cannot deliver them); `scale_err_a / _b` = mean |predicted - fitted handle scale| over the true-positive long links with a fit.
Degraded nodes (`--drop`): a share of the nodes of every VAL frame is removed at random (`keep_nodes`: seeded by the frame's token, the same removal for the head and the rules) before anything reads them; the GT links of such
a frame are the GT links contracted over the removed nodes (`sub_table`). Tests: training/grid/test_jtok.py
"""
import argparse
import json
import math
import os
import pickle
import subprocess
import sys
import time
import zlib

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jtok as JT  # noqa: E402
import lane_tok_data as TD  # noqa: E402
import qformat  # noqa: E402
import qgraph_eval as QE  # noqa: E402

RUNS = os.path.join(ROOT, "runs")
CARDS = os.path.join(RUNS, "eval")
PARTIAL = os.path.join(ROOT, "temp", "e357", "out")                         # a card of a subset of the frames (--max) is no evidence: never in runs/eval/
ARMS = ("lo", "lo_near", "lo_all", "hi")
READOUTS = ("sec0", "gtsec")
REF = {2.0: "e346_l2", 2.5: "e346_l25"}                                     # the stored card of the same code on the lattice (E346 / E356): arm hi with the gtsec read-out must reproduce it
CHECK = ("topo_F1", "lanes_found", "F1@1.0", "junction_F1")
CLASSES = ("short", "lane_long", "junction")
CHORDS = (("c0_8", -1.0, JT.SHORT), ("c8_20", JT.SHORT, 20.0), ("c20", 20.0, float("inf")))      # lo < chord <= hi (m)
SEC_VOID = ("pieces_per_gt_piece", "id_switches_per_100m")                  # the scorer's section numbers: meaningless when every node carries section 0
KEEP_P = 0.05                                                               # links below this probability are not written into a dump
HEAD = ("topo_F1", "junction_F1", "lanes_found", "F1@1.0", "dup_frac", "vert100", "link_junction_P", "link_junction_R", "link_junction_F1", "link_lane_long_F1", "link_short_F1")


# ------------------------------------------------------------------------------------------------------------------------------ the rule controls
def rule_links(pos, dirv, reach=8.0, cone=45.0, turn=60.0):
    """arm lo -> (src, dst): every node one link to the nearest other node within `reach` m that lies within `cone` degrees of its heading and whose heading differs by at most `turn` degrees"""
    M = len(pos)
    if M < 2:
        return np.zeros(0, int), np.zeros(0, int)
    ch = pos[None, :, :] - pos[:, None, :]                                   # [i, j] = pos_j - pos_i
    L = np.linalg.norm(ch, axis=2)
    ok = (L > 1e-9) & (L <= reach) & ((ch * dirv[:, None, :]).sum(2) >= math.cos(math.radians(cone)) * L) & (dirv @ dirv.T >= math.cos(math.radians(turn)))
    d = np.where(ok, L, np.inf)
    j = d.argmin(1)
    src = np.flatnonzero(np.isfinite(d[np.arange(M), j]))
    return src, j[src]


def open_links(pos, dirv, src, dst, every, reach=40.0):
    """the extra LONG links (chord above 8 m) of lo_near (every = False) / lo_all (True) -> (src, dst): from every node without a successor in the links (src, dst) to the nearest node without a predecessor inside the
    allowed gate and within `reach` m, or to every such node inside the gate"""
    M = len(pos)
    if M < 2:
        return np.zeros(0, int), np.zeros(0, int)
    free_out, free_in = np.ones(M, bool), np.ones(M, bool)
    free_out[src], free_in[dst] = False, False
    g, allowed = JT.pair_geometry(torch.from_numpy(pos).float()[None], torch.from_numpy(dirv).float()[None])
    L = g[0, ..., 2].numpy() * 20.0
    ok = allowed[0].numpy() & free_out[:, None] & free_in[None, :] & (L > JT.SHORT)
    if every:
        return np.nonzero(ok)
    d = np.where(ok & (L <= reach), L, np.inf)
    j = d.argmin(1)
    s = np.flatnonzero(np.isfinite(d[np.arange(M), j]))
    return s, j[s]


def rule_prediction(t, arm):
    src, dst = rule_links(t["pos"], t["dirv"])
    if arm != "lo":
        s, d = open_links(t["pos"], t["dirv"], src, dst, every=arm == "lo_all")
        src, dst = np.concatenate([src, s]), np.concatenate([dst, d])
    return edge_prediction(t, src, dst)


# ------------------------------------------------------------------------------------------------------------------------------ the writer
def edge_prediction(t, src, dst, p=None, a=None, b=None):
    """THE WRITER: the nodes of a table + an edge list -> a prediction in the format of `qformat` (nodes with e = 1, flags and section = the GT piece: for the read-out only; links with p (1), long = chord > 8 m,
    the handle scales a, b (1/3), no stored curves: the read-out draws them)"""
    src, dst = np.asarray(src, int), np.asarray(dst, int)
    M, L = len(t["pos"]), len(src)
    pos = t["pos"].astype(float)
    nodes = dict(pos=pos, dirv=t["dirv"].astype(float), e=np.ones(M), flags=t["flags"].astype(float), s=t["sstart"].astype(float), cell=t["cell"].astype(int), section=t["piece"].astype(int))
    val = lambda v: np.full(L, 1 / 3) if v is None else np.asarray(v, float)  # noqa: E731
    links = dict(src=src, dst=dst, p=np.ones(L) if p is None else np.asarray(p, float), long=np.linalg.norm(pos[src] - pos[dst], axis=1) > JT.SHORT, a=val(a), b=val(b), curves=None)
    return dict(nodes=nodes, links=links, assign=None, maps=None)


def gt_prediction(t):
    """the GT links of a table through the writer (fitted scales for the long links that have a fit, 1/3 else): scores as `qformat.oracle_prediction` (tested)"""
    s = np.where(t["long"][:, None] & np.isfinite(t["scale"]), t["scale"], 1 / 3)
    return edge_prediction(t, t["src"], t["dst"], a=s[:, 0], b=s[:, 1])


def pair_prediction(t, p, scale, keep=KEEP_P):
    """the head's output of one frame -> prediction: p (N, N) = sigmoid(logit) x allowed, scale (N, N, 2); links = the pairs with p >= keep, the predicted handle scales for chords above 8 m and 1/3 below"""
    M = len(t["pos"])
    src, dst = np.nonzero(p[:M, :M] >= keep)
    lg = np.linalg.norm(t["pos"][src] - t["pos"][dst], axis=1) > JT.SHORT
    return edge_prediction(t, src, dst, p[src, dst], np.where(lg, scale[src, dst, 0], 1 / 3), np.where(lg, scale[src, dst, 1], 1 / 3))


def with_readout(pred, readout):
    """sec0: every node in section 0 (no GT piece id reaches the read-out); gtsec: the prediction as written"""
    assert readout in READOUTS, readout
    return pred if readout == "gtsec" else dict(pred, nodes=dict(pred["nodes"], section=np.zeros(len(pred["nodes"]["pos"]), int)))


# ------------------------------------------------------------------------------------------------------------------------------ degraded nodes
def keep_nodes(token, n, drop, seed):
    """(n,) bool, the nodes of a frame that stay: each is removed with probability `drop` by a generator seeded with (seed, the frame's token)"""
    return np.ones(n, bool) if drop <= 0 else np.random.default_rng([seed, zlib.crc32(token.encode())]).random(n) >= drop


def sub_table(t, keep):
    """the table of the kept nodes. Links: a GT link between two kept nodes stays (with its scale and class); a chain of GT links through removed nodes becomes ONE link between its kept ends (conn = a link of the
    chain is on a connector path, no fitted scale), unless the direct link exists"""
    keep = np.asarray(keep, bool)
    new = np.cumsum(keep) - 1
    outs = [[] for _ in keep]
    for l, s in enumerate(t["src"]):
        outs[s].append(l)
    links = {}                                                              # (a, b) -> (conn, scale)
    for l, (a, b) in enumerate(zip(t["src"], t["dst"])):
        if keep[a] and keep[b] and a != b:
            links[(int(a), int(b))] = (bool(t["conn"][l]), t["scale"][l])
    for a in np.flatnonzero(keep):
        stack, seen = [(l, bool(t["conn"][l])) for l in outs[a] if not keep[t["dst"][l]]], set()
        while stack:
            l, conn = stack.pop()
            b = int(t["dst"][l])
            if keep[b]:
                if b != a:
                    links.setdefault((int(a), b), (conn, np.full(2, np.nan)))
            elif b not in seen:
                seen.add(b)
                stack += [(l2, conn or bool(t["conn"][l2])) for l2 in outs[b]]
    src, dst = (np.array([k[i] for k in links], int) for i in (0, 1))
    out = {k: t[k][keep] for k in ("pos", "dirv", "cell", "flags", "piece", "sstart")}
    out.update(src=new[src].astype(np.int32), dst=new[dst].astype(np.int32), long=np.linalg.norm(t["pos"][src] - t["pos"][dst], axis=1) > JT.SHORT,
               scale=np.array([v[1] for v in links.values()], float).reshape(-1, 2), conn=np.array([v[0] for v in links.values()], bool))
    return out


class Degraded(torch.utils.data.Dataset):
    """the frames of a `TokData` with a share of the nodes removed before anything reads them"""

    def __init__(self, ds, drop, seed):
        self.ds, self.drop, self.seed = ds, drop, seed

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        o = self.ds[i]
        return dict(o, **sub_table(o, keep_nodes(o["token"], len(o["pos"]), self.drop, self.seed)))


# ------------------------------------------------------------------------------------------------------------------------------ link tables
def link_counts(t, src, dst, a=None, b=None):
    """the additive link counts of one frame (module docstring): per class and chord bin [true positives, false positives, GT links missed, GT links outside the allowed gate], conn_short the same for its GT links
    (no false positives of its own), scale = [sum |a - fitted a|, sum |b - fitted b|, links] over the true-positive long links with a fit (a, b = the predicted scales; None: not counted)"""
    M = len(t["pos"])
    pos = t["pos"].astype(float)
    L = np.linalg.norm(pos[None] - pos[:, None], axis=2)
    G = np.zeros((M, M), np.int8)                                           # 0 no GT link, 1 short, 2 lane_long, 3 junction
    G[t["src"], t["dst"]] = np.where(~t["long"], 1, np.where(t["conn"], 3, 2))
    P = np.zeros((M, M), bool)
    P[src, dst] = True
    allowed = JT.pair_geometry(torch.from_numpy(pos).float()[None], torch.from_numpy(t["dirv"]).float()[None])[1][0].numpy() if M else np.zeros((0, 0), bool)
    false = P & (G == 0)
    cnt = lambda gt, fp: np.array([(P & gt).sum(), fp.sum(), (gt & ~P).sum(), (gt & ~allowed).sum()], float)  # noqa: E731
    c = {name: cnt(G == k, false & ((L <= JT.SHORT) if name == "short" else (L > JT.SHORT))) for k, name in enumerate(CLASSES, 1)}
    c.update({name: cnt((G > 0) & (L > lo) & (L <= hi), false & (L > lo) & (L <= hi)) for name, lo, hi in CHORDS})
    cs = ~t["long"] & t["conn"]
    S = np.zeros((M, M), bool)
    S[t["src"][cs], t["dst"][cs]] = True
    c["conn_short"] = cnt(S, np.zeros((M, M), bool))
    c["scale"] = np.zeros(3)
    if a is not None and len(src):
        GS = np.full((M, M, 2), np.nan)
        GS[t["src"], t["dst"]] = t["scale"]
        m = (G[src, dst] >= 2) & np.isfinite(GS[src, dst, 0])
        c["scale"] = np.array([np.abs(np.asarray(a)[m] - GS[src[m], dst[m], 0]).sum(), np.abs(np.asarray(b)[m] - GS[src[m], dst[m], 1]).sum(), m.sum()], float)
    return c


def link_numbers(cs):
    """the pooled link counts of a set of frames -> flat {name: float}. P / R / F1 by the convention of the scorer (`qgraph_eval._prf`): an empty denominator is 0.0 when the other side is not empty (a head without a
    long link has junction P = F1 = 0), NaN only when there is neither a GT nor a predicted link"""
    div = lambda x, y: float(x) / float(y) if y > 0 else float("nan")  # noqa: E731
    out = {}
    for name in CLASSES + tuple(n for n, _, _ in CHORDS) + ("conn_short",):
        tp, fp, fn, gate = sum(c[name] for c in cs)
        p, r, f1 = QE._prf(tp, tp + fp, tp + fn)
        out.update({f"link_{name}_R": r, f"link_{name}_n_gt": tp + fn, f"link_{name}_gate_out": div(gate, tp + fn)})
        if name != "conn_short":                                              # a part of the short class: it has no false positives of its own
            out.update({f"link_{name}_P": p, f"link_{name}_F1": f1, f"link_{name}_n_pred": tp + fp})
    sa, sb, n = sum(c["scale"] for c in cs)
    out.update(scale_err_a=div(sa, n), scale_err_b=div(sb, n), scale_n=n)
    return out


# ------------------------------------------------------------------------------------------------------------------------------ scoring
def frame_counts(pred, f, t, thr):
    """one frame of one arm (inside the lattice block): the scorer's graph counts under both read-outs and the link counts of the links with p >= thr against the GT links of the table `t`"""
    lk = pred["links"]
    _, L = QE.kept(pred, thr_link=thr)
    scales = (lk["a"][L], lk["b"][L]) if lk.get("curves") is None else (None, None)        # the oracle keeps its scales in the curves
    out = {r: QE.graph_counts(QE.readout(with_readout(pred, r), thr_link=thr)[0], f) for r in READOUTS}
    out["links"] = link_counts(t, np.asarray(lk["src"], int)[L], np.asarray(lk["dst"], int)[L], *scales)
    return out


def numbers(cs, readout):
    """the counts of a set of frames -> the flat metrics of one read-out: the graph numbers of the scorer + the link tables. sec0 has one section by construction: its two section numbers are void and left out"""
    g = QE.graph_numbers([c[readout] for c in cs])
    return dict({k: v for k, v in g.items() if readout != "sec0" or k not in SEC_VOID}, **link_numbers([c["links"] for c in cs]))


def score_arms(frames, make, res, thr=0.5, log=print):
    """make(frame) -> (table, {arm: prediction}) for one frame; -> {arm: {readout: flat metrics}}. Everything runs inside the lattice block (`LaneRef` builds a node table at score time: outside it would silently be a
    2 m table); the lane GT of a frame is loaded once for all arms and let go"""
    cs, t0 = {}, time.time()
    with JT.lattice(res):
        JT.check_lattice(res)
        for i, f in enumerate(frames):
            t, preds = make(f)
            for arm, p in preds.items():
                assert np.array_equal(p["nodes"]["pos"], t["pos"]), f"arm {arm}, frame {f.k}: the nodes of the prediction are not the nodes of the table (a stale dump or cache)"
                cs.setdefault(arm, []).append(frame_counts(p, f, t, thr))
            f._lane = f._ref = None
            if log and (i + 1) % 100 == 0:
                log(f"   {i + 1}/{len(frames)} frames, {time.time() - t0:.0f} s", flush=True)
    return {arm: {r: numbers(c, r) for r in READOUTS} for arm, c in cs.items()}


# ------------------------------------------------------------------------------------------------------------------------------ cards
def commit():
    h = os.environ.get("AR_COMMIT", "")[:10]
    if h:
        return h
    try:
        return subprocess.run(["git", "-c", "safe.directory=*", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip() + "+?"
    except Exception:  # noqa: BLE001
        return "no-git"


def make_card(name, n_frames, metrics, readout, **info):
    ok = {k: float(v) for k, v in metrics.items() if v == v and math.isfinite(v)}
    return dict(schema=1, name=name, split="val", frames=n_frames, seed=info.get("seed", 0),
                protocol=dict(scorer="qgraph_eval.readout -> graph_counts -> graph_numbers against the old graph GT (graph_gt_val.pkl) + the link tables of jtok_eval.py", reference="val, the frames of qgraph_eval.val_frames",
                              readout=readout + (" (of record: no GT section ids)" if readout == "sec0" else " (GT piece ids as sections)")),
                metrics=ok, commit=commit(), date=time.strftime("%Y-%m-%d %H:%M"), info=dict(info, lane_gt_signature=os.path.basename(TD.LD.lane_store()[1]), nan_metrics=sorted(k for k in metrics if k not in ok)))


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(path + ".tmp", path)


def suffix(res=2.5, drop=0.0, seed=0):
    """the part of a file name that says which lattice (when not 2.5 m) and which node removal"""
    return ("" if res == 2.5 else f"_l{res:g}") + (f"_drop{drop:g}_s{seed}" if drop > 0 else "")


def show(tag, m):
    print(f"{tag:<28}" + "  ".join(f"{k} {m.get(k, float('nan')):.4f}" for k in HEAD), flush=True)


def val_tables(res, drop=0.0, seed=0):
    """-> (the 873 VAL frames of the scorer, table(k): the GT table of frame k on the lattice with the node removal applied)"""
    frames = QE.val_frames()
    ds = TD.TokData("val", res, feats=False)
    assert [f.token for f in frames] == ds.tokens, "the VAL frames of the scorer are not the frames of the data set, in order"

    def table(k):
        t = ds.table(k)
        return sub_table(t, keep_nodes(ds.tokens[k], len(t["pos"]), drop, seed)) if drop > 0 else t
    return frames, table


# ------------------------------------------------------------------------------------------------------------------------------ commands
def check(a):
    """the kill check: arm hi with the gtsec read-out on the lattice against the stored card of the same code (E346 / E356) -> 0 (PASS) / 1 (FAIL)"""
    ref = os.path.join(ROOT, a.ref) if a.ref else os.path.join(CARDS, REF[a.lattice], "val.json")          # --ref: a path from the problem folder
    card = json.load(open(ref))
    frames = QE.val_frames()
    frames = frames[::math.ceil(len(frames) / a.max)] if a.max else frames
    sig = os.path.basename(TD.LD.lane_store()[1])
    assert card["frames"] == len(frames), f"{ref} holds {card['frames']} frames, this check {len(frames)}"
    assert card["info"].get("lane_gt_signature") in (None, sig), f"{ref} was made with the lane GT {card['info'].get('lane_gt_signature')}, this checkout reads {sig}"
    cs = []
    with JT.lattice(a.lattice):
        JT.check_lattice(a.lattice)
        for f in frames:
            cs.append(QE.graph_counts(QE.readout(qformat.oracle_prediction(f.lane()))[0], f))
            f._lane = f._ref = None
    got = QE.graph_numbers(cs)
    bad = [k for k in CHECK if abs(got[k] - card["metrics"][k]) > 1e-6]
    for k in CHECK:
        print(f"   {k}: {got[k]:.9f} stored {card['metrics'][k]:.9f}")
    print(f"{'FAIL' if bad else 'PASS'}: arm hi (gtsec) on the {a.lattice:g} m lattice, {len(frames)} frames, against {os.path.relpath(ref, ROOT)}" + (f": {bad} differ by more than 1e-6" if bad else ""))
    return int(bool(bad))


def ctl(a):
    if a.check:
        return check(a)
    frames, table = val_tables(a.lattice, a.drop, a.drop_seed)
    n_all = len(frames)
    frames = frames[::math.ceil(n_all / a.max)] if a.max else frames
    sfx = suffix(a.lattice, a.drop, a.drop_seed)
    path = {(arm, r): os.path.join(PARTIAL, f"e357_{arm}{sfx}_{r}_n{a.max}.json") if a.max else os.path.join(CARDS, f"e357_{arm}{sfx}_{r}", "val.json") for arm in ARMS for r in READOUTS}
    held = [p for p in path.values() if os.path.exists(p) and not a.max]
    assert not held, f"a stored card is never overwritten: {held}"

    def make(f):
        t = table(f.k)
        if a.drop > 0:
            hi = gt_prediction(t)                                             # the GT links contracted over the removed nodes (no oracle exists for them)
        else:
            hi = qformat.oracle_prediction(f.lane())
            assert all(np.array_equal(hi["nodes"]["pos"] if k == "pos" else hi["links"][k], t[k]) for k in ("pos", "src", "dst", "long")), f"frame {f.k}: the cached table is not the oracle's node table (a stale cache)"
        return t, dict({arm: rule_prediction(t, arm) for arm in ARMS[:3]}, hi=hi)

    t0 = time.time()
    res = score_arms(frames, make, a.lattice)
    # the kill of the card: the BEST rule above the GT links on topo F1 = the scale is upside down. Not under --drop: arm hi is no oracle there
    kill = [] if a.drop > 0 else [r for r in READOUTS if max(res[arm][r]["topo_F1"] for arm in ARMS[:3]) > res["hi"][r]["topo_F1"]]
    note = dict(kill="a rule arm is above arm hi on topo F1") if kill else {}
    for (arm, r), p in path.items():
        own = dict(note, hi_is_no_oracle="under --drop arm hi = the GT links CONTRACTED over the removed nodes; a contracted link has no fitted handle scales (1/3): a reference, not a ceiling") if arm == "hi" and a.drop > 0 else note
        write_json(p, make_card(f"e357_{arm}{sfx}_{r}", len(frames), res[arm][r], r, arm=arm, lattice=a.lattice, drop=a.drop, drop_seed=a.drop_seed, thr=0.5, frame_stride=math.ceil(n_all / len(frames)),
                                wall_s=round(time.time() - t0, 1), **own))
        show(f"{arm} {r}", res[arm][r])
    print(f"cards: {os.path.relpath(PARTIAL if a.max else CARDS, ROOT)}/e357_<arm>{sfx}_<readout>, {len(frames)} frames, {time.time() - t0:.0f} s")
    if kill:
        print(f"KILL: a rule arm scores above arm hi on topo F1 ({kill}): the scale is upside down", file=sys.stderr)
    return int(bool(kill))


def dump(a, runs=RUNS):
    assert a.cpu or torch.cuda.is_available(), "no GPU: the read-outs of a card are dumped on the GPU; --cpu only for tests"
    dev = "cpu" if a.cpu else "cuda"
    if dev == "cuda":                                                         # D100: a dump next to two trainings crashed one of them
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jtok_eval dump {a.run}")
    m, args = JT.load_model(a.run, dev, runs)
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    ds = TD.TokData("val", args["lattice"])
    data = Degraded(ds, a.drop, a.drop_seed) if a.drop > 0 else ds
    n = min(a.max or len(ds), len(ds))
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(data, range(n)), batch_size=8, shuffle=False, num_workers=0 if a.cpu else 2, collate_fn=TD.collate)
    preds = []
    with torch.no_grad():
        for b in dl:
            out = JT.forward_batch(m, b)
            p = (out["logit"].float().sigmoid() * out["allowed"]).cpu().numpy()
            sc = out["scale"].float().cpu().numpy()
            preds += [dict(pair_prediction(t, p[i], sc[i]), token=b["token"][i], row=int(b["row"][i])) for i, t in enumerate(b["table"])]
    path = os.path.join(runs, a.run, f"tok_val{suffix(args['lattice'], a.drop, a.drop_seed)}{f'_n{n}' if n < len(ds) else ''}.pkl")      # a partial dump never takes the name the score reads
    with open(path, "wb") as fh:
        pickle.dump(preds, fh)
    print(f"written {os.path.relpath(path, ROOT)}: {len(preds)} frames, {sum(len(p['links']['src']) for p in preds) / max(len(preds), 1):.0f} links with p >= {KEEP_P} per frame, device {dev}")
    return path


def score(a, runs=RUNS):
    ck = torch.load(os.path.join(runs, a.run, "model.pt"), map_location="cpu")
    args = ck["args"]
    assert args["seed"] == a.seed, f"--seed {a.seed} does not match the run (seed {args['seed']})"
    sfx = suffix(args["lattice"], a.drop, a.drop_seed)
    with open(os.path.join(runs, a.run, f"tok_val{sfx}.pkl"), "rb") as fh:
        preds = pickle.load(fh)
    frames, table = val_tables(args["lattice"], a.drop, a.drop_seed)
    assert len(preds) == len(frames) == 873 and [p["token"] for p in preds] == [f.token for f in frames], f"the dump holds {len(preds)} frames: the score needs the 873 VAL frames in order"
    res = score_arms(frames, lambda f: (table(f.k), {"p": preds[f.k]}), args["lattice"], a.thr)["p"]
    lane_long = int(res["sec0"]["link_lane_long_n_gt"])                       # the junction table's false positives are ALL false long links: 'false long links against junction links' only when no GT long lane link exists
    for r in READOUTS:
        name = f"tok_card_thr{a.thr:g}{sfx}_{r}"
        write_json(os.path.join(runs, a.run, name + ".json"), make_card(f"{a.run}_{name}", len(frames), res[r], r, run=a.run, seed=a.seed, lattice=args["lattice"], thr=a.thr, drop=a.drop, drop_seed=a.drop_seed,
                                                                         steps=ck.get("steps"), head_params_m=sum(v.numel() for k, v in ck["state"].items() if k.startswith("tok.")) / 1e6, lane_long_gt=lane_long))
        show(f"{a.run} thr {a.thr:g} {r}", res[r])
    if lane_long:
        print(f"WARNING: {lane_long} GT long LANE links (lane_long_gt in the cards): the junction-link precision also counts false long links that belong to them")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name_ in ("ctl", "dump", "score"):
        s = sub.add_parser(name_)
        s.add_argument("--drop", type=float, default=0.0, help="share of the nodes of every frame removed at random before anything reads them (the degraded-node read-out: 0.17)")
        s.add_argument("--drop-seed", type=int, default=0)
        if name_ == "ctl":
            s.add_argument("--lattice", type=float, default=2.5)
            s.add_argument("--max", type=int, default=0, help="every ceil(873 / N)-th frame; such cards go to temp/e357/out/, never to runs/eval/")
            s.add_argument("--check", action="store_true", help="only the kill check of arm hi against the stored card of the lattice (PASS / FAIL, exit 0 / 1)")
            s.add_argument("--ref", default=None, help="--check: another reference card (with --max: the card of the same frame stride)")
        else:
            s.add_argument("--run", required=True)
            s.add_argument("--seed", type=int, required=True)
        if name_ == "dump":
            s.add_argument("--cpu", action="store_true", help="tests only: dump on the CPU")
            s.add_argument("--max", type=int, default=0, help="only the first N frames (a test of the pipeline; the score needs all 873)")
        if name_ == "score":
            s.add_argument("--thr", type=float, default=0.5)
    a_ = ap.parse_args()
    torch.set_num_threads(2)
    r_ = {"ctl": ctl, "dump": dump, "score": score}[a_.cmd](a_)
    sys.exit(r_ if isinstance(r_, int) else 0)
