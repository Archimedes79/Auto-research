"""E395 (D142, card experiments/E395_catmull-rom-readout-and-position-head): Catmull-Rom curves through the delivered nodes. Step 1 = the read-out of the token head's links as Catmull-Rom (CR) curves or chords instead of the
Hermite connectors of the record, plus ORACLE node positions as the measured headroom of node accuracy; step 2 reads a head whose nodes were moved by a position head. CPU, the standard scorer (`jtoken_eval.score_preds`), VAL 873
frames at the D115 cell (short 0.3 / long 0.2 / keep 0.6 / none gate off).

    wsl bash tools/env/_py14.sh jcr_eval.py step1 --run e392_c_s0 --seed 0 [--live] [--modes H,P,CR,OLAT+H,OLAT+CR,OFULL+H] [--out runs/eval/e395_step1]   # -> <out>/<run>.json (never overwritten)
    wsl bash tools/env/_py14.sh jcr_eval.py selftest

A mode is `<positions>+<connector>`: connector H = the record (Hermite with the node headings and the head's handle scales: `links['curves']` None), P = the straight chord sampled at 0.5 m, CR = uniform Catmull-Rom;
CRS = Catmull-Rom for the short links (chord <= 8 m) and the Hermite connector of the record for the long ones (the long links are where the node heading and the learned handle scales carry the curvature: a Catmull-Rom curve of a long link has only reflected ghosts there);
positions: none (the delivered tokens), OLAT = every delivered token with a GT node (`gt_id >= 0`) moved onto that GT node's line along its normal (the rotation of `dirv` by 90 degrees), OFULL = moved onto the GT node itself
(the oracles need the cache dump: its tokens carry `gt_id`); `H` alone is `+H`. Catmull-Rom curve of a link i -> j: P1 = P_i, P2 = P_j; the ghost P0 = the delivered predecessor k of i (a link k -> i) with the smallest turning angle
between (P_i - P_k) and (P_j - P_i), P3 = the delivered successor l of j with the smallest turning angle between (P_l - P_j) and (P_j - P_i); a missing ghost is the reflection (2 P_i - P_j, 2 P_j - P_i). The curve is sampled at
max(2, ceil(chord / 0.5 m) + 1) points uniform in t and handed to the scorer as `links['curves']` (`qgraph_eval.readout` takes a given curve instead of the Hermite one). No GT field enters a mode without `OLAT` / `OFULL`."""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for sub_ in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jtoken_eval as JE  # noqa: E402
import qgraph_eval as QE  # noqa: E402
import qtargets as QT  # noqa: E402

STEP = QE.STEP                                                                  # 0.5 m: the scorer's sampling of the delivered curves
EVAL = os.path.join(ROOT, "runs", "eval")                                      # where `verdict` reads the cards (the selftest points it at a fixture folder)
D115 = ["--link-thr", "0.3", "--link-thr-long", "0.2", "--keep-thr", "0.6", "--no-none-gate"]
CONNECTORS = ("H", "P", "CR", "CRC", "CRS")
POSITIONS = ("", "OLAT", "OFULL")


# ------------------------------------------------------------------------------------------------------------------------------ curves
def cr_points(P0, P1, P2, P3, n):
    """n points of the uniform Catmull-Rom segment from P1 (t = 0) to P2 (t = 1) with the ghost nodes P0 and P3: 0.5 ((2 P1) + (-P0 + P2) t + (2 P0 - 5 P1 + 4 P2 - P3) t^2 + (-P0 + 3 P1 - 3 P2 + P3) t^3)"""
    t = np.linspace(0.0, 1.0, n)[:, None]
    P0, P1, P2, P3 = (np.asarray(p, float) for p in (P0, P1, P2, P3))
    return 0.5 * (2 * P1 + (-P0 + P2) * t + (2 * P0 - 5 * P1 + 4 * P2 - P3) * t ** 2 + (-P0 + 3 * P1 - 3 * P2 + P3) * t ** 3)


def cr_points_centripetal(P0, P1, P2, P3, n, alpha=0.5):
    """n points of the centripetal Catmull-Rom segment (Barry-Goldman, knot intervals |dP|^alpha): no cusps or loops at unequal chords"""
    P0, P1, P2, P3 = (np.asarray(p, float) for p in (P0, P1, P2, P3))
    t0 = 0.0
    t1 = t0 + max(np.linalg.norm(P1 - P0), 1e-6) ** alpha
    t2 = t1 + max(np.linalg.norm(P2 - P1), 1e-6) ** alpha
    t3 = t2 + max(np.linalg.norm(P3 - P2), 1e-6) ** alpha
    t = np.linspace(t1, t2, n)[:, None]
    a1 = (t1 - t) / (t1 - t0) * P0 + (t - t0) / (t1 - t0) * P1
    a2 = (t2 - t) / (t2 - t1) * P1 + (t - t1) / (t2 - t1) * P2
    a3 = (t3 - t) / (t3 - t2) * P2 + (t - t2) / (t3 - t2) * P3
    b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
    b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
    return (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2


def turn(u, v):
    """the angle (radians) between the vectors u and v; 0 when one of them has no length"""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    return float(np.arccos(np.clip((u @ v) / (nu * nv), -1.0, 1.0)))


def ghosts(pos, src, dst, preds, succs, l):
    """the ghost nodes (P0, P3) of link l (module docstring); the first candidate wins a tie; a neighbour that is the other end of the link itself (k == j, m == i: a two-node cycle) is skipped"""
    i, j = int(src[l]), int(dst[l])
    pi, pj = pos[i], pos[j]
    d = pj - pi
    best, p0 = None, 2 * pi - pj
    for k in preds.get(i, ()):
        if k != j:
            a = turn(pi - pos[k], d)
            if best is None or a < best[0]:
                best, p0 = (a, k), pos[k]
    best, p3 = None, 2 * pj - pi
    for m in succs.get(j, ()):
        if m != i:
            a = turn(pos[m] - pj, d)
            if best is None or a < best[0]:
                best, p3 = (a, m), pos[m]
    return p0, p3


def curves_of(pred, kind):
    """the `links['curves']` of the prediction for the connector kind 'CR', 'CRC', 'P' or 'CRS' (a list of (n, 2) arrays or None = the Hermite connector of the record; first point the source node, last the destination node)"""
    pos = np.asarray(pred["nodes"]["pos"], float).reshape(-1, 2)
    src, dst = np.asarray(pred["links"]["src"], int), np.asarray(pred["links"]["dst"], int)
    preds, succs = {}, {}
    for s, d in zip(src.tolist(), dst.tolist()):
        preds.setdefault(d, []).append(s)
        succs.setdefault(s, []).append(d)
    out = []
    for l in range(len(src)):
        i, j = int(src[l]), int(dst[l])
        if kind == "CRS" and bool(np.asarray(pred["links"]["long"])[l]):
            out.append(None)
            continue
        n = max(2, int(math.ceil(np.linalg.norm(pos[j] - pos[i]) / STEP)) + 1)
        if kind == "P":
            t = np.linspace(0.0, 1.0, n)[:, None]
            out.append(pos[i] * (1 - t) + pos[j] * t)
        else:
            p0, p3 = ghosts(pos, src, dst, preds, succs, l)
            out.append((cr_points_centripetal if kind == "CRC" else cr_points)(p0, pos[i], pos[j], p3, n))
    return out


# ------------------------------------------------------------------------------------------------------------------------------ oracle positions
def move_tokens(tokens, gt_id, gt_pos, how):
    """a copy of the tokens (`tokens_np` of a frame) with every token that has a GT node (`gt_id >= 0`) moved: OLAT along its normal onto the GT node's line, OFULL onto the GT node"""
    t = dict(tokens)
    pos = np.array(tokens["pos"], float)
    dirv = np.asarray(tokens["dirv"], float)
    gt_id = np.asarray(gt_id)
    ok = (gt_id >= 0) & (gt_id < len(gt_pos))
    G = np.zeros_like(pos)
    G[ok] = np.asarray(gt_pos, float)[gt_id[ok]]
    if how == "OFULL":
        pos[ok] = G[ok]
    else:
        n = np.stack([-dirv[:, 1], dirv[:, 0]], 1)
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        pos[ok] += ((G[ok] - pos[ok]) * n[ok]).sum(1)[:, None] * n[ok]
    t["pos"] = pos
    return t


# ------------------------------------------------------------------------------------------------------------------------------ step 1
def split_mode(mode):
    p, _, c = mode.rpartition("+")
    assert p in POSITIONS and c in CONNECTORS, f"mode {mode}: <{'|'.join(POSITIONS) or ''}>+<{'|'.join(CONNECTORS)}>"
    return p, c


def mode_preds(r, decf, a, mode, gt):
    """the predictions of one mode for every frame of the dump `r` (decoded frames `decf`)"""
    pos_how, conn = split_mode(mode)
    out = []
    for n, (tk, f) in enumerate(zip(r["toks"], decf)):
        tokens = move_tokens(tk, gt(n)["id"], gt(n)["pos"], pos_how) if pos_how else tk
        pred = JE.frame_prediction(tokens, f, "rec", a.keep_thr)
        if conn != "H":
            pred["links"]["curves"] = curves_of(pred, conn)
        out.append(pred)
    return out


def gt_nodes(r):
    """n -> dict(id (K,) the cache's gt_id, pos / dir (Ng, 2) the GT table's node positions and directions, src / dst the GT links) of the n-th frame of the dump; a live dump has no cache labels"""
    assert not r["live"], "the oracle modes need the cache dump (its tokens carry gt_id), not a live dump"
    import lane_tok_data as LT
    td = LT.TokData("val", res=2.0, feats=False)
    ds, idx = r["ds"], r["idx"]
    cache = {}

    def get(n):
        if n not in cache:
            t = td.table(int(idx[n]))
            cache[n] = dict(id=np.asarray(ds[int(idx[n])]["label"]["gt_id"]), pos=np.asarray(t["pos"], float), dir=np.asarray(t["dirv"], float), src=np.asarray(t["src"], int), dst=np.asarray(t["dst"], int))
        return cache[n]
    return get


def check_record_card(run, graph, live):
    """the kill 'step 1 does not reproduce the record': mode H must equal the stored D115 card of the run (topo F1, F1@0.5, F1@1.0, lanes found, junction F1) within 1e-4"""
    import glob
    base = os.path.basename
    fs = [f for f in sorted(glob.glob(os.path.join(JE.RUNS, run, "token_card_rec_lt0.3_ltl0.2_kt0.6_nng*.json"))) if "train" not in base(f) and "every" not in base(f) and (("live" in base(f)) == live)]
    assert fs, f"no stored D115 card of {run} to check the record against"
    ref = json.load(open(fs[0]))["metrics"]
    bad = {k: (graph[k], ref[k]) for k in ("topo_F1", "F1@0.5", "F1@1.0", "lanes_found", "junction_F1") if abs(graph[k] - ref[k]) > 1e-4}
    assert not bad, f"mode H does not reproduce the stored card {os.path.basename(fs[0])}: {bad}"


def step1(run, seed, live, modes, out_dir, log=print):
    out_path = os.path.join(out_dir, f"{run}.json")
    assert not os.path.exists(out_path), f"never overwritten: {out_path}"
    a = JE.parse_args(["score", "--run", run, "--seed", str(seed)] + D115 + (["--live"] if live else []))
    r = JE.read_dump(a, JE.RUNS, None)
    JE.check_decode(r, a)
    decf = JE.decode_frames(r, a)
    assert "H" in modes, "--modes must contain H: the record check (the STOP kill) and every paired difference read mode H"
    need_gt = any(split_mode(m)[0] for m in modes)
    gt = gt_nodes(r) if need_gt else None
    frames = r["frames"]
    quiet = lambda *x, **k: None  # noqa: E731
    res, card = {}, dict(run=run, seed=seed, live=live, cell="D115: short 0.3, long 0.2, keep 0.6, gate off", frames=len(frames), modes={})
    for m in modes:
        res[m] = JE.score_preds(mode_preds(r, decf, a, m, gt), frames, log=quiet)
        card["modes"][m] = dict(res[m]["graph"])
        if m == "H":
            check_record_card(run, res[m]["graph"], live)
        log(f"{run} {m:10s} " + "  ".join(f"{k} {res[m]['graph'][k]:.4f}" for k in ("topo_F1", "F1@0.5", "F1@1.0", "F1@1.5", "lanes_found", "junction_F1")), flush=True)
    ref = "H" if "H" in res else None
    if ref:
        for m in modes:
            if m != ref:
                card["modes"][m]["paired_vs_H"] = dict(QE.paired(res[m]["rec"], res[ref]["rec"], frames), **JE.paired_junction(res[m]["rec"], res[ref]["rec"], frames))
    os.makedirs(out_dir, exist_ok=True)
    json.dump(JE._jsonable(card), open(out_path, "w"), indent=1)
    log("written", os.path.relpath(out_path, ROOT))
    return card


# ------------------------------------------------------------------------------------------------------------------------------ step 2: nodes moved by a position head
def normals(dirv):
    """the unit normals (rotation of the unit direction by 90 degrees) of dirv (K, 2)"""
    n = np.stack([-dirv[:, 1], dirv[:, 0]], 1)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def move_by_dpos(tokens, dpos):
    """a copy of the tokens with pos + dpos * normal (the position head's refined nodes; `jtoken.refined_pos` in numpy)"""
    t = dict(tokens)
    t["pos"] = np.asarray(tokens["pos"], float) + np.asarray(dpos, float)[:, None] * normals(np.asarray(tokens["dirv"], float))
    return t


def read(run, seed, live=False):
    a = JE.parse_args(["score", "--run", run, "--seed", str(seed)] + D115 + (["--live"] if live else []))
    r = JE.read_dump(a, JE.RUNS, None)
    JE.check_decode(r, a)
    return a, r, JE.decode_frames(r, a)


def score_modes(r, decf, a, kinds, frames):
    """kind -> the standard scorer's result for the dump: 'U' = nodes unmoved, Hermite; 'M' = nodes moved by the dump's dpos, Hermite; 'M+CRS' / 'U+CRS' = with the Catmull-Rom short links"""
    quiet = lambda *x, **k: None  # noqa: E731
    out = {}
    for kind in kinds:
        moved, _, conn = kind.partition("+")
        preds = []
        for tk, f in zip(r["toks"], decf):
            pred = JE.frame_prediction(move_by_dpos(tk, f["dpos"]) if moved == "M" else tk, f, "rec", a.keep_thr)
            if conn:
                pred["links"]["curves"] = curves_of(pred, conn)
            preds.append(pred)
        out[kind] = JE.score_preds(preds, frames, log=quiet)
    return out


def step2(run, seed, control, out_dir, log=print, versus=None):
    """E395 step 2 / the score of a control: the standard scorer on the dump of `run` at the D115 cell: U (unmoved), and when the dump carries `dpos` also M (moved) and M+CRS; the paired differences M - U (the primary contrast), U - control (the topology effect of
    the extra loss) and M - control, with the control's own dump scored as H (nodes unmoved)"""
    out_path = os.path.join(out_dir, f"{run}.json")
    assert not os.path.exists(out_path), f"never overwritten: {out_path}"
    a, r, decf = read(run, seed)
    has = all("dpos" in f for f in r["fr"])
    kinds = ("U", "M", "M+CRS") if has else ("U",)
    res = score_modes(r, decf, a, kinds, r["frames"])
    card = dict(run=run, seed=seed, control=control, cell="D115: short 0.3, long 0.2, keep 0.6, gate off", frames=len(r["frames"]), has_dpos=has, modes={k: dict(v["graph"]) for k, v in res.items()})
    for k in kinds:
        log(f"{run} {k:6s} " + "  ".join(f"{m} {res[k]['graph'][m]:.4f}" for m in ("topo_F1", "F1@0.5", "F1@1.0", "F1@1.5", "lanes_found", "junction_F1")), flush=True)
    pair = lambda x, y: dict(QE.paired(x["rec"], y["rec"], r["frames"]), **JE.paired_junction(x["rec"], y["rec"], r["frames"]))  # noqa: E731
    if has:
        card["paired_M_vs_U"] = pair(res["M"], res["U"])
        card["paired_MCRS_vs_M"] = pair(res["M+CRS"], res["M"])
    if control:
        ac, rc, dc = read(control, seed)
        assert [f["token"] for f in rc["fr"]] == [f["token"] for f in r["fr"]], "the control's dump is not on the same frames"
        rcon = score_modes(rc, dc, ac, ("U",), rc["frames"])["U"]
        card["control"] = dict(run=control, **{m: rcon["graph"][m] for m in JE.KEYS + ("F1@0.5", "F1@1.5")})
        card["paired_U_vs_control"] = pair(res["U"], rcon)
        if has:
            card["paired_M_vs_control"] = pair(res["M"], rcon)
    if versus:                                                                              # another position arm's dump scored as M: M against M (the step-3 comparison p2s against p2)
        av, rv, dv = read(versus, seed)
        assert has and all("dpos" in f for f in rv["fr"]) and [f["token"] for f in rv["fr"]] == [f["token"] for f in r["fr"]], "--versus needs a dump with dpos on the same frames"
        rvm = score_modes(rv, dv, av, ("M",), rv["frames"])["M"]
        card["versus"] = dict(run=versus, **{m: rvm["graph"][m] for m in JE.KEYS + ("F1@0.5", "F1@1.5")})
        card["paired_M_vs_versusM"] = pair(res["M"], rvm)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(JE._jsonable(card), open(out_path, "w"), indent=1)
    log("written", os.path.relpath(out_path, ROOT))
    return card


RANGES = ((0, 15), (15, 30), (30, 50), (50, 100))


def nodeerr(run, seed, out_dir, log=print):
    """the lateral position error of the DELIVERED tokens that carry a GT node (gt_id >= 0), against the GT node's line (the normal of the GT node's direction): U = the tokens as they are, M = moved by the dump's dpos; the lateral rms and the SIGNED mean by ego
    range, the correlation of the remaining error between GT nodes h = 1 .. 5 hops apart, the share of delivered tokens without a GT node and their offsets |dpos|"""
    out_path = os.path.join(out_dir, f"{run}.json")
    assert not os.path.exists(out_path), f"never overwritten: {out_path}"
    a, r, decf = read(run, seed)
    gt = gt_nodes(r)
    has = all("dpos" in f for f in r["fr"])
    acc = {"U": [], "M": []}
    rng, unmatched, delivered, off_un = [], 0, 0, []
    pairs = {k: {h: [] for h in range(1, 6)} for k in ("U", "M")}
    for n, (tk, f) in enumerate(zip(r["toks"], decf)):
        g = gt(n)
        keep = np.asarray(f["keep"], bool)
        pos = np.asarray(tk["pos"], float)
        dp = np.asarray(f["dpos"], float) if has else np.zeros(len(pos))
        gid = np.asarray(g["id"])
        ok = keep & (gid >= 0) & (gid < len(g["pos"]))
        delivered += int(keep.sum())
        unmatched += int((keep & ~ok).sum())
        off_un += list(np.abs(dp[keep & ~ok]))
        if not ok.any():
            continue
        G, nG = g["pos"][gid[ok]], normals(g["dir"][gid[ok]])
        P = pos[ok]
        eU = ((P - G) * nG).sum(1)
        eM = ((P + dp[ok][:, None] * normals(np.asarray(tk["dirv"], float)[ok]) - G) * nG).sum(1)
        acc["U"].append(eU)
        acc["M"].append(eM)
        rng.append(np.linalg.norm(P, axis=1))
        by = {k: dict(zip(gid[ok].tolist(), e)) for k, e in (("U", eU), ("M", eM))}
        succ = {}
        for s_, d_ in zip(g["src"].tolist(), g["dst"].tolist()):
            succ.setdefault(s_, []).append(d_)
        for u in by["U"]:
            front = [u]
            for h in range(1, 6):
                front = [b for x in front for b in succ.get(x, [])]
                for b in front:
                    if b in by["U"]:
                        for k in ("U", "M"):
                            pairs[k][h].append((by[k][u], by[k][b]))
    rng = np.concatenate(rng)
    card = dict(run=run, seed=seed, has_dpos=has, delivered=delivered, unmatched=unmatched, unmatched_share=unmatched / max(delivered, 1), unmatched_abs_dpos_mean=float(np.mean(off_un)) if off_un else 0.0, by={})
    for k in ("U", "M") if has else ("U",):
        e = np.concatenate(acc[k])
        card["by"][k] = dict(lateral_rms=float(np.sqrt(np.mean(e ** 2))), lateral_abs_mean=float(np.abs(e).mean()), lateral_mean=float(e.mean()), n=int(len(e)),
                             by_range={f"{lo}-{hi}": dict(rms=float(np.sqrt(np.mean(e[(rng >= lo) & (rng < hi)] ** 2))), mean=float(e[(rng >= lo) & (rng < hi)].mean()), n=int(((rng >= lo) & (rng < hi)).sum())) for lo, hi in RANGES},
                             corr={str(h): float(np.corrcoef(np.array(pairs[k][h]).T)[0, 1]) for h in range(1, 6)})
        log(f"{run} {k}: lateral rms {card['by'][k]['lateral_rms']:.3f} m, mean {card['by'][k]['lateral_mean']:+.3f}; by range " + ", ".join(f"{lo}-{hi} m {v['rms']:.3f} ({v['mean']:+.3f})" for (lo, hi), v in zip(RANGES, card["by"][k]["by_range"].values())) + "; corr h1..5 " + " ".join(f"{card['by'][k]['corr'][str(h)]:.2f}" for h in range(1, 6)), flush=True)
    log(f"{run}: delivered {delivered}, without a GT node {unmatched} ({100 * card['unmatched_share']:.1f} %), their mean |dpos| {card['unmatched_abs_dpos_mean']:.3f} m")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(JE._jsonable(card), open(out_path, "w"), indent=1)
    log("written", os.path.relpath(out_path, ROOT))
    return card


# ------------------------------------------------------------------------------------------------------------------------------ the verdicts of the card (section 1.3), as code so that the case table is tested
BAND, FLOOR = 0.010, 0.010
GUARDS = {"topo_F1": -0.010, "lanes_found": -0.010, "junction_F1": -0.015, "F1@1.0": -0.005}       # the total M(X) against the control: no difference below these (card section 1.3)


def bar(h):
    """B = max(0.25 x H, 0.010): the bar of a move effect from the headroom H of the oracle positions"""
    return max(0.25 * h, FLOOR)


def step1_class(d):
    """one seed, one connector against H: d = dict(F1@0.5, F1@1.0, topo_F1) of (diff, lo, hi); LOSS when a metric is <= -0.010, GAIN when F1@0.5 and F1@1.0 are >= +0.010 with the intervals above 0, else TIE"""
    d05, d10, dt = d["F1@0.5"], d["F1@1.0"], d["topo_F1"]
    if not all(np.isfinite([x for m in (d05, d10, dt) for x in m])):
        return "NONFINITE"
    if min(d05[0], d10[0], dt[0]) <= -BAND:
        return "LOSS"
    if d05[0] >= BAND and d10[0] >= BAND and d05[1] > 0 and d10[1] > 0:
        return "GAIN"
    return "TIE"


def step1_verdict(seeds):
    """both seeds (a list of two `step1_class` inputs): the class when they agree; GAIN + TIE = TIE; LOSS on one seed with anything else, or GAIN + LOSS = INCONCLUSIVE; also whether the connector 'replaces H at no cost' (every difference >= -0.003 on both seeds)"""
    cl = [step1_class(d) for d in seeds]
    if "NONFINITE" in cl:
        return "INCONCLUSIVE", False
    if cl[0] == cl[1]:
        v = cl[0]
    elif "LOSS" in cl:
        v = "INCONCLUSIVE"
    else:
        v = "TIE"
    free = all(d[m][0] >= -0.003 for d in seeds for m in ("F1@0.5", "F1@1.0", "topo_F1"))
    return v, free


def step2_verdict(seeds):
    """the verdict of a position arm (card section 1.3) from two seeds; each seed = dict(D05, D10: (diff, lo, hi), B05, B10, guards: dict(topo_F1, lanes_found, junction_F1: the total difference against the control), kill_cut, kill_inc: bool). 'The bar is met' includes
    the paired intervals above 0. ORDER: 0 a CUT kill; 1 INCONCLUSIVE (a non-finite number; an INCONCLUSIVE kill; the bar met on exactly one seed; the two seeds differ by more than 0.020 in D05 or D10; a GAIN on both seeds (D05 >= 0.010 with the interval above 0
    and D10 >= 0) but a guard fails on exactly one seed); 2 CUT (D10 or D05 <= -0.010 on a seed and the other seed's value not >= +0.010; a guard fails on both seeds); 3 KEEP-quality (the bar on both seeds, no guard fails); 4 PARTIAL (a GAIN on both seeds, no guard
    fails, the bar not met: real, below the bar, the owner decides); 5 TIE (no effect established: inside the band or the intervals not above 0). INCONCLUSIVE is final for the card (an extension is an addendum written before its runs). Returns (verdict, reason)"""
    LIM = GUARDS
    nums = [x for sd in seeds for k in ("D05", "D10") for x in sd[k]] + [sd["guards"][k] for sd in seeds for k in LIM] + [sd["B05"] for sd in seeds] + [sd["B10"] for sd in seeds]
    gfail = [any(sd["guards"][k] < LIM[k] for k in LIM) for sd in seeds]
    met = [sd["D05"][0] >= sd["B05"] and sd["D05"][1] > 0 and sd["D10"][0] >= sd["B10"] and sd["D10"][1] > 0 for sd in seeds]
    gain = [sd["D05"][0] >= BAND and sd["D05"][1] > 0 and sd["D10"][0] >= 0 for sd in seeds]
    if any(sd["kill_cut"] for sd in seeds):
        return "CUT", "a CUT kill (the refined nodes are worse than the unmoved ones, or the offsets saturate)"
    if not all(np.isfinite(nums)):
        return "INCONCLUSIVE", "a non-finite number in the inputs"
    if any(sd["kill_inc"] for sd in seeds):
        return "INCONCLUSIVE", "a kill fired (loss not finite / no learning / not finished / not parity / tests)"
    if sum(met) == 1:
        return "INCONCLUSIVE", "the bar is met on exactly one seed"
    if abs(seeds[0]["D05"][0] - seeds[1]["D05"][0]) > 0.020 or abs(seeds[0]["D10"][0] - seeds[1]["D10"][0]) > 0.020:
        return "INCONCLUSIVE", "the two seeds differ by more than 0.020 in D05 or D10"
    if all(gain) and sum(gfail) == 1:
        return "INCONCLUSIVE", "a gain on both seeds but a guard fails on exactly one seed"
    for m in ("D05", "D10"):
        d = [sd[m][0] for sd in seeds]
        if any(x <= -BAND for x in d) and not any(x >= BAND for x in d):
            return "CUT", f"{m} <= -0.010 on a seed and the other seed does not reach +0.010"
    if all(gfail):
        return "CUT", "a guard fails on both seeds"
    if all(met) and not any(gfail):
        return "KEEP-quality", "the bar on both seeds, the guards hold"
    if all(gain) and not any(gfail):
        return "PARTIAL", "a real gain (D05 >= 0.010, D10 >= 0, intervals above 0) below the bar: the owner decides"
    why = ["no effect established (inside the band, or the intervals are not above 0)"]
    if any(sd["D10"][0] < 0 for sd in seeds):
        why.append("D10 is negative on a seed")
    if any(gfail):
        why.append("a guard fails on a seed")
    return "TIE", "; ".join(why)


def step3_verdict(seeds):
    """the soft-neighbour arm p2s against p2 (section 1.5): per seed D05 / D10 = M(p2s) - M(p2) with paired intervals and the guards against r2; KEEP-quality when both seeds have D05 >= 0.010 with the interval above 0 and D10 >= 0 and no guard fails;
    one such seed or a guard failing on exactly one seed = INCONCLUSIVE; else TIE (p2 stands)"""
    LIM = GUARDS
    nums = [x for sd in seeds for k in ("D05", "D10") for x in sd[k]] + [sd["guards"][k] for sd in seeds for k in LIM]
    if any(sd.get("kill_cut") for sd in seeds):
        return "CUT", "a CUT kill (the refined nodes are worse than the unmoved ones, or the offsets saturate)"
    if not all(np.isfinite(nums)):
        return "INCONCLUSIVE", "a non-finite number in the inputs"
    if any(sd.get("kill_inc") for sd in seeds):
        return "INCONCLUSIVE", "a kill fired (loss not finite / no learning / not finished / not parity / tests)"
    gain = [sd["D05"][0] >= BAND and sd["D05"][1] > 0 and sd["D10"][0] >= 0 for sd in seeds]
    gfail = [any(sd["guards"][k] < LIM[k] for k in LIM) for sd in seeds]
    if all(gain) and not any(gfail):
        return "KEEP-quality", "soft neighbours beat p2 on both seeds, the guards hold"
    if sum(gain) == 1 or (all(gain) and sum(gfail) == 1):
        return "INCONCLUSIVE", "a gain on one seed only, or a guard fails on exactly one seed"
    return "TIE", "no gain over p2 established: the simpler p2 stands"


def read_card(path):
    return json.load(open(path))


def p3(p):
    return (p["diff"], p["lo"], p["hi"])


KILL_NAMES = {"no_learning": "inc", "nonfinite": "inc", "notparity": "inc", "notfinished": "inc", "tests": "inc", "saturate": "cut"}
STEP1_DUMPS = {"c": "e392_c_s{s}", "r2": "e392_r2_s{s}", "s4": "e372_s4_s{s}"}


def verdict_cmd(kind, arm, control, seeds=(0, 1), flags=None):
    """`verdict step1 --arm all` reads runs/eval/e395_step1/<run>.json of c, r2 and the S4 live dump (both seeds): per dump and connector the class and the no-cost flag; the verdict of a connector = the S4 class, 'not consistent' when c or r2 disagree, 'no cost'
    only when every dump is free; prints the headroom H05 / H10 and the bars B05 / B10 per seed and the OFULL + H sanity. `verdict step2 --arm <p0|p0n|p2> --control <c|r2>` reads runs/eval/e395_step2/e395_<arm>_s<S>.json, the headroom from the control's step-1
    cards and the nodeerr cards (the refined-worse kill: mean |lateral error| after above the one before by more than 0.01 m); `verdict step3 --arm p2s --control p2` (the cards of p2s with --control r2 --versus p2); `verdict diag`: the point differences p0 - p0n, p2 - p0
    and the interaction (p2 - r2) - (p0 - c). The flags (--flags 0:no_learning,1:saturate ...) carry the kills that only the training log shows; names are asserted (KILL_NAMES)"""
    flags = flags or {}
    assert set(flags) <= set(seeds), f"--flags names seeds {sorted(set(flags) - set(seeds))} that are not read"
    for sd_, names in flags.items():
        bad = [n for n in names if n not in KILL_NAMES]
        assert not bad, f"unknown kill flag(s) {bad}: {sorted(KILL_NAMES)}"
    d1 = os.path.join(EVAL, "e395_step1")
    if kind == "step1":
        cards = {d: [read_card(os.path.join(d1, STEP1_DUMPS[d].format(s=s) + ".json")) for s in seeds] for d in STEP1_DUMPS}
        for d in ("c", "r2"):
            for s, c in zip(seeds, cards[d]):
                m = c["modes"]
                h05, h10 = m["OLAT+H"]["F1@0.5"] - m["H"]["F1@0.5"], m["OLAT+H"]["F1@1.0"] - m["H"]["F1@1.0"]
                ok = m["OFULL+H"]["F1@0.5"] >= m["OLAT+H"]["F1@0.5"] - 0.010 and m["OFULL+H"]["F1@1.0"] >= m["OLAT+H"]["F1@1.0"] - 0.010
                print(f"headroom {d} seed {s}: H05 {h05:.4f} (bar B05 {bar(h05):.4f}), H10 {h10:.4f} (bar B10 {bar(h10):.4f}); OFULL+H sanity {'ok' if ok else 'FAILED'} (F1@0.5 {m['OFULL+H']['F1@0.5']:.4f}, OLAT+H {m['OLAT+H']['F1@0.5']:.4f})")
                assert ok, f"{d} seed {s}: OFULL + H reads below OLAT + H by more than 0.010"
        out = {}
        for mode in ("P", "CR", "CRC", "CRS"):
            per = {}
            for d in STEP1_DUMPS:
                seeds_d = [{k: p3(c["modes"][mode]["paired_vs_H"][k]) for k in ("F1@0.5", "F1@1.0", "topo_F1")} for c in cards[d]]
                per[d] = step1_verdict(seeds_d)
                print(f"step1 dump {d} {mode}: {per[d][0]}" + (" (free)" if per[d][1] else "") + "   " + "; ".join(f"seed {s}: " + " ".join(f"{k} {x[k][0]:+.4f}" for k in x) for s, x in zip(seeds, seeds_d)))
            consistent = per["c"][0] == per["s4"][0] and per["r2"][0] == per["s4"][0]
            free = all(v[1] for v in per.values())
            out[mode] = (per["s4"][0], consistent, free)
            print(f"STEP1 {mode}: {per['s4'][0]} on S4" + ("" if consistent else " (NOT CONSISTENT with c / r2)") + (", replaces H at no cost" if free else ""))
        return out
    e2, ne_dir = os.path.join(EVAL, "e395_step2"), os.path.join(EVAL, "e395_nodeerr")
    if kind == "diag":
        for s in seeds:
            cd = {a: read_card(os.path.join(e2, f"e395_{a}_s{s}.json")) for a in ("p0", "p0n", "p2")}
            mv = {a: {m: cd[a]["paired_M_vs_U"][m]["diff"] for m in ("F1@0.5", "F1@1.0")} for a in cd}
            tot = {a: {m: cd[a]["modes"]["M"][m] for m in ("F1@0.5", "F1@1.0")} for a in cd}
            ctl = {a: {m: cd[a]["control"][m] for m in ("F1@0.5", "F1@1.0")} for a in ("p0", "p2")}
            for m in ("F1@0.5", "F1@1.0"):
                print(f"seed {s} {m}: move effect D(p0) - D(p0n) {mv['p0'][m] - mv['p0n'][m]:+.4f} (the step-3 gate (a) reads F1@0.5 >= +0.005 on both seeds); D(p2) - D(p0) {mv['p2'][m] - mv['p0'][m]:+.4f}; total p0 - p0n {tot['p0'][m] - tot['p0n'][m]:+.4f}; "
                      f"interaction (p2 - r2) - (p0 - c) on the totals {(tot['p2'][m] - ctl['p2'][m]) - (tot['p0'][m] - ctl['p0'][m]):+.4f}")
        return None
    sd = []
    for s in seeds:
        c = read_card(os.path.join(e2, f"e395_{arm}_s{s}.json"))
        fl = flags.get(s, ())
        ne = read_card(os.path.join(ne_dir, f"e395_{arm}_s{s}.json"))
        kc = ne["by"]["M"]["lateral_abs_mean"] > ne["by"]["U"]["lateral_abs_mean"] + 0.010 or any(KILL_NAMES[n] == "cut" for n in fl)
        ki = any(KILL_NAMES[n] == "inc" for n in fl)
        if kind == "step3":
            assert c["control"]["run"] == f"e392_r2_s{s}" and c["versus"]["run"] == f"e395_{control}_s{s}", f"seed {s}: the card's control / versus are not e392_r2 / e395_{control}"
            pm, pc = c["paired_M_vs_versusM"], c["paired_M_vs_control"]
            sd.append(dict(D05=p3(pm["F1@0.5"]), D10=p3(pm["F1@1.0"]), guards={k: pc[k]["diff"] for k in GUARDS}, kill_cut=kc, kill_inc=ki))
            print(f"seed {s}: M({arm}) - M({control}): D05 {sd[-1]['D05'][0]:+.4f} [{sd[-1]['D05'][1]:+.4f}, {sd[-1]['D05'][2]:+.4f}], D10 {sd[-1]['D10'][0]:+.4f} [{sd[-1]['D10'][1]:+.4f}, {sd[-1]['D10'][2]:+.4f}]; guards " + " ".join(f"{k} {v:+.4f}" for k, v in sd[-1]["guards"].items()))
            continue
        assert c["control"]["run"] == f"e392_{control}_s{s}", f"seed {s}: the step-2 card's control is {c['control']['run']}, not e392_{control}_s{s} (the headroom and the guards must share the control)"
        ctl = read_card(os.path.join(d1, f"e392_{control}_s{s}.json"))["modes"]
        h05, h10 = ctl["OLAT+H"]["F1@0.5"] - ctl["H"]["F1@0.5"], ctl["OLAT+H"]["F1@1.0"] - ctl["H"]["F1@1.0"]
        pm, pc = c["paired_M_vs_U"], c["paired_M_vs_control"]
        sd.append(dict(D05=p3(pm["F1@0.5"]), D10=p3(pm["F1@1.0"]), B05=bar(h05), B10=bar(h10), guards={k: pc[k]["diff"] for k in GUARDS}, kill_cut=kc, kill_inc=ki))
        print(f"seed {s}: D05 {sd[-1]['D05'][0]:+.4f} [{sd[-1]['D05'][1]:+.4f}, {sd[-1]['D05'][2]:+.4f}] (bar {sd[-1]['B05']:.4f}, H05 {h05:.4f}), D10 {sd[-1]['D10'][0]:+.4f} [{sd[-1]['D10'][1]:+.4f}, {sd[-1]['D10'][2]:+.4f}] (bar {sd[-1]['B10']:.4f}, H10 {h10:.4f}); guards "
              + " ".join(f"{k} {v:+.4f}" for k, v in sd[-1]["guards"].items()) + f"; total M - control F1@0.5 {pc['F1@0.5']['diff']:+.4f}, F1@1.0 {pc['F1@1.0']['diff']:+.4f}; mean |lateral error| U {ne['by']['U']['lateral_abs_mean']:.3f} -> M {ne['by']['M']['lateral_abs_mean']:.3f} m")
    v = step3_verdict(sd) if kind == "step3" else step2_verdict(sd)
    print(f"VERDICT {arm} against {control}: {v[0]} ({v[1]})")
    return v


# ------------------------------------------------------------------------------------------------------------------------------ run.py launch (hard rule 3)
def launch(step, arm, seed, run, extra, dry=False, again=False):
    """a scoring step of the card through run.py with the card's words (as `jarm.py`): id E395.<step>_<arm>_s<S>, command `jcr_eval.py <cmd> --run <run> --seed S <extra>`; the ids must be listed in the card;
    `again` = run.py's --again, a deliberate repeat of an id that finished with exit 0 (the old record stays in the ledger notes; used when a kill flag from the training log was missing in the first run)"""
    import subprocess
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import jarm
    rid = f"E395.verdict_{arm}" if step == "verdict" else f"E395.{step}_{arm}_s{seed}"
    assert rid in jarm.card_run_ids("E395"), f"{rid} is not in the card's run_ids"
    cmd = {"step1": "step1", "score": "step2", "nodeerr": "nodeerr", "verdict": "verdict"}[step]
    h, p, r = jarm.card_texts("E395")
    what = extra if step == "verdict" else f"--run {run} --seed {seed} {extra}"                  # verdict: `step1 --arm c` / `step2 --arm p0 --control c [--flags ...]`
    inner = f"env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh evaluation/jcr_eval.py {cmd} {what}"                                  # the folder-qualified name: _py14.sh refuses a basename that exists twice (a scratch copy under temp/)
    line = ["python3", os.path.join(jarm.AR, "run.py"), "--id", rid, "--seed", str(seed), "--require-card", "--split", "val", "--timeout", "7200"] + (["--again"] if again else []) + ["--hyp", h, "--pred", p, "--rule", r, "--", "bash", "-c", inner]
    print(rid, "->", inner, flush=True)
    return 0 if dry else subprocess.call(line, cwd=ROOT, env=dict(os.environ, BFLC_SIG="1.0"))


# ------------------------------------------------------------------------------------------------------------------------------ tests
def selftest_fixture():
    """`verdict_cmd` wiring on fixture cards in a temporary folder: the headroom from the control's step-1 card, the paired move effect, the guards, the nodeerr kill, the control assert; `check_record_card` on a fixture run folder"""
    import contextlib
    import io
    import tempfile
    global EVAL
    old_eval, old_runs = EVAL, JE.RUNS
    with tempfile.TemporaryDirectory() as tmp:
        EVAL = os.path.join(tmp, "eval")
        try:
            for sub_ in ("e395_step1", "e395_step2", "e395_nodeerr"):
                os.makedirs(os.path.join(EVAL, sub_))
            tri = lambda d: dict(diff=d, lo=d - 0.003, hi=d + 0.003)  # noqa: E731
            for sd in (0, 1):
                json.dump(dict(modes={"H": {"F1@0.5": 0.648, "F1@1.0": 0.846}, "OLAT+H": {"F1@0.5": 0.781, "F1@1.0": 0.879}, "OFULL+H": {"F1@0.5": 0.785, "F1@1.0": 0.880}}), open(os.path.join(EVAL, "e395_step1", f"e392_c_s{sd}.json"), "w"))
                json.dump(dict(control=dict(run=f"e392_c_s{sd}", **{"F1@0.5": 0.648, "F1@1.0": 0.846}), modes={"M": {"F1@0.5": 0.69, "F1@1.0": 0.86}}, paired_M_vs_U={"F1@0.5": tri(0.04), "F1@1.0": tri(0.012)},
                               paired_M_vs_control={"topo_F1": tri(0.002), "lanes_found": tri(0.0), "junction_F1": tri(0.0), "F1@1.0": tri(0.01), "F1@0.5": tri(0.04)}), open(os.path.join(EVAL, "e395_step2", f"e395_p0_s{sd}.json"), "w"))
                json.dump(dict(by={"U": {"lateral_abs_mean": 0.30}, "M": {"lateral_abs_mean": 0.26}}), open(os.path.join(EVAL, "e395_nodeerr", f"e395_p0_s{sd}.json"), "w"))
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                v = verdict_cmd("step2", "p0", "c")
            assert v[0] == "KEEP-quality" and "bar 0.0333" in buf.getvalue(), (v, buf.getvalue())
            with contextlib.redirect_stdout(io.StringIO()):
                assert verdict_cmd("step2", "p0", "c", flags={1: ["saturate"]})[0] == "CUT" and verdict_cmd("step2", "p0", "c", flags={0: ["no_learning"]})[0] == "INCONCLUSIVE"
            for sd in (0, 1):                                                                # the refined-worse kill: the mean |lateral error| after above the one before by more than 0.01 m
                json.dump(dict(by={"U": {"lateral_abs_mean": 0.30}, "M": {"lateral_abs_mean": 0.32}}), open(os.path.join(EVAL, "e395_nodeerr", f"e395_p0_s{sd}.json"), "w"))
            with contextlib.redirect_stdout(io.StringIO()):
                assert verdict_cmd("step2", "p0", "c")[0] == "CUT"
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    verdict_cmd("step2", "p0", "r2")
                raise SystemExit("the control assert did not fire")
            except (AssertionError, FileNotFoundError):
                pass
            JE.RUNS = os.path.join(tmp, "runs")                                              # check_record_card: the stored D115 card of the run, 1e-4
            os.makedirs(os.path.join(JE.RUNS, "fix_run"))
            ref = {"topo_F1": 0.6679, "F1@0.5": 0.6477, "F1@1.0": 0.8459, "lanes_found": 0.7191, "junction_F1": 0.2517}
            json.dump(dict(metrics=ref), open(os.path.join(JE.RUNS, "fix_run", "token_card_rec_lt0.3_ltl0.2_kt0.6_nng.json"), "w"))
            check_record_card("fix_run", dict(ref, topo_F1=0.66795), False)
            try:
                check_record_card("fix_run", dict(ref, topo_F1=0.6685), False)
                raise SystemExit("check_record_card accepted a difference above 1e-4")
            except AssertionError:
                pass
        finally:
            EVAL, JE.RUNS = old_eval, old_runs


def selftest():
    # the Catmull-Rom segment: through its end nodes, a line through collinear nodes, the tangent (P2 - P0) / 2
    P = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    C = cr_points(P[0], P[1], P[2], P[3], 21)
    assert np.allclose(C[0], P[1]) and np.allclose(C[-1], P[2]) and np.abs(C[:, 1]).max() < 1e-12 and np.all(np.diff(C[:, 0]) > 0)
    Q = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 4.0], [30.0, 4.0]])
    h = 1e-6
    seg = lambda t: 0.5 * (2 * Q[1] + (-Q[0] + Q[2]) * t + (2 * Q[0] - 5 * Q[1] + 4 * Q[2] - Q[3]) * t ** 2 + (-Q[0] + 3 * Q[1] - 3 * Q[2] + Q[3]) * t ** 3)
    assert np.allclose((seg(h) - seg(0.0)) / h, (Q[2] - Q[0]) / 2, atol=1e-4) and np.allclose((seg(1.0) - seg(1.0 - h)) / h, (Q[3] - Q[1]) / 2, atol=1e-4)
    # the ghost rule by hand: nodes A B C and two successors of C, the straighter one wins, the reflection fills a missing end
    pos = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 5.0], [30.0, -20.0]])
    src, dst = np.array([0, 1, 2, 2]), np.array([1, 2, 3, 4])
    preds, succs = {1: [0], 2: [1], 3: [2], 4: [2]}, {0: [1], 1: [2], 2: [3, 4]}
    p0, p3 = ghosts(pos, src, dst, preds, succs, 1)
    assert np.allclose(p0, pos[0]) and np.allclose(p3, pos[3]), (p0, p3)                      # B -> C: the predecessor A, the successor D (26.6 degrees, E is 63.4)
    p0, p3 = ghosts(pos, src, dst, preds, succs, 0)
    assert np.allclose(p0, [-10.0, 0.0]) and np.allclose(p3, pos[2]), (p0, p3)                 # A -> B: no predecessor: reflection 2 A - B; successor of B = C
    assert np.allclose(ghosts(pos, src, dst, preds, succs, 3)[1], 2 * pos[4] - pos[2])        # C -> E: E has no successor: the reflection
    # curves_of: first point the source node, last the destination, sampling about 0.5 m, chords and Catmull-Rom agree on collinear nodes
    pred = dict(nodes=dict(pos=pos), links=dict(src=src, dst=dst, long=np.array([False, False, False, True])))
    assert [c is None for c in curves_of(pred, "CRS")] == [False, False, False, True], "CRS leaves the long links to the Hermite connector"
    for kind in ("CR", "P"):
        cs = curves_of(pred, kind)
        assert len(cs) == 4 and all(np.allclose(c[0], pos[s]) and np.allclose(c[-1], pos[d]) for c, s, d in zip(cs, src, dst)) and len(cs[0]) == 21, kind
    line = dict(nodes=dict(pos=P), links=dict(src=np.array([0, 1, 2]), dst=np.array([1, 2, 3]), long=np.zeros(3, bool)))
    assert all(np.allclose(a_, b_, atol=1e-9) for a_, b_ in zip(curves_of(line, "CR"), curves_of(line, "P")))
    # the oracle: OLAT moves along the normal only, OFULL onto the GT node, a token without a GT node stays
    tok = dict(pos=np.array([[0.0, 0.3], [5.0, 1.0], [9.0, 9.0]]), dirv=np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]))
    gt_pos = np.array([[0.4, 0.0], [5.5, 0.0]])
    lat = move_tokens(tok, np.array([0, 1, -1]), gt_pos, "OLAT")
    assert np.allclose(lat["pos"], [[0.0, 0.0], [5.0, 0.0], [9.0, 9.0]]) and np.allclose(tok["pos"][0], [0.0, 0.3]), lat["pos"]      # x is kept, y goes to the GT node's line, the input is not modified
    full = move_tokens(tok, np.array([0, 1, -1]), gt_pos, "OFULL")
    assert np.allclose(full["pos"], [[0.4, 0.0], [5.5, 0.0], [9.0, 9.0]])
    assert split_mode("H") == ("", "H") and split_mode("OLAT+CR") == ("OLAT", "CR")
    # the scorer takes the curves: the oracle prediction with its Hermite curves given explicitly scores exactly as with `curves` None; a bent curve changes the number
    import qformat
    g, old = QE.toy_lane_gt()
    pred = qformat.oracle_prediction(g)
    pred["links"]["curves"] = None
    m0 = QE.graph_metrics(QE.readout(pred)[0], old)
    nd, lk = pred["nodes"], pred["links"]
    pred["links"]["curves"] = [QT.hermite(nd["pos"][s], nd["dirv"][s], nd["pos"][d], nd["dirv"][d], float(lk["a"][k]), float(lk["b"][k]), QE.STEP) for k, (s, d) in enumerate(zip(lk["src"], lk["dst"]))]
    m1 = QE.graph_metrics(QE.readout(pred)[0], old)
    assert all(abs(m0[k] - m1[k]) < 1e-12 for k in ("F1@1.0", "topo_F1", "lanes_found", "junction_F1")), (m0, m1)
    pred["links"]["curves"] = [c + np.array([0.0, 2.5]) if len(c) > 3 else c for c in pred["links"]["curves"]]
    assert QE.graph_metrics(QE.readout(pred)[0], old)["F1@1.0"] < m0["F1@1.0"]
    # Catmull-Rom curves from the toy prediction score close to the Hermite ones (smooth GT)
    pred["links"]["curves"] = curves_of(pred, "CR")
    mc = QE.graph_metrics(QE.readout(pred)[0], old)
    assert mc["F1@1.0"] < m0["F1@1.0"], "the one long link of the toy (36 m, two nodes) has only reflected ghosts: Catmull-Rom is worse than the Hermite connector there"
    pred["links"]["curves"] = curves_of(pred, "CRS")
    mcs = QE.graph_metrics(QE.readout(pred)[0], old)
    assert mcs["F1@1.0"] > 0.99 and mcs["topo_F1"] > 0.99, mcs                                     # the short links are dense and collinear: Catmull-Rom is exact there, the long one is the Hermite curve
    # centripetal Catmull-Rom: through its end nodes, a line through collinear nodes, equal to the uniform one for equal chords
    C2 = cr_points_centripetal(P[0], P[1], P[2], P[3], 21)
    assert np.allclose(C2[0], P[1]) and np.allclose(C2[-1], P[2]) and np.abs(C2[:, 1]).max() < 1e-12
    ang = np.array([0.0, 0.2, 0.4, 0.6])
    Qs = 10.0 * np.stack([np.cos(ang), np.sin(ang)], 1)                                           # four points of a circle at equal angles: equal chords
    assert np.abs(cr_points_centripetal(*Qs, 9) - cr_points(*Qs, 9)).max() < 1e-9, "equal chords: centripetal = uniform"
    # the refined nodes: dpos along the normal of dirv only
    tk = dict(pos=np.array([[1.0, 2.0], [0.0, 0.0]]), dirv=np.array([[0.0, 1.0], [1.0, 0.0]]))
    mv = move_by_dpos(tk, np.array([0.5, -0.25]))
    assert np.allclose(mv["pos"], [[0.5, 2.0], [0.0, -0.25]]) and np.allclose(tk["pos"][0], [1.0, 2.0]) and np.allclose(normals(tk["dirv"]), [[-1.0, 0.0], [0.0, 1.0]])
    # the verdict table (card section 1.3): the cases of the critic and the order
    P_ = lambda d, w=0.003: (d, d - w, d + w)  # noqa: E731
    G0 = {"topo_F1": 0.0, "lanes_found": 0.0, "junction_F1": 0.0, "F1@1.0": 0.0}
    mk = lambda d05, d10, g=G0, b05=0.033, b10=0.010, kc=False, ki=False: dict(D05=P_(d05), D10=P_(d10), B05=b05, B10=b10, guards=g, kill_cut=kc, kill_inc=ki)  # noqa: E731
    assert step2_verdict([mk(0.04, 0.012), mk(0.05, 0.015)])[0] == "KEEP-quality"
    assert step2_verdict([mk(0.04, 0.04), mk(0.04, 0.00)])[0] == "INCONCLUSIVE", "D10 +0.04 on one seed, 0.00 on the other: the bar is met on one seed only"
    assert step2_verdict([mk(0.04, 0.012), mk(0.05, 0.015, dict(G0, topo_F1=-0.02))])[0] == "INCONCLUSIVE", "the bar on both seeds, a guard fails on one"
    assert step2_verdict([mk(0.04, 0.012, dict(G0, topo_F1=-0.02)), mk(0.05, 0.015, dict(G0, lanes_found=-0.02))])[0] == "CUT", "a guard (any) fails on both seeds"
    assert step2_verdict([mk(0.0, -0.012), mk(0.0, 0.0)])[0] == "CUT"
    assert step2_verdict([mk(0.0, -0.012), mk(0.0, 0.015)])[0] == "INCONCLUSIVE", "+0.015 / -0.012 on D10: the seeds differ by more than 0.020"
    assert step2_verdict([mk(0.02, 0.004), mk(0.02, 0.003)])[0] == "PARTIAL", "a real gain (D05 0.02, intervals above 0) below the bar"
    assert step2_verdict([mk(0.004, 0.004), mk(0.003, 0.003)])[0] == "TIE"
    assert step2_verdict([mk(0.04, 0.012, kc=True), mk(0.05, 0.015)])[0] == "CUT" and step2_verdict([mk(0.04, 0.012, ki=True), mk(0.05, 0.015)])[0] == "INCONCLUSIVE"
    assert step2_verdict([mk(float("nan"), 0.012), mk(0.05, 0.015)])[0] == "INCONCLUSIVE", "a NaN input is never a verdict"
    assert step2_verdict([mk(0.02, 0.004, dict(G0, junction_F1=-0.02)), mk(0.02, 0.003)])[0] == "INCONCLUSIVE", "a gain below the bar with a guard failing on exactly one seed"
    assert step2_verdict([mk(-0.05, 0.0), mk(-0.05, 0.02)])[0] == "CUT", "D05 -0.05 on both seeds is a CUT even when D10 is not negative"
    assert step2_verdict([mk(0.05, 0.02), (lambda d: dict(d, D05=(0.05, -0.01, 0.11)))(mk(0.05, 0.02))])[0] == "INCONCLUSIVE", "the bar is met on one seed only (the other interval crosses 0)"
    assert step2_verdict([(lambda d: dict(d, D05=(0.05, -0.01, 0.11)))(mk(0.05, 0.02))] * 2)[0] == "TIE", "a large point estimate with intervals crossing 0: no effect established"
    assert step3_verdict([mk(0.012, 0.0), mk(0.015, 0.002)])[0] == "KEEP-quality" and step3_verdict([mk(0.012, 0.0), mk(0.004, 0.0)])[0] == "INCONCLUSIVE" and step3_verdict([mk(0.004, 0.0), mk(0.003, 0.0)])[0] == "TIE"
    for bad in ("no-learning", "nonfinite_"):
        try:
            verdict_cmd("step2", "p0", "c", flags={0: [bad]})
            raise SystemExit("an unknown kill flag was accepted")
        except AssertionError:
            pass
    assert step2_verdict([mk(0.04, 0.012, dict(G0, **{"F1@1.0": -0.01})), mk(0.05, 0.015)])[0] == "INCONCLUSIVE", "the total F1@1.0 guard: a gain with the total F1@1.0 more than 0.005 below the control on one seed"
    assert step3_verdict([mk(0.012, 0.0, kc=True), mk(0.015, 0.002)])[0] == "CUT" and step3_verdict([mk(0.012, 0.0, ki=True), mk(0.015, 0.002)])[0] == "INCONCLUSIVE"
    assert "D10 is negative" in step2_verdict([mk(0.05, -0.009), mk(0.05, -0.009)])[1] and "guard" in step2_verdict([mk(0.004, 0.0, dict(G0, topo_F1=-0.03)), mk(0.003, 0.0)])[1]
    try:
        verdict_cmd("step2", "p0", "c", flags={2: ["saturate"]})
        raise SystemExit("a flag for a seed that is not read was accepted")
    except AssertionError:
        pass
    selftest_fixture()
    assert bar(0.133) == 0.25 * 0.133 and bar(0.033) == 0.010
    cl = lambda a, b, c: dict({"F1@0.5": P_(a), "F1@1.0": P_(b), "topo_F1": P_(c)})  # noqa: E731
    assert step1_verdict([cl(0.0, 0.0, 0.0), cl(0.001, 0.0, 0.0)]) == ("TIE", True)
    assert step1_verdict([cl(-0.02, -0.02, -0.03), cl(-0.01, -0.025, -0.03)])[0] == "LOSS"
    assert step1_verdict([cl(-0.02, -0.02, -0.03), cl(0.0, 0.0, 0.0)])[0] == "INCONCLUSIVE" and step1_verdict([cl(0.02, 0.02, 0.0), cl(0.0, 0.0, 0.0)])[0] == "TIE"
    assert step1_verdict([cl(0.02, 0.02, 0.0), cl(0.02, 0.03, 0.0)])[0] == "GAIN" and step1_verdict([cl(0.0, 0.0, -0.0065), cl(0.0, 0.0, -0.0065)]) == ("TIE", False)
    assert step1_verdict([cl(float("nan"), 0.0, 0.0), cl(0.0, 0.0, 0.0)])[0] == "INCONCLUSIVE"
    print("jcr_eval selftest ok")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("step1")
    s1.add_argument("--run", required=True)
    s1.add_argument("--seed", type=int, required=True)
    s1.add_argument("--live", action="store_true")
    s1.add_argument("--modes", default="H,P,CR")
    s1.add_argument("--out", default=os.path.join(ROOT, "runs", "eval", "e395_step1"))
    s2 = sub.add_parser("step2")
    s2.add_argument("--run", required=True)
    s2.add_argument("--seed", type=int, required=True)
    s2.add_argument("--control")
    s2.add_argument("--versus", help="another position arm's run: its dump is scored as M and paired against this run's M (step 3: p2s against p2)")
    s2.add_argument("--out", default=os.path.join(ROOT, "runs", "eval", "e395_step2"))
    ne = sub.add_parser("nodeerr")
    ne.add_argument("--run", required=True)
    ne.add_argument("--seed", type=int, required=True)
    ne.add_argument("--out", default=os.path.join(ROOT, "runs", "eval", "e395_nodeerr"))
    vd = sub.add_parser("verdict")
    vd.add_argument("kind", choices=("step1", "step2", "step3", "diag"))
    vd.add_argument("--arm", default="")
    vd.add_argument("--control", default="c")
    vd.add_argument("--flags", default="", help="kills the training log shows, e.g. 0:no_learning,1:saturate")
    ln = sub.add_parser("launch")
    ln.add_argument("step", choices=("step1", "score", "nodeerr", "verdict"))
    ln.add_argument("arm")
    ln.add_argument("--run", default="")
    ln.add_argument("--seed", type=int, default=0)
    ln.add_argument("--extra", default="")
    ln.add_argument("--dry", action="store_true")
    ln.add_argument("--again", action="store_true", help="run.py --again: a deliberate repeat of an id that finished with exit 0")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    ab = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)  # noqa: E731                      # a relative --out is relative to the problem folder, not to this file
    if a.cmd == "verdict":
        fl = {}
        for it in [x for x in a.flags.split(",") if x]:
            sd_, _, f_ = it.partition(":")
            fl.setdefault(int(sd_), []).append(f_)
        return verdict_cmd(a.kind, a.arm, a.control, (0, 1), fl)
    if a.cmd == "launch":
        return launch(a.step, a.arm, a.seed, a.run, a.extra, a.dry, a.again)
    if a.cmd == "step2":
        return step2(a.run, a.seed, a.control, ab(a.out), versus=a.versus)
    if a.cmd == "nodeerr":
        return nodeerr(a.run, a.seed, ab(a.out))
    return step1(a.run, a.seed, a.live, [m.strip() for m in a.modes.split(",")], ab(a.out))


if __name__ == "__main__":
    main()
