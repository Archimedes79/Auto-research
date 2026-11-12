"""Read-out and scoring of the junction end head (E284), on the delivered graph of the lane head (baseline B + the E275 hard close-node step).

    wsl bash tools/env/_py14.sh qends_eval.py dump  --run e284_diag_s0                       # GPU or CPU: the end head on the 873 VAL frames -> runs/<run>/ends_val.pkl + the head's own numbers
    wsl bash tools/env/_py14.sh qends_eval.py score --run e284_diag_s0 --seed 0 [--thr 0.5] [--snap [--snap-all] [--snap-guard] [--snap-dedup]]  # CPU: the delivered graph with the head's connectors, scored by the E273 scorer; D79's rule against the plain delivered graph
    wsl bash tools/env/_py14.sh qends_eval.py selftest

What is delivered: the graph of B read out as before (`qgraph_readout` close:1.5@0.5, `qgraph_eval.readout`), and for every pair of the end head with probability >= thr the cubic Hermite curve of `qends.curve`
(tangent = chord) from the entry keypoint to the exit keypoint, wired to the delivered section pieces whose last point lies within 4 m (and 45 degrees) of the entry and whose first point lies within 4 m (and 45
degrees) of the exit (the wiring of the E283 oracles `temp/e284/p19..p22`, copied here). A delivered LONG-link piece is dropped when a curve covers it (a curve starts within 4 m of its first point and ends within 4 m
of its last point); the others stay (the 'keep' integration: a head with gaps does not remove the connectors it does not know). The stage trains on the frozen pixel / node / link stages of a B run
(`qgraph_train.py --phases 8 --resume-from`), so the delivered graph of the run IS B's: `score` reads it from B's full prediction dump of the same seed (`--full`, default `temp/e276/predfull_e273_p_s<seed>.pkl`,
written by `qgraph_run_eval.py score --dump-full`) and checks that the run's pixel / node / link tensors equal those of the B run (`--base`, default `runs/e273_p_s<seed>/model.pt`).
Rule (D79, applied to the run's own card of the experiment, never decided here): topo_F1 of the graph with the head's connectors against the plain delivered graph: KEEP > +0.02 on both seeds, guards F1@1.0 not lower
by more than 0.01, dup_frac not higher by more than 0.02, vertices per 100 m (DP) not higher by more than 1.0, junction F1 not lower by more than 0.03; lanes found, Q, topo P / R, junction P / R are reported.
`score` prints which conditions hold for THIS seed; the decision over both seeds is the card's.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import qends as QEND  # noqa: E402

KEYS = ("Q", "F1@1.0", "lanes_found", "topo_F1", "topo_P", "topo_R", "junction_F1", "junction_P", "junction_R", "dup_frac", "vert100_dp")
WIRE_POS, WIRE_ANG = 4.0, 45.0
COVER_POS = 4.0


def _tan(P, at_end):
    q = P[-3:] if at_end else P[:3]
    d = q[-1] - q[0]
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.array([0.0, 1.0])


def _ang(u, v):
    return float(np.degrees(np.arccos(np.clip(u @ v, -1.0, 1.0))))


def make_pairs(e, ia, ib, k, attach=False, chord=False, fit=None):
    """the `pairs` of `add_connectors` for the candidates (ia[n], ib[n]) of a dump frame `e` (`jends_eval.py dump`; k = keypoints per kind): (entry end, entry heading, exit end, exit heading) + the learned handle scales (pr[4], pr[5]) when the dump has them and not `chord`
    (`fit`: a dict {(i, j): scales} instead, the oracle with fitted scales of `jmaneuver.py`). attach (E320): the ATTACHED ends and headings of the dump (att_pos / att_dir) instead of the keypoints'; the one place where the read-outs (cmd_score, jmaneuver) choose between the two."""
    qp, qd = (e["att_pos"], e["att_dir"]) if attach else (e["kp_pos"], e["kp_dir"])
    if fit is not None:
        return [(qp[i], qd[i], qp[k + j], qd[k + j]) + fit.get((int(i), int(j)), ()) for i, j in zip(ia, ib)]
    return [(qp[i], qd[i], qp[k + j], qd[k + j]) + (tuple(e["scale"][i, j]) if e.get("scale") is not None and not chord else ()) for i, j in zip(ia, ib)]


def add_connectors(lines, edges, nodes, kinds, links, long_mask, pairs, snap=False, snap_all=False, snap_guard=False, snap_dedup=False, snap_nogeom=False, snap_both=False, wire_pos=None):
    """the delivered graph (lines, edges, nodes as of `qgraph_eval.readout`; kinds / links = its `bk["kind"]` / `bk["piece_links"]`; long_mask = which links are long) with the end head's connectors:
    pairs = [(entry xy, entry heading, exit xy, exit heading)] -> (raw graph dict for the scorer, curves added, curves wired at both ends).
    snap (E311): the connector is drawn from the end of the NEAREST delivered piece wired to the entry (position and direction of that piece end) to the start of the nearest piece wired to the exit, instead of from the keypoints (1.4 m off the GT
    ends on average: the curve then ends beside the pieces and the topology scores see a gap); an end with no piece in reach keeps its keypoint; only that one piece is wired at each end.
    Options of the snapped read-out (each only with snap, all default off; E311 refinements, the E311 verdict is the plain `snap`): snap_all = wire EVERY piece in reach of the snapped end (the default wiring loop on the snapped ends: separates the effect of the geometry from the
    effect of wiring one piece only), snap_guard = a pair whose snapped ends make a degenerate curve (chord under 1 m or a chord that runs against one of the two headings) keeps its keypoints, snap_dedup = a second curve between the same two snapped pieces is not added, snap_both = no curve for a pair whose entry or exit finds no piece in reach (no floating curves), snap_nogeom = a control: wire only the nearest piece per end but draw the curve from the keypoints (separates the wiring rule from the geometry)"""
    wp = WIRE_POS if wire_pos is None else float(wire_pos)
    cp_ = COVER_POS if wire_pos is None else min(COVER_POS, wp)                                                   # N1 (critic 2026-11-13): a long link is dropped only when a connector's ends are within the WIRING radius: under a tight radius a connector that is not wired to the pieces must not delete the long link of B either                                                     # E320: the radius (m) of the proximity wiring, default today's 4 m; the tight read-outs of the attached ends use 1.5 / 1.0
    keep = []
    for i, k in enumerate(kinds):
        if k == "link" and bool(long_mask[links[i][0]]):
            P = np.asarray(lines[i], float)
            if any(np.linalg.norm(P[0] - pr[0]) <= cp_ and np.linalg.norm(P[-1] - pr[2]) <= cp_ for pr in pairs):
                continue
        keep.append(i)
    new = {old: j for j, old in enumerate(keep)}
    L2 = [np.asarray(lines[i], float) for i in keep]
    N2 = [list(nodes[i]) for i in keep]
    E2 = [(new[a], new[b]) for a, b in edges if a in new and b in new]
    ends = [(P[-1], _tan(P, True), P[0], _tan(P, False)) for P in L2]
    wired = 0
    seen = set()
    for n, pr in enumerate(pairs):
        pa, ta, pb, tb = pr[:4]                                                                       # E303: a pair may carry the two handle scales (pr[4], pr[5])
        w_in = w_out = None
        default_wire = True                                                                                     # wire every piece in reach (the E284 / E287 way); the snapped read-out wires only the chosen piece per end
        if snap:
            default_wire = snap_all
            pa, ta, pb, tb = (np.asarray(x, float) for x in (pa, ta, pb, tb))
            d_in = d_out = np.inf
            for i, (le, te, fs, ts) in enumerate(ends):
                d = np.linalg.norm(le - pa)
                if d <= wp and _ang(te, ta) <= WIRE_ANG and d < d_in:
                    w_in, d_in = i, d
                d = np.linalg.norm(fs - pb)
                if d <= wp and _ang(ts, tb) <= WIRE_ANG and d < d_out:
                    w_out, d_out = i, d
            if snap_both and (w_in is None or w_out is None):
                continue
            if snap_dedup and w_in is not None and w_out is not None:
                if (w_in, w_out) in seen:
                    continue
                seen.add((w_in, w_out))
            pk = (pa, ta, pb, tb)
            if w_in is not None and not snap_nogeom:
                pa, ta = ends[w_in][0], ends[w_in][1]
            if w_out is not None and not snap_nogeom:
                pb, tb = ends[w_out][2], ends[w_out][3]
            if snap_guard:
                ch = pb - pa
                if np.linalg.norm(ch) < 1.0 or ch @ ta <= 0 or ch @ tb <= 0:
                    pa, ta, pb, tb = pk
                    w_in = w_out = None
                    default_wire = True
        m = len(L2)
        if len(pr) > 4:
            L2.append(QEND.curve_s(np.asarray(pa, float), np.asarray(ta, float), np.asarray(pb, float), np.asarray(tb, float), float(pr[4]), float(pr[5])))
        else:
            L2.append(QEND.curve(np.asarray(pa, float), np.asarray(ta, float), np.asarray(pb, float), np.asarray(tb, float), 1.0, 0.25))
        N2.append([10000 + n, 20000 + n])
        if not default_wire:
            if w_in is not None:
                E2.append((w_in, m))
            if w_out is not None:
                E2.append((m, w_out))
            wired += int(w_in is not None and w_out is not None)
            continue
        w_in = w_out = False
        for i, (le, te, fs, ts) in enumerate(ends):
            if np.linalg.norm(le - pa) <= wp and _ang(te, ta) <= WIRE_ANG:
                E2.append((i, m))
                w_in = True
            if np.linalg.norm(fs - pb) <= wp and _ang(ts, tb) <= WIRE_ANG:
                E2.append((m, i))
                w_out = True
        wired += int(w_in and w_out)
    return dict(pts=L2, edges=E2, nodes=N2), len(pairs), wired


def holds(num, base):
    g = lambda k: num[k] - base[k]  # noqa: E731
    cond = dict(topo_F1=g("topo_F1") > 0.02, F1_1=g("F1@1.0") >= -0.01, dup_frac=g("dup_frac") <= 0.02, vert100_dp=g("vert100_dp") <= 1.0, junction_F1=g("junction_F1") >= -0.03)
    return all(cond.values()), cond


def cmd_dump(a):
    import torch
    import lane_gt_data as LD
    import qendnet as QEN
    import qpredict as QP
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(ROOT, "runs", a.run)
    m = QP.load_model(run_dir, dev)
    assert m.ends is not None, f"{a.run} was trained without the end head (args['ends'])"
    LD.QData.DEFAULT_ENDS = True
    ds, dl = LD.loader("val_clean", bs=8, workers=2, shuffle=False, map_mode="error", pixels=False)
    assert ds.missing == 0, f"{ds.missing} VAL frames have no stored lane GT"
    frames, tot = [], {}
    with torch.no_grad():
        for b in dl:
            out = m.ends_forward(b, gt=False)
            for k, v in QEN.end_counts(out, b, a.thr).items():
                tot[k] = tot.get(k, 0) + v
            for i in range(len(b["token"])):
                frames.append(dict(token=b["token"][i], row=int(b["row"][i]), kp_pos=out["kp_pos"][i].cpu().numpy(), kp_dir=out["kp_dir"][i].cpu().numpy(), kp_p=out["kp_p"][i].cpu().numpy(),
                                   valid=out["valid"][i].cpu().numpy(), logit=out["logit"][i].float().cpu().numpy(), allowed=out["allowed"][i].cpu().numpy()))
    r = lambda x, y: x / max(y, 1)  # noqa: E731
    head = dict(ent_recall=r(tot["ent_hit"], tot["ent_n"]), ent_precision=r(tot["ent_ok"], tot["ent_kp"]), ext_recall=r(tot["ext_hit"], tot["ext_n"]), ext_precision=r(tot["ext_ok"], tot["ext_kp"]),
                pair_recall=r(tot["pairs_hit"], tot["pairs_n"]), pair_precision=r(tot["pairs_ok"], tot["pairs_kp"]), counts=tot, thr=a.thr, frames=len(frames))
    pickle.dump(frames, open(os.path.join(run_dir, "ends_val.pkl"), "wb"))
    json.dump(head, open(os.path.join(run_dir, "ends_head.json"), "w"), indent=1)
    print(json.dumps(head, indent=1))


def cmd_score(a):
    import torch
    import qgraph_eval as QE
    import qgraph_readout as QR
    run_dir = os.path.join(ROOT, "runs", a.run)
    base_pt = a.base or os.path.join(ROOT, "runs", f"e273_p_s{a.seed}", "model.pt")
    mine = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu")["state"]
    base = torch.load(base_pt, map_location="cpu")["state"]
    shared = [k for k in base if not k.startswith("ends.")]
    bad = [] if getattr(a, "skip_base_check", False) else [k for k in shared if k not in mine or not torch.equal(mine[k], base[k])]      # E287: a pair head on the junction maps run is not a QGraph checkpoint (jends_eval.py says so)
    assert not bad, f"the run changed tensors of the base run ({len(bad)}): {bad[:5]}; its delivered graph is not B's"
    full = a.full or os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{a.seed}.pkl")
    dump = pickle.load(open(full, "rb"))
    ends = {f["row"]: f for f in pickle.load(open(os.path.join(run_dir, "ends_val.pkl"), "rb"))}
    frames = QE.val_frames()
    assert [d["row"] for d in dump] == [f.row for f in frames], "the full dump is not in the order of the VAL frames"
    spec = QR.parse("close:1.5@0.5")
    rec = dict(base=[], head=[])
    attach = bool(getattr(a, "attach", False))
    attach_r = getattr(a, "attach_r", None)                                                                                  # E320: the wiring radius (m) of the attached read-out (default: today's 4 m); the card file name carries it
    assert attach_r is None or attach, "--attach-r needs --attach"
    assert not attach or ("att_pos" in next(iter(ends.values()))), f"{a.run}: the dump has no attached ends (train with --attach, then jends_eval.py dump)"
    n_curves = n_wired = n_drop = 0
    gap = dict(n=0, **{k_: 0 for k_ in ("in", "out", "kp_in", "kp_out")})
    t0 = time.time()
    for b, (f, d) in enumerate(zip(frames, dump)):
        pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                             links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
        graph, bk = QE.readout(pred, 0.5, 0.5)
        rec["base"].append(QE.graph_counts(graph, f))
        e = ends[f.row]
        k = e["kp_pos"].shape[0] // 2
        p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
        ia, ib = np.nonzero(p >= a.thr)
        pairs = make_pairs(e, ia, ib, k, attach=attach, chord=a.chord)                                                      # E320: attach = the attached ends of the dump instead of the keypoints
        le = np.array([np.asarray(P, float)[-1] for P in graph.lines if len(P)]).reshape(-1, 2)                              # critic 2026-11-12 (C3): do the connector ends touch a delivered piece end (an end of a line, within 1 m)?
        fs = np.array([np.asarray(P, float)[0] for P in graph.lines if len(P)]).reshape(-1, 2)
        near = lambda pts, q: bool(len(pts) and np.linalg.norm(pts - q, axis=1).min() <= 1.0)  # noqa: E731
        for i, j in zip(ia, ib):
            gap["n"] += 1
            gap["in"] += near(le, e["att_pos"][i] if attach else e["kp_pos"][i])
            gap["out"] += near(fs, e["att_pos"][k + j] if attach else e["kp_pos"][k + j])
            gap["kp_in"] += near(le, e["kp_pos"][i])
            gap["kp_out"] += near(fs, e["kp_pos"][k + j])
        raw, n_c, n_w = add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), pairs, snap=bool(getattr(a, "snap", False)), snap_all=bool(getattr(a, "snap_all", False)), snap_guard=bool(getattr(a, "snap_guard", False)), snap_dedup=bool(getattr(a, "snap_dedup", False)), snap_nogeom=bool(getattr(a, "snap_nogeom", False)), snap_both=bool(getattr(a, "snap_both", False)), wire_pos=attach_r)
        n_curves += n_c
        n_wired += n_w
        n_drop += len(graph.lines) - (len(raw["pts"]) - len(pairs))                                                          # B's long links that a connector covers and the read-out drops
        rec["head"].append(QE.graph_counts(raw, f))
        f._lane = f._ref = None
        if (b + 1) % 200 == 0:
            print(f"  scored {b + 1}/{len(frames)}  {time.time() - t0:.0f} s", flush=True)
    num = {v: QE.graph_numbers(rec[v]) for v in rec}
    ok, cond = holds(num["head"], num["base"])
    out = dict(run=a.run, seed=a.seed, thr=a.thr, frames=len(frames), curves=n_curves, wired_both_ends=n_wired, numbers={v: {k: float(num[v][k]) for k in KEYS if k in num[v]} for v in num},
               difference={k: float(num["head"][k] - num["base"][k]) for k in KEYS if k in num["head"]}, rule_this_seed=dict(holds=ok, conditions={k: bool(v) for k, v in cond.items()}),
               note="by the rule of D79 against the plain delivered graph of the same seed; the verdict over both seeds is the card's")
    snap = bool(getattr(a, "snap", False))
    out["snap"] = snap
    out["attach"] = attach
    out["attach_r"] = attach_r
    out["long_links_dropped"] = n_drop
    r_ = lambda x: float(x) / max(gap["n"], 1)  # noqa: E731
    out["ends_within_1m_of_a_delivered_piece_end"] = dict(pairs=gap["n"], entry=r_(gap["in"]), exit=r_(gap["out"]), keypoint_entry=r_(gap["kp_in"]), keypoint_exit=r_(gap["kp_out"]))      # of the ends the connectors use (attached ends under attach) and of the raw keypoints
    opts = [k for k in ("snap_all", "snap_guard", "snap_dedup", "snap_nogeom", "snap_both") if getattr(a, k, False)]
    out["snap_options"] = opts
    # the card of record (`ends_card.json`) is the unsnapped read-out at the threshold 0.5 only: another threshold or a snapped read-out gets its own file (the E308 incident of 2026-11-12: a scratch sweep emptied the card of record)
    chord = bool(getattr(a, "chord", False))
    name = "ends_card.json" if (not snap and not chord and not attach and a.thr == 0.5) else "ends_card_" + ("chord_" if chord else "") + ("snap" + "".join("_" + k[5:] for k in opts) + "_" if snap else "") + (("attach_" + (f"r{attach_r}_" if attach_r else "")) if attach else "") + f"thr{a.thr}.json"     # the chord control has its own file too (it overwrote the card of the learned scales on 2026-11-12)
    json.dump(out, open(os.path.join(run_dir, name), "w"), indent=1)
    pickle.dump(dict(scenes=[f.scene for f in frames], base=rec["base"], head=rec["head"]), open(os.path.join(run_dir, name[:-5] + ".counts.pkl"), "wb"))        # the per-frame counts: `jdiag.py boot` resamples them by scene (paired interval of two read-outs, E320)
    print(f"{'':16s}" + "".join(f"{k:>12s}" for k in KEYS))
    for v in ("base", "head"):
        print(f"{v:16s}" + "".join(f"{num[v][k]:12.4f}" for k in KEYS))
    print(f"{'difference':16s}" + "".join(f"{out['difference'][k]:+12.4f}" for k in KEYS))
    print(f"curves {n_curves} (wired at both ends {n_wired}); the rule of D79 holds for this seed: {ok} {cond}")


def selftest():
    # a delivered graph with a lane piece before the junction (0), a lane piece after it (1) and a delivered long link between them (2); the head delivers one connector over the same ends
    lines = [np.c_[np.zeros(10), np.linspace(-20, -2, 10)], np.c_[np.zeros(10) + 8, np.linspace(2, 20, 10)], np.c_[np.linspace(0, 8, 8), np.linspace(-2, 2, 8)]]
    kinds, links, long_mask = ["lane", "lane", "link"], [(0,), (1,), (0,)], np.array([True])
    pairs = [(np.array([0.0, -2.5]), np.array([0.0, 1.0]), np.array([8.0, 2.5]), np.array([0.0, 1.0]))]
    raw, n, w = add_connectors(lines, [(0, 2), (2, 1)], [[0], [1], [2]], kinds, links, long_mask, pairs)
    assert n == 1 and w == 1 and len(raw["pts"]) == 3, (n, w, len(raw["pts"]))                       # the covered long link is gone, the curve is wired at both ends
    assert (0, 2) in raw["edges"] and (2, 1) in raw["edges"], raw["edges"]
    far = [(np.array([30.0, 0.0]), np.array([0.0, 1.0]), np.array([40.0, 10.0]), np.array([0.0, 1.0]))]
    raw2, n2, w2 = add_connectors(lines, [(0, 2), (2, 1)], [[0], [1], [2]], kinds, links, long_mask, far)
    assert len(raw2["pts"]) == 4 and w2 == 0, (len(raw2["pts"]), w2)                                  # an unrelated curve covers nothing: the delivered long link stays
    off = [(np.array([1.2, -2.5]), np.array([0.0, 1.0]), np.array([8.9, 2.5]), np.array([0.0, 1.0]))]            # keypoints 1.2 / 0.9 m beside the piece ends
    raw3, n3, w3 = add_connectors(lines, [(0, 2), (2, 1)], [[0], [1], [2]], kinds, links, long_mask, off, snap=True)
    assert w3 == 1 and np.allclose(raw3["pts"][-1][0], lines[0][-1]) and np.allclose(raw3["pts"][-1][-1], lines[1][0]), (w3, raw3["pts"][-1][[0, -1]])      # the connector runs from the end of piece 0 to the start of piece 1
    assert (0, len(raw3["pts"]) - 1) in raw3["edges"] and (len(raw3["pts"]) - 1, 1) in raw3["edges"] and len([e for e in raw3["edges"] if len(raw3["pts"]) - 1 in e]) == 2, raw3["edges"]
    raw4, n4, w4 = add_connectors(lines, [(0, 2), (2, 1)], [[0], [1], [2]], kinds, links, long_mask, [(np.array([30.0, 0.0]), np.array([0.0, 1.0]), np.array([40.0, 10.0]), np.array([0.0, 1.0]))], snap=True)
    assert w4 == 0 and np.allclose(raw4["pts"][-1][0], [30.0, 0.0]), w4                                    # no piece in reach: the keypoints stay the ends
    # E311 decoys (critic pass): which piece is chosen. Entry keypoint (0.9, -2.3) heading +y; piece ends: 0 heading +x at (2.5, -2.2) (wrong direction), 1 heading +y at (0, -0.5) (right direction, 2.0 m away, FIRST in the list),
    # 2 heading +x at (0.5, -2.3) (nearest, wrong direction), 3 heading +y at (0, -2) (the nearest right one, 0.95 m); exit keypoint (8.5, 2.3): piece starts 4 heading +y at (8, 2.9) (0.78 m, first in the list), 5 heading +y at (8, 2) (0.58 m, the nearest),
    # 6 heading +x at (8.7, 2.3) (0.2 m: the nearest, wrong direction)
    def ln(a, b, n=10):
        return np.c_[np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)]
    dl = [ln((-7.5, -2.2), (2.5, -2.2)), ln((0, -9), (0, -0.5)), ln((-9.5, -2.3), (0.5, -2.3)), ln((0, -12), (0, -2)), ln((8, 2.9), (8, 12)), ln((8, 2), (8, 12)), ln((8.7, 2.3), (18.7, 2.3))]
    dk, dlk, dm = ["lane"] * 7, [(i,) for i in range(7)], np.zeros(7, bool)
    pr = [(np.array([0.9, -2.3]), np.array([0.0, 1.0]), np.array([8.5, 2.3]), np.array([0.0, 1.0]))]
    rd, nd, wd = add_connectors(dl, [], [[i] for i in range(7)], dk, dlk, dm, pr, snap=True)
    cv = rd["pts"][-1]
    assert wd == 1 and np.allclose(cv[0], (0, -2)) and np.allclose(cv[-1], (8, 2)), (wd, cv[[0, -1]])                      # from the end of the nearest right-direction piece to the start of the nearest right-direction piece
    assert sorted(e for e in rd["edges"] if 7 in e) == [(3, 7), (7, 5)], rd["edges"]                                      # exactly one piece per end
    ra, na, wa = add_connectors(dl, [], [[i] for i in range(7)], dk, dlk, dm, pr, snap=True, snap_all=True)
    assert np.allclose(ra["pts"][-1][0], (0, -2)) and {(1, 7), (3, 7), (7, 4), (7, 5)} <= set(ra["edges"]) and (0, 7) not in ra["edges"] and (2, 7) not in ra["edges"] and (7, 6) not in ra["edges"], ra["edges"]   # snap_all: every right-direction piece in reach of the snapped end
    rg, _, wg = add_connectors(dl, [], [[i] for i in range(7)], dk, dlk, dm, pr, snap=True, snap_nogeom=True)
    assert np.allclose(rg["pts"][-1][0], (0.9, -2.3)) and sorted(e for e in rg["edges"] if 7 in e) == [(3, 7), (7, 5)] and wg == 1, (rg["pts"][-1][0], rg["edges"])          # snap_nogeom: the keypoints stay the curve ends, one piece per end is wired
    rb, nb, wb = add_connectors(lines, [(0, 2), (2, 1)], [[0], [1], [2]], kinds, links, long_mask, far, snap=True, snap_both=True)
    assert len(rb["pts"]) == 3 and nb == 1 and wb == 0, (len(rb["pts"]), nb, wb)                    # the unwired far pair adds no curve (the delivered long link stays)
    # snap_guard: two abutting pieces make a chord of 0.5 m: the guard keeps the keypoints, without it the curve has no length
    ab = [ln((0, -10), (0, 0)), ln((0, 0.5), (0, 10))]
    pa_ = [(np.array([0.3, 0.0]), np.array([0.0, 1.0]), np.array([0.3, 0.4]), np.array([0.0, 1.0]))]
    r0, _, _ = add_connectors(ab, [], [[0], [1]], ["lane", "lane"], [(0,), (1,)], np.zeros(2, bool), pa_, snap=True)
    r1, _, _ = add_connectors(ab, [], [[0], [1]], ["lane", "lane"], [(0,), (1,)], np.zeros(2, bool), pa_, snap=True, snap_guard=True)
    assert np.allclose(r0["pts"][-1][0], (0, 0)) and np.allclose(r1["pts"][-1][0], (0.3, 0.0)) and np.isfinite(r0["pts"][-1]).all(), (r0["pts"][-1][0], r1["pts"][-1][0])
    # snap_dedup: two pairs that snap to the same two pieces give one curve
    two = pr + [(np.array([0.7, -2.4]), np.array([0.0, 1.0]), np.array([8.3, 2.4]), np.array([0.0, 1.0]))]
    assert len(add_connectors(dl, [], [[i] for i in range(7)], dk, dlk, dm, two, snap=True)[0]["pts"]) == 9 and len(add_connectors(dl, [], [[i] for i in range(7)], dk, dlk, dm, two, snap=True, snap_dedup=True)[0]["pts"]) == 8
    base = dict(topo_F1=0.60, **{"F1@1.0": 0.80}, dup_frac=0.2, vert100_dp=19.0, junction_F1=0.25)
    assert holds(dict(base, topo_F1=0.63), base)[0] and not holds(dict(base, topo_F1=0.61), base)[0] and not holds(dict(base, topo_F1=0.63, dup_frac=0.3), base)[0]
    # E320, critic 2026-11-12: the attached ends of a dump frame reach the curves (the mutation `qp, qd = kp_pos, kp_dir` passed every earlier test), also in the primary read-out (attached ends + one nearest piece per end, the geometry kept)
    e = dict(kp_pos=np.array([[1.2, -2.5], [8.9, 2.5]]), kp_dir=np.array([[0.0, 1.0], [0.0, 1.0]]), att_pos=np.array([[0.0, -2.0], [8.0, 2.0]]), att_dir=np.array([[0.0, 1.0], [0.0, 1.0]]), scale=np.full((1, 1, 2), 0.3))
    pk, pa_ = make_pairs(e, [0], [0], 1), make_pairs(e, [0], [0], 1, attach=True)
    assert np.allclose(pk[0][0], [1.2, -2.5]) and np.allclose(pa_[0][0], [0.0, -2.0]) and np.allclose(pa_[0][2], [8.0, 2.0]) and len(pa_[0]) == 6 and len(make_pairs(e, [0], [0], 1, attach=True, chord=True)[0]) == 4
    ck = add_connectors(lines, [(0, 2), (2, 1)], [[0], [1], [2]], kinds, links, long_mask, pk)[0]["pts"][-1]
    ca = add_connectors(lines, [(0, 2), (2, 1)], [[0], [1], [2]], kinds, links, long_mask, pa_)[0]["pts"][-1]
    assert np.allclose(ck[0], [1.2, -2.5]) and np.allclose(ca[0], [0.0, -2.0]) and np.allclose(ca[-1], [8.0, 2.0]), (ck[0], ca[0])
    rp, n_p, w_p = add_connectors(lines, [(0, 2), (2, 1)], [[0], [1], [2]], kinds, links, long_mask, pa_, snap=True, snap_nogeom=True)
    assert w_p == 1 and np.allclose(rp["pts"][-1][0], [0.0, -2.0]) and (0, len(rp["pts"]) - 1) in rp["edges"] and (len(rp["pts"]) - 1, 1) in rp["edges"], rp["edges"]                      # the curve stays on the attached ends, one piece per end is wired
    assert len(make_pairs(dict(e, scale=None), [0], [0], 1)[0]) == 4 and make_pairs(e, [0], [0], 1, fit={(0, 0): (0.5, 0.6)})[0][4:] == (0.5, 0.6)
    off_ = [(np.array([2.0, -2.0]), np.array([0.0, 1.0]), np.array([8.0, 2.0]), np.array([0.0, 1.0]))]                # the wiring radius: an end 2 m beside the piece end is wired by the default 4 m, not by a tight 1 m radius, and wired again at 2.5 m
    g_ = (lines, [(0, 2), (2, 1)], [[0], [1], [2]], kinds, links, long_mask, off_)
    assert add_connectors(*g_)[2] == 1 and add_connectors(*g_, wire_pos=1.0)[2] == 0 and add_connectors(*g_, wire_pos=2.5)[2] == 1 and add_connectors(*g_, wire_pos=4.0)[2] == 1
    off2_ = [(np.array([2.5, -2.0]), np.array([0.0, 1.0]), np.array([10.5, 2.0]), np.array([0.0, 1.0]))]              # N1: both ends 2.5 m from the delivered long link (piece ends at (0, -2) and (8, 2)): covered, hence dropped, by the default (4 m) and by a 4 m / 2.5 m radius; KEPT under a 1.0 m radius (not wired, so it must not delete the link)
    g2_ = (lines, [(0, 2), (2, 1)], [[0], [1], [2]], kinds, links, long_mask, off2_)
    assert len(add_connectors(*g2_)[0]["pts"]) == 3 and len(add_connectors(*g2_, wire_pos=4.0)[0]["pts"]) == 3 and len(add_connectors(*g2_, wire_pos=2.5)[0]["pts"]) == 3 and len(add_connectors(*g2_, wire_pos=1.0)[0]["pts"]) == 4
    print("qends_eval selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("dump")
    s.add_argument("--run", required=True)
    s.add_argument("--thr", type=float, default=0.5)
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--thr", type=float, default=0.5)
    s.add_argument("--full", default=None)
    s.add_argument("--base", default=None)
    s.add_argument("--chord", action="store_true", help="E303: use the chord tangent for the connectors even when the dump carries learned handle scales (the control)")
    s.add_argument("--snap", action="store_true", help="E311: draw the connectors from the end of the delivered piece wired to the entry to the start of the piece wired to the exit (the keypoints only decide which entry joins which exit); the card goes to ends_card_snap_thr<t>.json")
    for k, h in (("--snap-all", "with --snap: wire every piece in reach of the snapped end"), ("--snap-guard", "with --snap: a degenerate snapped curve keeps its keypoints"), ("--snap-dedup", "with --snap: no second curve between the same two snapped pieces"), ("--snap-nogeom", "with --snap: control, nearest piece per end for the wiring but the curve from the keypoints"), ("--snap-both", "with --snap: no curve for a pair that is not wired at both ends")):
        s.add_argument(k, action="store_true", help=h)
    sub.add_parser("selftest")
    args = ap.parse_args()
    {"dump": cmd_dump, "score": cmd_score, "selftest": lambda _: selftest()}[args.cmd](args)
