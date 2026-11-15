"""E346 driver (card experiments/E346_one-graph-code-ceiling): one arm of the ceiling of B's node + link code on the 873 VAL frames, its metric card, the self-test and the table.

    wsl bash tools/env/_py14.sh e346_arm.py g0|g1|l2|l1|l25|l31 [--max N] [--out FILE] [--skip-selftest]   # one arm (run.py job, CPU); writes runs/eval/e346_<arm>/val.json
    wsl bash tools/env/_py14.sh e346_arm.py s20|s25|s31|v20|v25|v31|m20|m25|c25 [--max N] [--out FILE]  # E355 step 0 arms (e355_lib.py); writes runs/eval/e355_<arm>/val.json
    wsl bash tools/env/_py14.sh e346_arm.py selftest                                                    # the kill-line checks alone, on every grid of the arms
    wsl bash tools/env/_py14.sh e346_arm.py report [--max N]                                            # table of the E346 cards (l2 / l1 columns only; the E356 table is temp/e356/e356_table.py)
    wsl bash tools/env/_py14.sh e346_arm.py report355 [--max N]                                         # the E355 table: the rule against g1, the three questions of the card, the counts
    wsl bash tools/env/_py14.sh e346_arm.py killrow --ref l2|s20 --card FILE                            # the kill rows of E355: a card against the stored l2 card (1e-6) / the E276 P1 row (5e-5)

--max N = every ceil(873 / N)-th frame (N = 31: every 29th, k = 0, 29 ... 870, the 31 frames of the critic's hand check); such a card goes to temp/e346/out/ and never to runs/eval/ (a partial card is no
evidence). Exit 0 = done; 1 = the kill line (g0 below 1.000, a self-test check failed, the parts of a metric do not add up to the scorer's counts): no card is written to runs/eval/, no arm is read;
2 = an error (data missing, crash). Arms: see e346_lib.py (E356: l25 = 2.5 m cells, N = 40; l31 = 3.125 m cells, N = 32: 3.0 m does not divide the 100 m window). Defaults of qtargets (2 m) are
restored after every arm and checked. The self-test of a job runs on the grid of its own arm (g0 / g1: 2 m); `selftest` runs it on all grids.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback

import numpy as np

import bflc_paths as BP
import e346_lib as L
import e355_lib as E5
import qformat
import qgraph_eval as QE
import qtargets as QT

CARDS = f"{BP.ROOT}/runs/eval"
SMOKE = f"{BP.ROOT}/temp/e346/out"
RULE = ("topo_F1", "lanes_found", "F1@1.0")        # the card's rule metrics
MAX_LOSS = 0.03
CHAIN_MAX = 0.05                                    # connectors with a collision cell: above this the chain code is rejected on that grid
E276_P1 = dict(topo_F1=0.9669, lanes_found=0.9853, **{"F1@1.0": 0.9957}, junction_F1=0.9535, Q=0.9842)   # E276 section 7a, P1, "sparse 0.3 m / 7.5 m" (4 decimals): the kill row of s20
SLOW = ("oracle_ms_mean", "readout_ms_mean", "readout_ms_p90")                                           # timing keys: not compared by killrow
LINE_CELLS = {2.0: 41, 1.0: 81, 2.5: 33, 3.125: 26}   # cells that a straight 80 m line (y from -40 to 40 m) crosses, by hand: floor(90 / res) - floor(10 / res) + 1 (the known answer of the collision toys)
HIGHER = ("Q", "S", "F1@1.0", "F1@0.5", "F1@1.5", "P@1.0", "R@1.0", "lanes_found", "topo_F1", "topo_P", "topo_R", "junction_F1", "junction_P", "junction_R", "R@1_far", "F1@1.0_lane", "F1@1.0_connector",
          "lanes_found_lane", "lanes_found_connector", "P@1.0_lane", "P@1.0_connector", "R@1.0_lane", "R@1.0_connector")
LOWER = ("vert100", "vert100_dp", "dup_frac", "false_len_frac", "readout_ms_mean", "readout_ms_p90", "oracle_ms_mean")
TABLE = ("topo_F1", "lanes_found", "F1@1.0", "junction_F1", "vert100", "vert100_dp", "Q", "topo_P", "topo_R", "dup_frac", "R@1_far", "F1@1.0_lane", "F1@1.0_connector", "lanes_found_lane", "lanes_found_connector",
         "P@1.0_lane", "R@1.0_lane", "P@1.0_connector", "R@1.0_connector", "link_conn_short_n", "link_conn_short_d_med", "link_conn_short_d_p90", "link_conn_long_n", "link_conn_long_d_med", "link_conn_long_d_p90",
         "link_lane_short_d_med", "link_lane_short_d_p90", "conn_supp_share", "conn_supp_on_long", "conn_supp_on_short_only", "conn_supp_on_no_link", "nodes_per_frame", "short_links_per_frame",
         "long_links_per_frame", "cells_per_frame", "channels_per_cell", "outputs_per_frame", "node_cell_share", "dropped_candidates_per_node", "oracle_ms_mean", "readout_ms_mean", "readout_ms_p90", "coll_raw_share", "coll_crossing_share",
         "coll_cell_share", "coll_connector_share", "coll_connector_cross_share", "coll_inter_connector_share", "coll_old_raw_share", "coll_old_crossing_share", "coll_old_connector_share",
         "coll_old_connector_cross_share")
TABLE355 = ("topo_F1", "lanes_found", "F1@1.0", "junction_F1", "vert100", "vert100_dp", "nodes_per_frame", "nodes_per_frame_p95", "nodes_per_frame_max", "junc_nodes_per_connector", "conn_nodes_per_connector",
            "nn_share_lt20", "nn_share_lt25", "wbl_connector_share", "wbl_inter_connector_share", "wbl_lost_share", "coll_raw_share", "coll_connector_share", "conn_cov_03", "conn_cov_10", "link_all_d_med",
            "link_all_d_p90", "link_all_rev_p90", "merge_ev_into_ev", "merge_norm_into_ev", "merge_norm_into_norm", "merge_unlinked", "merge_unlinked_kin", "merge_selfloops", "merge_dups", "merge_event_loss_share",
            "merge_node_share", "dropped_candidates_per_node", "events_merged_per_frame", "long_links_per_frame", "readout_ms_mean")


# ------------------------------------------------------------------------------------------------------------------------------ frames
def pick(frames, n):
    """every ceil(len / n)-th frame (all when n is None)"""
    return frames if not n else frames[::math.ceil(len(frames) / n)]


def run_frames(arm, frames, log=True):
    """-> the pooled records of `arm` over `frames` (the cell size of the arm is in force only inside this call)"""
    t0, recs = time.time(), []
    res = L.GRID.get(arm, QT.RES)
    with L.cell_size(res):
        if arm in L.GRID:
            L.check_cell_size(res)
        for i, f in enumerate(frames):
            recs.append(L.frame_record(arm, f))
            if log and (i + 1) % 100 == 0:
                print(f"   {arm}: {i + 1}/{len(frames)} frames, {time.time() - t0:.0f} s", flush=True)
    assert (QT.N, QT.RES) == (50, 2.0), "the defaults of qtargets were not restored"
    return L.pool(recs)


# ------------------------------------------------------------------------------------------------------------------------------ card
def commit():
    h = os.environ.get("AR_COMMIT", "")[:10]
    if h:
        return h
    try:
        return subprocess.run(["git", "-c", "safe.directory=*", "-C", BP.ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip() + "+?"
    except Exception:  # noqa: BLE001
        return "no-git"


def lane_signature():
    sys.path.insert(0, BP.REPO)
    from GtGenerator import lane_gt
    return lane_gt.signature()


def card_name(arm):
    return f"{'e355' if arm in L.ARMS355 else 'e346'}_{arm}"


def make_card(arm, frames, n_all, nums, wall):
    metrics = {k: v for k, v in nums.items() if v == v and math.isfinite(v)}
    card = dict(schema=1, name=card_name(arm), split="val", frames=len(frames), seed=0,
                protocol=dict(scorer="qgraph_eval.graph_counts against the old graph GT (graph_gt_val.pkl), the lane suite's settings", reference="val, the 873 frames of qgraph_eval.val_frames"),
                metrics=metrics, better={**{k: "higher" for k in HIGHER}, **{k: "lower" for k in LOWER}}, commit=commit(), date=time.strftime("%Y-%m-%d %H:%M"),
                info=dict(arm=arm, grid_m=L.GRID.get(arm), variant=E5.variant(arm) if arm in L.GRID else None, threads=os.environ.get("OMP_NUM_THREADS", ""), wall_s=round(wall, 1), s_per_frame=round(wall / max(len(frames), 1), 3),
                          frame_stride=math.ceil(n_all / len(frames)), frames_k=[frames[0].k, frames[-1].k], frame_hash=hashlib.sha1("".join(f.token for f in frames).encode()).hexdigest()[:12],
                          lane_gt_signature=None if arm == "g0" else lane_signature(), nan_metrics=sorted(k for k, v in nums.items() if v != v)))
    card["better"] = {k: v for k, v in card["better"].items() if k in metrics}
    if len(frames) != n_all:
        card["subset"] = f"stride{math.ceil(n_all / len(frames))}"
    return card


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(path + ".tmp", path)


def out_path(arm, n):
    return f"{SMOKE}/{card_name(arm)}_n{n}.json" if n else f"{CARDS}/{card_name(arm)}/val.json"


# ------------------------------------------------------------------------------------------------------------------------------ one arm
def run_arm(arm, n=None, out=None, selftest=True):
    if n and out and os.path.realpath(out).startswith(os.path.realpath(CARDS)):
        print("a --max card never goes to runs/eval/", file=sys.stderr)
        return 2
    path = out or out_path(arm, n)
    if os.path.exists(path) and os.path.realpath(path).startswith(os.path.realpath(CARDS)):
        print(f"{path} exists: a stored card is never overwritten (E356; use --out temp/... for a re-run)", file=sys.stderr)
        return 2
    if selftest:
        try:
            self_test((L.GRID.get(arm, 2.0),), arm)
        except AssertionError:
            traceback.print_exc()
            print("KILL: the self-test failed: no arm is read", file=sys.stderr)
            return 1
    all_frames = QE.val_frames()
    frames = pick(all_frames, n)
    print(f"arm {arm}: {len(frames)} of {len(all_frames)} VAL frames (every {math.ceil(len(all_frames) / len(frames))}th, k {frames[0].k} .. {frames[-1].k})", flush=True)
    t0 = time.time()
    try:
        pl = run_frames(arm, frames)
    except AssertionError:
        traceback.print_exc()
        print("KILL: the parts of a metric do not add up to the scorer's counts, or the defaults were not restored", file=sys.stderr)
        return 1
    nums = L.numbers(pl)
    card = make_card(arm, frames, len(all_frames), nums, time.time() - t0)
    if arm == "g0" and any(nums[k] < 1.0 - 1e-9 for k in RULE):
        write_json(os.path.join(os.path.dirname(path), "killed_" + os.path.basename(path)), card)
        print(f"KILL: arm g0 does not score 1.000: " + ", ".join(f"{k} {nums[k]:.6f}" for k in RULE), file=sys.stderr)
        return 1
    write_json(path, card)
    print(f"card {path}\n   " + "  ".join(f"{k} {nums[k]:.4f}" for k in ("topo_F1", "F1@1.0", "lanes_found", "junction_F1", "vert100", "vert100_dp")) + f"   ({card['info']['s_per_frame']} s per frame)")
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ self-test (the kill line)
def expect_fail(what, fn):
    try:
        fn()
    except AssertionError:
        return
    raise AssertionError(f"mutation not caught: {what}")


def toy_pieces(pieces, events=()):
    return dict(pieces=pieces, events=list(events), lines=dict(lane_divider=[], road_divider=[], boundary=[]))


def check_cell_harness(grids):
    L.check_cell_size(2.0)
    with L.cell_size(1.0):
        L.check_cell_size(1.0)
        assert (QT.N, QT.RES) == (100, 1.0)
    assert (QT.N, QT.RES) == (50, 2.0), "cell_size did not restore"
    for res in grids:
        with L.cell_size(res):
            L.check_cell_size(res)
            assert (QT.N, QT.RES) == (int(round(2 * QT.HALF / res)), float(res)), (QT.N, QT.RES, res)
    try:
        with L.cell_size(3.0):
            raise AssertionError("cell_size(3.0) did not refuse: 3 m does not divide the 100 m window (E356 uses 3.125 m)")
    except ValueError:
        pass
    L.check_cell_size(2.0)
    stale = L.stale_qtargets()
    import types
    for what, mutate, undo in (("only RES patched", lambda: setattr(QT, "RES", 1.0), lambda: setattr(QT, "RES", 2.0)),
                               ("only N patched", lambda: setattr(QT, "N", 100), lambda: setattr(QT, "N", 50))):
        mutate()
        try:
            expect_fail(what, lambda: L.check_cell_size(1.0 if "RES" in what else 2.0))
        finally:
            undo()
    for mod, attr in ((qformat, "QT"), (QE, "QT")):
        keep = getattr(mod, attr)
        for what, fake in ((f"{mod.__name__} holds a stale copy", stale), (f"{mod.__name__} holds a copy that hides its name", types.SimpleNamespace(**{k: v for k, v in vars(stale).items() if not k.startswith("__")}))):
            setattr(mod, attr, fake)
            try:
                with L.cell_size(1.0):
                    expect_fail(what, lambda: L.check_cell_size(1.0))
            finally:
                setattr(mod, attr, keep)
    L.check_cell_size(2.0)
    try:
        import qnodes
        assert max(v.stop - 1 if isinstance(v, slice) else v for v in qnodes.CH.values()) + 1 == 16
        g, _ = QE.toy_lane_gt()
        assert L.output_counts(qformat.oracle_prediction(g)["table"])[5] == 16, "channels per cell differ from the head's qnodes.CH"
    except ImportError:
        print("   (qnodes not importable: channel count not cross-checked)")


def check_geometry_and_collisions(grids):
    q = np.array([[0.0, 1.0], [3.0, 0.0], [0.0, 0.0]])
    assert np.allclose(L.poly_dist(q, [[-1, 0], [1, 0]]), [1.0, 2.0, 0.0]), "poly_dist"
    assert np.allclose(L.nn_distances([[0, 0], [1.5, 0], [5, 0], [5, 3]]), [1.5, 1.5, 3.0, 3.0]) and not len(L.nn_distances([[1.0, 1.0]])), "nn_distances"
    ys = lambda x, a, b: np.stack([np.full(int((b - a) / 0.5) + 1, x), np.arange(a, b + 1e-9, 0.5)], 1)
    A = QT._piece("A", ys(0.0, -40, 40))
    B = QT._piece("B", ys(0.0, -40, 40)[:, ::-1], kind="connector", inter=True)                      # crosses A at the origin at 90 degrees
    for res, occupied in ((r, 2 * LINE_CELLS[r] - 1) for r in grids):
        with L.cell_size(res):
            c = L.collision_counts(toy_pieces([A, B]))
        assert c.tolist() == [occupied, 1, 1, 1, 1, 1, 1, 1, 1], (res, c)                              # one shared cell: raw, crossing, collision; the one connector (in an intersection) is hit
    C = QT._piece("C", ys(0.0, -40, 40), kind="connector", inter=True)                              # the same points as A: runs together, one node per cell, no collision
    D = QT._piece("D", ys(0.5, -40, 40), kind="connector", inter=True)                              # 0.5 m beside A: every shared cell collides (more than 0.3 m apart, headings equal)
    for res, cells in ((r, LINE_CELLS[r]) for r in grids):
        with L.cell_size(res):
            same, side = L.collision_counts(toy_pieces([A, C])), L.collision_counts(toy_pieces([A, D]))
        assert same[1] == cells and same[2] == 0 and same[3] == 0 and same[5] == 0, (res, same)
        assert side[1] == cells and side[2] == 0 and side[3] == cells and side[5] == 1 and side[8] == 0, (res, side)
    A1 = QT._piece("A1", ys(0.0, -40, 0), succ=["J"])                                              # a join: the end point of A1 is the first point of J (as in the lane GT: gap 0)
    J = QT._piece("J", ys(0.0, 0, 40), kind="connector", inter=True, pred=["A1"])
    J2 = QT._piece("J", ys(0.0, 0.5, 40), kind="connector", inter=True, pred=["A1"])               # the same with a 0.5 m gap: two points more than 0.3 m apart, not one node
    for res in grids:
        with L.cell_size(res):
            joined, gap = L.collision_counts(toy_pieces([A1, J])), L.collision_counts(toy_pieces([A1, J2]))
        assert joined[1] == 1 and joined[3] == 0 and joined[5] == 0, (res, joined)
        assert gap[1] == 1 and gap[3] == 1 and gap[5] == 1, (res, gap)
    g, old = QE.toy_lane_gt()
    raw = L.lane_polylines(g)
    assert raw["edges"] == [(0, 1), (0, 2)], raw["edges"]
    m = QE.graph_metrics(raw, old)
    assert m["topo_F1"] == 1.0 and m["F1@1.0"] == 1.0, m


def check_breakdown_toy(grids):
    """(c) and (d) on the toy lane GT (a lane that splits into a straight piece and one curved connector inside an intersection): the connector is suppressed (no node), comes back as the one long
    link, which lies on the GT polyline; the other links are short lane links (about one per cell of the 80 m of straight lane: more than 75 % of 80 / res)"""
    g, old = QE.toy_lane_gt()
    old["kind"] = ["lane", "lane", "connector"]
    for res in grids:
        with L.cell_size(res):
            r = L.frame_record("l2", QE.Frame(0, 0, "toy", 0, old, _lane=g))                    # "l2": any decoded arm, the grid in force is the one that counts
        m = L.numbers(L.pool([r]))
        c = np.asarray(g["pieces"][2]["pts"], float)
        assert abs(m["conn_len_m"] - np.linalg.norm(np.diff(c, axis=0), axis=1).sum()) < 1e-9 and m["conn_supp_share"] == 1.0 and m["conn_supp_on_long"] == 1.0, (res, m)
        assert [m[f"link_{k}_n"] for k in ("conn_long", "lane_long", "conn_short")] == [1, 0, 0] and m["link_lane_short_n"] > 0.75 * 80 / res, (res, m)
        assert m["link_conn_long_d_p90"] < 0.3 and m["link_lane_short_d_p90"] < 1e-6 and m["lanes_found_connector"] == 1.0 and m["F1@1.0_connector"] > 0.95 and m["F1@1.0_lane"] == 1.0, (res, m)
        assert m["channels_per_cell"] == 16 and m["long_links_per_frame"] == 1 and m["cells_per_frame"] == (100 / res) ** 2 and m["coll_connector_share"] == 1.0 and m["coll_connector_cross_share"] == 0.0, (res, m)     # B and C leave the split together, then 0.3 - 3 m apart in the same cells
        assert m["nn_nodes_n"] == m["nodes_per_frame"] and 0.0 < m["nn_q05"] <= m["nn_q50"] <= m["nn_q75"] and 0.0 <= m["nn_share_lt20"] <= m["nn_share_lt25"] <= 1.0, (res, m)     # every delivered node has a neighbour; the spacing numbers are ordered
        assert m["nodes_per_frame_p95"] == m["nodes_per_frame_max"] == m["nodes_per_frame"] and m["outputs_per_frame_max"] == m["outputs_per_frame"] == 16 * (100 / res) ** 2, (res, m)
        r2 = dict(r, out=r["out"] + np.array([10.0, 0, 0, 0, 0, 0, 0, 0, 0]))                    # a second frame with 10 nodes more: mean + 5, max + 10, p95 (linear) + 9.5 (known answers of the pooling)
        m2 = L.numbers(L.pool([r, r2]))
        assert abs(m2["nodes_per_frame"] - m["nodes_per_frame"] - 5) < 1e-9 and m2["nodes_per_frame_max"] == m["nodes_per_frame"] + 10 and abs(m2["nodes_per_frame_p95"] - m["nodes_per_frame"] - 9.5) < 1e-9, (res, m2)


def planted(arm, frames, make):
    """pooled numbers of the frames with the planted graph / prediction made by make(frame) (arm g: a raw graph of the old GT; arm l2: a prediction)"""
    return L.numbers(L.pool([L.frame_record(arm, f, override=make(f)) for f in frames]))


def check_degradations(grids, step=97):
    """g0 and its planted copies once (no grid), then the decoded arm and its planted copies on every grid of `grids`"""
    import test_qgraph_eval as T                                              # the degraded copies of evaluation/tests/test_qgraph_eval.py
    fr, preds2 = T.frames(step=step)                                          # the oracle predictions on the 2 m grid: built here, outside any cell-size block
    check_degradations_g0(fr)
    for res in grids:
        with L.cell_size(res):
            check_degradations_decoded(T, fr, preds2 if res == 2.0 else [qformat.oracle_prediction(f.lane()) for f in fr], res)
    T._CACHE.clear()


def check_degradations_g0(fr):
    base0 = L.numbers(L.pool([L.frame_record("g0", f) for f in fr]))
    assert all(base0[k] >= 1.0 - 1e-12 for k in RULE), {k: base0[k] for k in RULE}               # kill line: g0 scores 1.000
    shifted = lambda f: dict(f.gt, pts=[p + [1.5, 0.0] for p in f.gt["pts"]])
    fewer = lambda f: dict(f.gt, pts=[p for i, p in enumerate(f.gt["pts"]) if i % 5], kind=[k for i, k in enumerate(f.gt["kind"]) if i % 5], edges=[])
    cut = lambda f: dict(f.gt, edges=[])
    nothing = lambda f: dict(pts=[], edges=[], kind=[])
    s, d, c, z = (planted("g0", fr, fn) for fn in (shifted, fewer, cut, nothing))
    assert s["F1@1.0"] < 0.3 and d["lanes_found"] < 0.9 and d["R@1.0"] < base0["R@1.0"] - 0.1 and c["topo_F1"] < base0["topo_F1"] - 0.05, (s["F1@1.0"], d["lanes_found"], c["topo_F1"])
    assert z["F1@1.0"] == 0.0 and z["lanes_found"] == 0.0 and z["topo_F1"] == 0.0, "an empty graph scores 0"
    lanes_only = lambda f: dict(f.gt, pts=[p for p, k in zip(f.gt["pts"], f.gt["kind"]) if k == "lane"], kind=["lane"] * sum(k == "lane" for k in f.gt["kind"]), edges=[])
    lo = planted("g0", fr, lanes_only)                                            # the parts by kind: no connector delivered -> the lane parts stay 1, the connector parts fall
    assert lo["P@1.0_lane"] == 1.0 and lo["R@1.0_lane"] == 1.0 and lo["lanes_found_lane"] == 1.0 and lo["P@1.0_connector"] == 0.0, lo
    assert lo["lanes_found_connector"] < 0.3 and lo["R@1.0_connector"] < base0["R@1.0_connector"] - 0.2, (lo["lanes_found_connector"], lo["R@1.0_connector"])


def check_degradations_decoded(T, fr, preds, res, arm="l2", bound=None):
    """arm l2 (any decoded arm) on the grid in force (`res` m): the oracle lies on the old GT, it is the stored scorer's number, the planted degradations of the predictions fall. The sanity bound on the
    oracle is E346's for 2 m and below; on the coarser grids of E356 it is looser (a coarse grid may lose some, that is a result, not a broken harness); `bound` = (topo F1, F1@1.0) replaces it
    (E355: the arms that lose by design, e.g. cell-centre positions)"""
    lo_topo, lo_f1 = bound or ((0.95, 0.99) if res <= 2.0 else (0.90, 0.97))
    p0 = dict(zip([f.k for f in fr], preds))
    lmap = lambda fn: (lambda f: dict(fn(p0[f.k]), table=p0[f.k]["table"]))
    base = L.numbers(L.pool([L.frame_record(arm, f, override=p0[f.k]) for f in fr]))
    assert base["topo_F1"] > lo_topo and base["F1@1.0"] > lo_f1, (res, "the oracle graph lies on the old GT", base["topo_F1"], base["F1@1.0"])
    ref = QE.score_frames(preds, fr)["graph"]                                    # the default path is the stored scorer's: same counts through graph_counts
    assert max(abs(base[k] - ref[k]) for k in ref if k in base and ref[k] == ref[k]) < 1e-12, f"arm {arm} differs from qgraph_eval.score_frames on the {res} m grid"
    rng = np.random.RandomState(0)

    def drop(p):
        q = T.clone(p)
        q["nodes"]["e"] = np.where(rng.rand(len(q["nodes"]["e"])) < 0.10, 0.0, 1.0)
        return q

    def cutlinks(p):
        q = T.clone(p)
        q["links"]["p"] = np.where(rng.rand(len(q["links"]["p"])) < 0.20, 0.0, 1.0)
        return q

    def empty(p):
        q = T.clone(p)
        q["nodes"] = {k: v[:0] for k, v in q["nodes"].items()}
        q["links"] = {k: (v[:0] if v is not None and k != "curves" else ([] if v is not None else None)) for k, v in q["links"].items()}
        return q

    sh = planted(arm, fr, lambda f: dict(T.shifted([p0[f.k]], 1.0)[0], table=p0[f.k]["table"]))
    dn, cl, em = (planted(arm, fr, lmap(fn)) for fn in (drop, cutlinks, empty))
    assert sh["F1@1.0"] < base["F1@1.0"] - 0.25 and sh["topo_F1"] < base["topo_F1"] - 0.2, (res, sh["F1@1.0"], sh["topo_F1"])
    assert dn["R@1.0"] < base["R@1.0"] - 0.05 and dn["topo_F1"] < base["topo_F1"] - 0.05, (res, dn["R@1.0"], dn["topo_F1"])
    assert cl["topo_R"] < base["topo_R"] - 0.05 and cl["topo_F1"] < base["topo_F1"] - 0.05, (res, cl["topo_R"], cl["topo_F1"])
    assert em["F1@1.0"] == 0.0 and em["lanes_found"] == 0.0 and em["topo_F1"] == 0.0 and abs(em["Q"] - 0.15 / 0.85) < 1e-9, (res, em["Q"])


# ------------------------------------------------------------------------------------------------------------------------------ self-test of the E355 arms
def toy_table(pos, ev=(), piece=None, links=()):
    """a hand-built node table (the keys `merge_nodes` reads): one node per position on a line, the cell index rising with the node number, links (src, dst) with the two nodes as path"""
    pos = np.asarray(pos, float).reshape(-1, 2)
    M, src, dst = len(pos), np.array([l[0] for l in links], int), np.array([l[1] for l in links], int)
    dist = np.linalg.norm(pos[src] - pos[dst], axis=1)
    return dict(cell=np.stack([np.arange(M), np.zeros(M, int)], 1), pos=pos, dirv=np.tile([1.0, 0.0], (M, 1)), flags=np.zeros((M, 8), bool), piece=np.zeros(M, int) if piece is None else np.asarray(piece, int),
                j=np.arange(M), ev=np.isin(np.arange(M), ev), sstart=np.zeros(M, bool), tier=np.zeros(M, np.uint8), vis=np.ones(M, bool), gid=np.arange(M), src=src, dst=dst, long=dist > QT.SHORT, dist=dist,
                path=[np.array(l) for l in links], pts=pos, stats=dict(nodes=M, events=len(ev), links=len(links), long=int((dist > QT.SHORT).sum())))


def check_merge_toys():
    """the merge rule on hand-built node tables (known answers): priority order, the nearest kept node, survivor keeps position and type, links re-pointed, self-loops and duplicates dropped, `long` by the
    chord, the counts by kind, the unlinked-piece counts, the spacing of the survivors"""
    line = lambda xs: [[x, 0.0] for x in xs]
    chain = [(0, 1), (1, 2), (2, 3), (3, 4)]
    nt, c = E5.merge_nodes(toy_table(line([0, 1.5, 3.0, 4.5, 6.0]), links=chain), 2.0)                       # every second node of a chain at 1.5 m: the others are closer than 2.0 m to a kept node
    assert nt["pos"][:, 0].tolist() == [0.0, 3.0, 6.0] and nt["src"].tolist() == [0, 1] and nt["dst"].tolist() == [1, 2] and c.tolist() == [0, 0, 2, 0, 0, 2, 0, 5, 0], (nt["pos"], c)
    assert np.diff(nt["pos"][:, 0]).min() >= 2.0 and nt["stats"]["nodes"] == 3 and nt["stats"]["links"] == 2, "survivors closer than the radius"
    nt, c = E5.merge_nodes(toy_table(line([0, 1.5, 3.0, 4.5, 6.0]), ev=(1,), links=chain), 2.0)              # an event node wins over the lower cell index: 1.5 and 4.5 survive
    assert nt["pos"][:, 0].tolist() == [1.5, 4.5] and nt["ev"].tolist() == [True, False] and c.tolist() == [0, 2, 1, 0, 0, 3, 0, 5, 1] and nt["src"].tolist() == [0] and nt["dst"].tolist() == [1], (nt["pos"], c)
    t = toy_table(line([0, 1.5, 3.0]), ev=(0, 1))                                                             # event into event: the first (lower cell index) survives with its own type, the type of the other is lost
    t["flags"][0, QT.FI["stop"]] = t["flags"][1, QT.FI["split"]] = True
    nt, c = E5.merge_nodes(t, 2.0)
    assert nt["pos"][:, 0].tolist() == [0.0, 3.0] and c[:3].tolist() == [1, 0, 0] and nt["flags"][0].tolist() == [True] + [False] * 7, (nt["flags"], c)
    nt, c = E5.merge_nodes(toy_table(line([0, 2.5, 1.8, 8.5]), ev=(0, 1), links=[(0, 2), (2, 3)]), 2.0)        # 1.8 is 1.8 m from the node at 0 and 0.7 m from the node at 2.5: into the NEAREST
    assert nt["pos"][:, 0].tolist() == [0.0, 2.5, 8.5] and nt["src"].tolist() == [0, 1] and nt["dst"].tolist() == [1, 2] and np.allclose(nt["dist"], [2.5, 6.0]) and c[1] == 1, (nt["pos"], nt["src"], nt["dst"])
    nt, c = E5.merge_nodes(toy_table(line([0, 1.0, 5.0]), ev=(0,), links=[(0, 2), (1, 2)]), 2.0)              # both links end at the same pair after the merge: the duplicate goes, the first link and its path stay
    assert len(nt["src"]) == 1 and c[6] == 1 and nt["path"][0].tolist() == [0, 2], (nt["src"], c)
    nt, c = E5.merge_nodes(toy_table(line([0, 4.5, 9.0]), ev=(0, 2), links=[(0, 1), (1, 2)]), 5.0)             # two short links (4.5 m) become one 9 m link: long again by the chord
    assert nt["long"].tolist() == [True] and not toy_table(line([0, 4.5, 9.0]), links=[(0, 1)])["long"].any() and c[5] == 1, nt["long"]
    cv = QT.decode(nt)[0]                                                                                     # the curve of the new link joins the survivors; its handle scales are fitted against the path (ids 1, 2)
    assert np.allclose(cv[0], nt["pos"][nt["src"][0]]) and np.allclose(cv[-1], nt["pos"][nt["dst"][0]]), "the curve does not join the survivors"
    nt, c = E5.merge_nodes(toy_table(line([0, 7.0, 10.0]), ev=(0, 1), links=[(0, 2)]), 3.5)                    # a 10 m link (long) ends in a node merged into the node at 7: 7 m, short
    assert toy_table(line([0, 7.0, 10.0]), links=[(0, 2)])["long"].tolist() == [True] and nt["long"].tolist() == [False], nt["long"]
    adj, kin = {(0, 1), (1, 0)}, {(2, 3), (3, 2)}
    nt, c = E5.merge_nodes(toy_table(line([0, 1.0, 10, 11, 20, 21]), ev=(0, 2, 4), piece=[0, 1, 2, 3, 4, 5]), 2.0, adj, kin)   # merges 1 -> 0 (adjacent pieces), 3 -> 2 (kin), 5 -> 4 (strangers)
    assert c[:3].tolist() == [0, 3, 0] and c[3] == 2 and c[4] == 1, c
    g, _ = QE.toy_lane_gt()
    assert E5.piece_relations(g) == ({(0, 1), (1, 0), (0, 2), (2, 0)}, {(1, 2), (2, 1)}), "A -> B, A -> C: B and C are kin"


def same_prediction(p, q):
    """the oracle prediction dicts agree (nodes, links with curves, assignment)"""
    for part in ("nodes", "links", "assign"):
        for k, v in p[part].items():
            w = q[part][k]
            if k in ("curves", "map_ids", "road_ids"):
                assert len(v) == len(w) and all(np.array_equal(a, b) for a, b in zip(v, w)), (part, k)
            else:
                assert np.array_equal(v, w), (part, k)


def crossing_lane():
    """lane A (x = 0.5 m, y from -7 to 7) and a connector K inside an intersection that starts at the origin and runs along +x: one shared cell (the one of the origin) on every grid, and A's candidate is
    strictly nearer to its centre, so K loses it"""
    A = QT._piece("A", np.stack([np.full(29, 0.5), np.arange(-7, 7.01, 0.5)], 1))
    K = QT._piece("K", np.stack([np.arange(0, 7.01, 0.5), np.zeros(15)], 1), kind="connector", inter=True)
    return toy_pieces([A, K])


def check_e355_variants(grids):
    """per grid on the toy lanes: the copy with the suppression off (the original untouched, equal to a deep copy with the flags cleared, nodes inside the junction), `prediction_of` = qformat's
    oracle, the sparse target, the cell-centre arm (minimum spacing = the lattice), the merge at the radius of the lattice, the wanted-but-lost count (hand-counted on the crossing toy)"""
    g, old = QE.toy_lane_gt()
    ref = copy.deepcopy(g)
    for p in ref["pieces"]:
        for s in p["segments"]:
            s["in_intersection"] &= s["kind"] != "connector"
    off = E5.suppress_off(g)
    assert g["pieces"][2]["segments"][0]["in_intersection"] and not off["pieces"][2]["segments"][0]["in_intersection"], "suppress_off must copy, not edit"
    cl = np.linalg.norm(np.diff(g["pieces"][2]["pts"], axis=0), axis=1).sum()
    X = crossing_lane()
    for res in grids:
        with L.cell_size(res):
            a, b, on = QT.node_table(off), QT.node_table(ref), QT.node_table(g)
            assert all(np.array_equal(a[k], b[k]) for k in ("pos", "cell", "src", "dst", "flags")), "suppress_off differs from a deep copy with the flags cleared"
            assert (on["piece"] == 2).sum() == 1 and (a["piece"] == 2).sum() >= 0.5 * cl / res, (res, (on["piece"] == 2).sum(), (a["piece"] == 2).sum())
            same_prediction(qformat.oracle_prediction(g), E5.prediction_of(g, on))
            dense, sparse = QT.node_table(g), E5.oracle_variant(g, "s20")["table"]
            assert 80 / E5.SPARSE[1] - 2 <= (sparse["piece"] <= 1).sum() < 0.8 * (dense["piece"] <= 1).sum(), (res, len(sparse["pos"]), len(dense["pos"]))      # the straight 80 m lane: a node every 7.5 m at most, fewer than one per cell
            c = E5.oracle_variant(g, "c25")["table"]
            assert np.allclose(c["pos"], QT.centre_of(c["cell"])) and abs(L.nn_distances(c["pos"]).min() - res) < 1e-9, (res, L.nn_distances(c["pos"]).min())
            m, cnt = E5.merge_nodes(dense, res)
            assert cnt[:3].sum() == len(dense["pos"]) - len(m["pos"]) and L.nn_distances(m["pos"]).min() >= res and cnt[7] == len(dense["pos"]), (res, len(m["pos"]), cnt)
            tables = {arm: QT.node_table(X if E5.variant(arm)["suppress"] else E5.suppress_off(X), sparse=E5.variant(arm)["sparse"]) for arm in ("s20", "v20")}
            s20, v20 = (E5.junction_nodes(X, tables[arm]) for arm in ("s20", "v20"))
            assert s20.tolist() == [0, 1, 0, 1] and v20[2] == (tables["v20"]["piece"] == 1).sum() >= 2 and v20[3] == 1, (res, s20, v20)           # nodes inside the junction: none with the suppression, the anchors of K without
            dl, s20, v20 = (E5.wanted_lost(X, arm) for arm in ("l2", "s20", "v20"))
            assert dl[:4].tolist() == [1, 0, 1, 0] and dl[5] == 0 and s20[:4].tolist() == [1, 0, 1, 0] and s20[5] == 0, (res, dl, s20)                # suppression ON: the connector has no candidates, nothing is lost
            assert v20[:4].tolist() == [1, 1, 1, 1] and v20[5] == 1 and v20[4] >= 5, (res, v20)                  # OFF: K's first candidate (its anchor, in the cell of the origin) is wanted and lost to A: one wanted-lost, the connector counts


def check_variant(arm, step=97):
    """an E355 arm on its grid: the planted degradations of the arm's own predictions fall and the arm's numbers are qgraph_eval.score_frames' (the checks of arm l2); the merge and the cell-centre
    arms keep their spacing"""
    import test_qgraph_eval as T
    fr, _ = T.frames(step=step)
    res, v = L.GRID[arm], E5.variant(arm)
    with L.cell_size(res):
        preds = [E5.oracle_variant(f.lane(), arm) for f in fr]
        check_degradations_decoded(T, fr, preds, res, arm, bound=(0.5, 0.5) if v["centre"] else None)
        floor = v["merge"] or (res if v["centre"] else 0.0)
        assert all(L.nn_distances(p["nodes"]["pos"]).min(initial=np.inf) >= floor - 1e-9 for p in preds), f"nodes closer than {floor} m in arm {arm}"
    T._CACHE.clear()


def self_test(grids, arm=None):
    """the kill-line checks on the cell sizes `grids` (a job: the grid of its arm; `selftest`: all grids of the arms); an E355 arm adds the checks of its variant"""
    t = time.time()
    check_cell_harness(grids)
    check_geometry_and_collisions(grids)
    check_breakdown_toy(grids)
    check_degradations(grids)
    check_merge_toys()
    check_e355_variants(grids)
    if arm in L.ARMS355:
        check_variant(arm)
    print(f"e346 self-test ok ({time.time() - t:.0f} s, grids {', '.join(f'{g:g} m' for g in grids)}): cell-size harness (mutations caught, 3.0 m refused), g0 = 1.000, planted degradations fall, parts by kind, l2 = qgraph_eval.score_frames, collision and link toys, "
          f"E355: merge toys, suppression off / sparse / cell centres / merge on the toy lanes, wanted-but-lost by hand" + (f", arm {arm}'s own predictions" if arm in L.ARMS355 else ""), flush=True)


# ------------------------------------------------------------------------------------------------------------------------------ table
def report(n=None):
    cards = {}
    for arm in L.ARMS:
        p = out_path(arm, n)
        if os.path.exists(p):
            cards[arm] = json.load(open(p, encoding="utf-8"))
    if not cards:
        print("no cards found", file=sys.stderr)
        return 2
    g0, g1 = cards.get("g0", {}).get("metrics", {}), cards.get("g1", {}).get("metrics", {})
    arms = list(cards)
    sizes = ", ".join("%s: %d frames" % (a, c["frames"]) for a, c in cards.items())
    print(f"E346 VAL, {sizes}\n")
    print("| metric | " + " | ".join(arms) + " | " + " | ".join(f"{a} - g1" for a in arms if a in ("l2", "l1")) + " | " + " | ".join(f"{a} - g0" for a in arms if a in ("l2", "l1")) + " |")
    print("|---|" + "---|" * (len(arms) + 2 * sum(a in ("l2", "l1") for a in arms)))
    for k in TABLE:
        vals = {a: cards[a]["metrics"].get(k) for a in arms}
        if all(v is None for v in vals.values()):
            continue
        fmt = lambda v: "-" if v is None else f"{v:.4f}"
        dg = lambda a, ref: "-" if vals[a] is None or ref.get(k) is None else f"{vals[a] - ref[k]:+.4f}"
        print(f"| {k} | " + " | ".join(fmt(vals[a]) for a in arms) + " | " + " | ".join(dg(a, g1) for a in arms if a in ("l2", "l1")) + " | " + " | ".join(dg(a, g0) for a in arms if a in ("l2", "l1")) + " |")
    print()
    for a in ("l2", "l1"):
        if a in cards and g1:
            loss = {k: g1[k] - cards[a]["metrics"][k] for k in RULE}
            print(f"rule {a}: loss against g1 " + ", ".join(f"{k} {v:+.4f}" for k, v in loss.items()) + f" (bar {MAX_LOSS}): {'PASS' if all(v <= MAX_LOSS for v in loss.values()) else 'FAIL'} (by hand; decide.py reads the card, see HANDOVER of this step)")
        if a in cards:
            m = cards[a]["metrics"]
            sh = m["coll_connector_share"]
            print(f"chain rule {a}: connectors with a collision cell {sh:.4f} (bar {CHAIN_MAX}): {'rejected' if sh > CHAIN_MAX else 'own card'}; of it by crossing alone {m['coll_connector_cross_share']:.4f}; "
                  f"raw / crossing share of occupied cells {m['coll_raw_share']:.4f} / {m['coll_crossing_share']:.4f} on the lane GT, {m['coll_old_raw_share']:.4f} / {m['coll_old_crossing_share']:.4f} on the old graph GT "
                  f"(E273: {'0.184 / 0.037' if a == 'l2' else '0.121 / 0.019'})")
    return 0


def report355(n=None):
    """the E355 table (card section 0): the numbers of the arms, the bar against g1, and the three questions as the card states them (readings by hand: GT replay; the verdict is the card author's)"""
    arms = [a for a in ("g1", "l2", "l25", "l31", *L.ARMS355) if os.path.exists(out_path(a, n))]
    cards = {a: json.load(open(out_path(a, n), encoding="utf-8")) for a in arms}
    if "g1" not in cards or len(arms) < 2:
        print("no g1 card or no E355 card found", file=sys.stderr)
        return 2
    mt = {a: c["metrics"] for a, c in cards.items()}
    print(f"E355 step 0, VAL, " + ", ".join(f"{a}: {c['frames']} frames" for a, c in cards.items()) + "\n")
    print("| metric | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms))
    for k in TABLE355:
        if any(k in mt[a] for a in arms):
            print(f"| {k} | " + " | ".join("-" if k not in mt[a] else f"{mt[a][k]:.4f}" for a in arms) + " |")
    print()
    loss = {a: {k: mt["g1"][k] - mt[a][k] for k in RULE} for a in arms if a != "g1"}
    ok = {a: max(l.values()) <= MAX_LOSS for a, l in loss.items()}
    print(f"bar (loss against g1 at most {MAX_LOSS} on {', '.join(RULE)}): " + "; ".join(f"{a} {'PASS' if ok[a] else 'FAIL'} ({', '.join(f'{v:+.4f}' for v in l.values())})" for a, l in loss.items()))
    for lat, v, s in (("2.0", "v20", "s20"), ("2.5", "v25", "s25"), ("3.125", "v31", "s31")):
        s = s if s in mt else "l2" if lat == "2.0" else None                      # at 2.0 m the reference is the row of E276 P1 = E346 l2
        if v in mt and s in mt:
            gain, wbl = mt[v]["junction_F1"] - mt[s]["junction_F1"], mt[v]["wbl_connector_share"]
            print(f"X2 at {lat} m: {v} {'passes' if ok[v] else 'FAILS'} the bar; junction F1 {mt[v]['junction_F1']:.4f} against {s} {mt[s]['junction_F1']:.4f} = {gain:+.4f} (bar +0.01); connectors with a wanted-but-lost node "
                  f"{wbl:.4f} (bar 0.05; raw collision share {mt[v]['coll_raw_share']:.4f}): {'YES' if ok[v] and gain >= 0.01 and wbl <= 0.05 else 'NO'}")
    for m, d in (("m20", "l2"), ("m25", "l25")):
        if m in mt and d in mt:
            dt, dj = mt[m]["topo_F1"] - mt[d]["topo_F1"], mt[m]["junction_F1"] - mt[d]["junction_F1"]
            print(f"spacing, {m} against {d}: {'passes' if ok[m] else 'FAILS'} the bar; topo F1 {dt:+.4f} (bar -0.01), junction F1 {dj:+.4f} (bar -0.02), event nodes lost {mt[m]['merge_event_loss_share']:.4f}, "
                  f"merges between pieces not linked {mt[m]['merge_unlinked']:.0f} of {mt[m]['merge_ev_into_ev'] + mt[m]['merge_norm_into_ev'] + mt[m]['merge_norm_into_norm']:.0f}: {'AFFORDABLE' if ok[m] and dt >= -0.01 and dj >= -0.02 else 'NOT AFFORDABLE'}")
    if "c25" in mt and "l25" in mt:
        print(f"c25 against l25: F1@1.0 {mt['c25']['F1@1.0'] - mt['l25']['F1@1.0']:+.4f}, topo F1 {mt['c25']['topo_F1'] - mt['l25']['topo_F1']:+.4f}, fit error p90 {mt['c25']['link_all_d_p90']:.3f} m against {mt['l25']['link_all_d_p90']:.3f} m")
    return 0


def killrow(ref, card_path):
    """the kill rows of E355: a full card of arm l2 run by THIS driver = the stored card runs/eval/e346_l2/val.json (every number but the timings, 1e-6); a full card of arm s20 (suppression ON, sparse 0.3 / 7.5,
    2.0 m) = the row of E276 P1 (5e-5)"""
    card = json.load(open(card_path, encoding="utf-8"))
    if card["frames"] != 873:
        print(f"killrow: {card_path} has {card['frames']} frames, not the 873 of VAL", file=sys.stderr)
        return 2
    new = card["metrics"]
    old, tol = (E276_P1, 5e-5) if ref == "s20" else ({k: v for k, v in json.load(open(f"{CARDS}/e346_l2/val.json", encoding="utf-8"))["metrics"].items() if k not in SLOW}, 1e-6)
    bad = {k: (v, new.get(k)) for k, v in old.items() if k not in new or abs(new[k] - v) > tol}
    print(f"killrow {ref}: {len(old) - len(bad)} of {len(old)} numbers within {tol:g} of {'E276 P1' if ref == 's20' else 'the stored l2 card'}" + (f"; DIFFER (reference, new): {bad}" if bad else ""))
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=(*L.ARMS, *L.ARMS355, "selftest", "report", "report355", "killrow"))
    ap.add_argument("--max", type=int, default=None, help="about this many frames, evenly strided (a smoke test; the card goes to temp/e346/out/)")
    ap.add_argument("--out", default=None, help="card path (default runs/eval/e346_<arm>/val.json; the E355 arms runs/eval/e355_<arm>/val.json)")
    ap.add_argument("--skip-selftest", action="store_true")
    ap.add_argument("--ref", choices=("l2", "s20"), help="killrow: the reference")
    ap.add_argument("--card", help="killrow: the card to check")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "selftest":
            self_test(tuple(dict.fromkeys(L.GRID.values())))
            return 0
        if a.cmd == "report":
            return report(a.max)
        if a.cmd == "report355":
            return report355(a.max)
        if a.cmd == "killrow":
            return killrow(a.ref, a.card)
        return run_arm(a.cmd, a.max, a.out, not a.skip_selftest)
    except AssertionError:
        traceback.print_exc()
        print("KILL: an assertion of the harness failed", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
