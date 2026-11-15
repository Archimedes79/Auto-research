"""E346 driver (card experiments/E346_one-graph-code-ceiling): one arm of the ceiling of B's node + link code on the 873 VAL frames, its metric card, the self-test and the table.

    wsl bash tools/env/_py14.sh e346_arm.py g0|g1|l2|l1 [--max N] [--out FILE] [--skip-selftest]    # one arm (run.py job, CPU); writes runs/eval/e346_<arm>/val.json
    wsl bash tools/env/_py14.sh e346_arm.py selftest                                                    # the kill-line checks alone
    wsl bash tools/env/_py14.sh e346_arm.py report [--max N]                                            # table of the four cards: absolute, against g1 and g0, the rule, the chain rule

--max N = every ceil(873 / N)-th frame (N = 31: every 29th, k = 0, 29 ... 870, the 31 frames of the critic's hand check); such a card goes to temp/e346/out/ and never to runs/eval/ (a partial card is no
evidence). Exit 0 = done; 1 = the kill line (g0 below 1.000, a self-test check failed, the parts of a metric do not add up to the scorer's counts): no card is written to runs/eval/, no arm is read;
2 = an error (data missing, crash). Arms: see e346_lib.py. Defaults of qtargets (2 m) are restored after every arm and checked.
"""
import argparse
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
import qformat
import qgraph_eval as QE
import qtargets as QT

CARDS = f"{BP.ROOT}/runs/eval"
SMOKE = f"{BP.ROOT}/temp/e346/out"
RULE = ("topo_F1", "lanes_found", "F1@1.0")        # the card's rule metrics
MAX_LOSS = 0.03
CHAIN_MAX = 0.05                                    # connectors with a collision cell: above this the chain code is rejected on that grid
HIGHER = ("Q", "S", "F1@1.0", "F1@0.5", "F1@1.5", "P@1.0", "R@1.0", "lanes_found", "topo_F1", "topo_P", "topo_R", "junction_F1", "junction_P", "junction_R", "R@1_far", "F1@1.0_lane", "F1@1.0_connector",
          "lanes_found_lane", "lanes_found_connector", "P@1.0_lane", "P@1.0_connector", "R@1.0_lane", "R@1.0_connector")
LOWER = ("vert100", "vert100_dp", "dup_frac", "false_len_frac", "readout_ms_mean", "readout_ms_p90", "oracle_ms_mean")
TABLE = ("topo_F1", "lanes_found", "F1@1.0", "junction_F1", "vert100", "vert100_dp", "Q", "topo_P", "topo_R", "dup_frac", "R@1_far", "F1@1.0_lane", "F1@1.0_connector", "lanes_found_lane", "lanes_found_connector",
         "P@1.0_lane", "R@1.0_lane", "P@1.0_connector", "R@1.0_connector", "link_conn_short_n", "link_conn_short_d_med", "link_conn_short_d_p90", "link_conn_long_n", "link_conn_long_d_med", "link_conn_long_d_p90",
         "link_lane_short_d_med", "link_lane_short_d_p90", "conn_supp_share", "conn_supp_on_long", "conn_supp_on_short_only", "conn_supp_on_no_link", "nodes_per_frame", "short_links_per_frame",
         "long_links_per_frame", "cells_per_frame", "channels_per_cell", "outputs_per_frame", "node_cell_share", "oracle_ms_mean", "readout_ms_mean", "readout_ms_p90", "coll_raw_share", "coll_crossing_share",
         "coll_cell_share", "coll_connector_share", "coll_inter_connector_share")


# ------------------------------------------------------------------------------------------------------------------------------ frames
def pick(frames, n):
    """every ceil(len / n)-th frame (all when n is None)"""
    return frames if not n else frames[::math.ceil(len(frames) / n)]


def run_frames(arm, frames, log=True):
    """-> the pooled records of `arm` over `frames` (the cell size of the arm is in force only inside this call)"""
    t0, recs = time.time(), []
    with L.cell_size(L.GRID.get(arm, QT.RES)):
        L.check_cell_size(QT.RES) if arm in L.GRID else None
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


def make_card(arm, frames, n_all, nums, wall):
    metrics = {k: v for k, v in nums.items() if v == v and math.isfinite(v)}
    card = dict(schema=1, name=f"e346_{arm}", split="val", frames=len(frames), seed=0,
                protocol=dict(scorer="qgraph_eval.graph_counts against the old graph GT (graph_gt_val.pkl), the lane suite's settings", reference="val, the 873 frames of qgraph_eval.val_frames"),
                metrics=metrics, better={**{k: "higher" for k in HIGHER}, **{k: "lower" for k in LOWER}}, commit=commit(), date=time.strftime("%Y-%m-%d %H:%M"),
                info=dict(arm=arm, grid_m=L.GRID.get(arm), threads=os.environ.get("OMP_NUM_THREADS", ""), wall_s=round(wall, 1), s_per_frame=round(wall / max(len(frames), 1), 3),
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
    return f"{SMOKE}/e346_{arm}_n{n}.json" if n else f"{CARDS}/e346_{arm}/val.json"


# ------------------------------------------------------------------------------------------------------------------------------ one arm
def run_arm(arm, n=None, out=None, selftest=True):
    if n is None and out and os.path.realpath(out).startswith(os.path.realpath(CARDS)) is False:
        pass
    if n and out and os.path.realpath(out).startswith(os.path.realpath(CARDS)):
        print("a --max card never goes to runs/eval/", file=sys.stderr)
        return 2
    if selftest:
        try:
            self_test()
        except AssertionError:
            traceback.print_exc()
            print(f"KILL: the self-test failed: no arm is read", file=sys.stderr)
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
    path = out or out_path(arm, n)
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


def check_cell_harness():
    L.check_cell_size(2.0)
    with L.cell_size(1.0):
        L.check_cell_size(1.0)
        assert (QT.N, QT.RES) == (100, 1.0)
    assert (QT.N, QT.RES) == (50, 2.0), "cell_size did not restore"
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
        for what, fake in ((f"{mod.__name__} holds a stale copy", stale), (f"{mod.__name__} holds a copy that hides its name", types.SimpleNamespace(**vars(stale)))):
            setattr(mod, attr, fake)
            try:
                with L.cell_size(1.0):
                    expect_fail(what, lambda: L.check_cell_size(1.0))
            finally:
                setattr(mod, attr, keep)
    L.check_cell_size(2.0)
    try:
        import qnodes
        assert max(max(v.stop - 1 if isinstance(v, slice) else v for v in qnodes.CH.values()) + 1 for _ in (0,)) == 16
        g, _ = QE.toy_lane_gt()
        assert L.output_counts(qformat.oracle_prediction(g)["table"])[5] == 16, "channels per cell differ from the head's qnodes.CH"
    except ImportError:
        print("   (qnodes not importable: channel count not cross-checked)")


def check_geometry_and_collisions():
    q = np.array([[0.0, 1.0], [3.0, 0.0], [0.0, 0.0]])
    assert np.allclose(L.poly_dist(q, [[-1, 0], [1, 0]]), [1.0, 2.0, 0.0]), "poly_dist"
    ys = lambda x, a, b: np.stack([np.full(int((b - a) / 0.5) + 1, x), np.arange(a, b + 1e-9, 0.5)], 1)
    A = QT._piece("A", ys(0.0, -40, 40))
    B = QT._piece("B", ys(0.0, -40, 40)[:, ::-1], kind="connector", inter=True)                      # crosses A at the origin at 90 degrees
    for res, occupied in ((2.0, 81 + 81 - 1), (1.0, 161 + 161 - 1)):
        with L.cell_size(res):
            c = L.collision_counts(toy_pieces([A, B]))
        assert c.tolist() == [occupied, 1, 1, 1, 1, 1, 1, 1], (res, c)                              # one shared cell: raw, crossing, collision; the one connector (in an intersection) is hit
    C = QT._piece("C", ys(0.0, -40, 40), kind="connector", inter=True)                              # the same points as A: runs together, one node per cell, no collision
    D = QT._piece("D", ys(0.5, -40, 40), kind="connector", inter=True)                              # 0.5 m beside A: every shared cell collides (more than 0.3 m apart, headings equal)
    for res, cells in ((2.0, 41), (1.0, 81)):
        with L.cell_size(res):
            same, side = L.collision_counts(toy_pieces([A, C])), L.collision_counts(toy_pieces([A, D]))
        assert same[1] == cells and same[2] == 0 and same[3] == 0 and same[5] == 0, (res, same)
        assert side[1] == cells and side[2] == 0 and side[3] == cells and side[5] == 1, (res, side)
    g, old = QE.toy_lane_gt()
    raw = L.lane_polylines(g)
    assert raw["edges"] == [(0, 1), (0, 2)], raw["edges"]
    m = QE.graph_metrics(raw, old)
    assert m["topo_F1"] == 1.0 and m["F1@1.0"] == 1.0, m


def planted(arm, frames, make):
    """pooled numbers of the frames with the planted graph / prediction made by make(frame) (arm g: a raw graph of the old GT; arm l2: a prediction)"""
    return L.numbers(L.pool([L.frame_record(arm, f, override=make(f)) for f in frames]))


def check_degradations(step=97):
    import test_qgraph_eval as T                                              # the degraded copies of evaluation/tests/test_qgraph_eval.py
    fr, preds = T.frames(step=step)
    base0 = L.numbers(L.pool([L.frame_record("g0", f) for f in fr]))
    assert all(base0[k] >= 1.0 - 1e-12 for k in RULE), {k: base0[k] for k in RULE}               # kill line: g0 scores 1.000
    shifted = lambda f: dict(f.gt, pts=[p + [1.5, 0.0] for p in f.gt["pts"]])
    fewer = lambda f: dict(f.gt, pts=[p for i, p in enumerate(f.gt["pts"]) if i % 5], kind=[k for i, k in enumerate(f.gt["kind"]) if i % 5], edges=[])
    cut = lambda f: dict(f.gt, edges=[])
    nothing = lambda f: dict(pts=[], edges=[], kind=[])
    s, d, c, z = (planted("g0", fr, fn) for fn in (shifted, fewer, cut, nothing))
    assert s["F1@1.0"] < 0.3 and d["lanes_found"] < 0.9 and d["R@1.0"] < base0["R@1.0"] - 0.1 and c["topo_F1"] < base0["topo_F1"] - 0.05, (s["F1@1.0"], d["lanes_found"], c["topo_F1"])
    assert z["F1@1.0"] == 0.0 and z["lanes_found"] == 0.0 and z["topo_F1"] == 0.0, "an empty graph scores 0"
    p0 = dict(zip([f.k for f in fr], preds))
    lmap = lambda fn: (lambda f: dict(fn(p0[f.k]), table=p0[f.k]["table"]))
    base = L.numbers(L.pool([L.frame_record("l2", f, override=p0[f.k]) for f in fr]))
    assert base["topo_F1"] > 0.95 and base["F1@1.0"] > 0.99, "the oracle graph lies on the old GT"
    ref = QE.score_frames(preds, fr)["graph"]                                    # the default path is the stored scorer's: same counts through graph_counts
    assert max(abs(base[k] - ref[k]) for k in ref if k in base and ref[k] == ref[k]) < 1e-12, "arm l2 differs from qgraph_eval.score_frames"
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

    sh = planted("l2", fr, lambda f: dict(T.shifted([p0[f.k]], 1.0)[0], table=p0[f.k]["table"]))
    dn, cl, em = (planted("l2", fr, lmap(fn)) for fn in (drop, cutlinks, empty))
    assert sh["F1@1.0"] < base["F1@1.0"] - 0.25 and sh["topo_F1"] < base["topo_F1"] - 0.2, (sh["F1@1.0"], sh["topo_F1"])
    assert dn["R@1.0"] < base["R@1.0"] - 0.05 and dn["topo_F1"] < base["topo_F1"] - 0.05, (dn["R@1.0"], dn["topo_F1"])
    assert cl["topo_R"] < base["topo_R"] - 0.05 and cl["topo_F1"] < base["topo_F1"] - 0.05, (cl["topo_R"], cl["topo_F1"])
    assert em["F1@1.0"] == 0.0 and em["lanes_found"] == 0.0 and em["topo_F1"] == 0.0 and abs(em["Q"] - 0.15 / 0.85) < 1e-9, em["Q"]
    T._CACHE.clear()


def self_test():
    t = time.time()
    check_cell_harness()
    check_geometry_and_collisions()
    check_degradations()
    print(f"e346 self-test ok ({time.time() - t:.0f} s): cell-size harness (mutations caught), g0 = 1.000, planted degradations fall, l2 = qgraph_eval.score_frames, collision toys", flush=True)


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
    print(f"E346 VAL, {', '.join(f'{a}: {c['frames']} frames' for a, c in cards.items())}\n")
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
            sh = cards[a]["metrics"]["coll_connector_share"]
            print(f"chain rule {a}: connectors with a collision cell {sh:.4f} (bar {CHAIN_MAX}): {'rejected' if sh > CHAIN_MAX else 'own card'}; raw / crossing share {cards[a]['metrics']['coll_raw_share']:.4f} / {cards[a]['metrics']['coll_crossing_share']:.4f}"
                  f" (E273: {'0.184 / 0.037' if a == 'l2' else '0.121 / 0.019'})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=(*L.ARMS, "selftest", "report"))
    ap.add_argument("--max", type=int, default=None, help="about this many frames, evenly strided (a smoke test; the card goes to temp/e346/out/)")
    ap.add_argument("--out", default=None, help="card path (default runs/eval/e346_<arm>/val.json)")
    ap.add_argument("--skip-selftest", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "selftest":
            self_test()
            return 0
        if a.cmd == "report":
            return report(a.max)
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
