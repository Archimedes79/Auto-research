"""Trace a detection error back to its frame: every false positive and every missed box of an evaluated run, as a table that says what kind of error it is, where it
is, and how to look at it.

    python evaluation/bev_errors.py RUN [--n 1000|all] [--tag ""] [--store NAME] [--min-score 0.3] [--thr 2.0] [--top 20] [--class vehicle] [--check] [--out FILE.csv]
    python evaluation/bev_errors.py --selftest

Only boxes inside the scored window of the metric (x < 60 m, +-32.3 deg of the front axis) are counted, as the numbers do.
Input: the boxes of every frame that `bev_suite.py RUN eval` stored (RUN/eval/scene_records<tag>_n<N>.pkl: detections and ground-truth boxes per scene and frame) and the
frame list of its suite store (RUN/eval/suite/quant/eval<tag>_n<N>/frames.jsonl: scene, index, token). Nothing is re-run, no GPU. The matching is the metric's own
(`evalsuite.comparison.boxes.match`: best score first, nearest free box of the class within `thr` metres), applied to the detections with score >= `min-score` (what the
pictures draw), so a row here is a box the picture shows in the error colour.

Kinds (the reason an error is what it is; the first that applies wins, in the order listed):
  FP  duplicate       a ground-truth box of its class lies within `thr` m but another detection has it
      wrong_class     a ground-truth box of ANOTHER class within `thr` m that no detection of its own class found (the class confusion: that box is a miss of its class)
      isolated        nothing of the ground truth within `thr` m (a ghost; the nearest box is given)
  FN  taken           a detection of its class lies within `thr` m but went to another box (two objects, one detection)
      below_score     a detection of its class within `thr` m exists but its score is under `min-score` (a threshold effect, not a miss of the network)
      wrong_class     a detection of another class sits on it and is itself a false positive (the class confusion)
      missed          nothing of the kinds above (`explained_by` then names the nearest detection within 2 x `thr`, if any)
The tables count every annotated box of the scored window, like the plain `det` numbers; the pictures (bev_views) apply the D38 'no care' rule on top (boxes with less than half of
their footprint in the observed area are neither right nor false nor missed there), so a count on a picture can be smaller. The table also lists, for every error, the frame (scene, index, token), the position and range, the score (FP) or the size (FN) and the command that redraws the frame.
Summaries: by class and kind, by range bin, by score bin, by scene (where the errors concentrate: the scenes to watch the video of)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]
from evalsuite.comparison import boxes as BX  # noqa: E402
from evaluation.metrics import in_front  # noqa: E402

RANGE_BINS = ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0))                       # = evaluation/metrics.RANGE_BINS (not imported: this module needs no torch)
SCORE_BINS = ((0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01))


def _dist(a, b):
    return math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])


def classify_frame(dets, gts, groups, thr=2.0, min_score=0.3):
    """One frame -> {"tp": [(det, gt)], "fp": [(det, kind, other)], "fn": [(gt, kind, other)]}; `other` is the box that explains the kind (or None). dets / gts: box dicts
    (group, cx, cy, ...; score for dets). The detections with score < min_score are not matched (they only explain a 'below_score' miss)."""
    shown = [d for d in dets if d.get("score", 1.0) >= min_score]
    low = [d for d in dets if d.get("score", 1.0) < min_score]
    per = {}
    for cls in groups:                                                   # pass 1: the matching of every class (the metric's own)
        ds, gs = [d for d in shown if d["group"] == cls], [g for g in gts if g["group"] == cls]
        match = BX.match(ds, gs, thr)
        per[cls] = (ds, gs, match, {j for j in match if j >= 0})
    free_gt = [g for cls in groups for j, g in enumerate(per[cls][1]) if j not in per[cls][3]]            # ground-truth boxes nobody found
    false_det = [d for cls in groups for d, j in zip(per[cls][0], per[cls][2]) if j < 0]                  # detections that found nothing
    out = {"tp": [], "fp": [], "fn": []}
    for cls in groups:
        ds, gs, match, got = per[cls]
        taken_by = {j: i for i, j in enumerate(match) if j >= 0}
        for i, (d, j) in enumerate(zip(ds, match)):
            if j >= 0:
                out["tp"].append((d, gs[j]))
                continue
            twin = next((gs[k] for k in range(len(gs)) if _dist(d, gs[k]) <= thr), None)
            other = [g for g in free_gt if g["group"] != cls and _dist(d, g) <= thr]
            if twin is not None:
                out["fp"].append((d, "duplicate", twin))                                      # a box of its class is near, but another detection has it
            elif other:
                out["fp"].append((d, "wrong_class", min(other, key=lambda g: _dist(d, g))))
            else:
                near = min(gts, key=lambda g: _dist(d, g), default=None)
                out["fp"].append((d, "isolated", near))
        for j, g in enumerate(gs):
            if j in taken_by:
                continue
            same = [d for d in shown if d["group"] == cls and _dist(d, g) <= thr]            # (all matched elsewhere, or this one would have taken g)
            lows = [d for d in low if d["group"] == cls and _dist(d, g) <= thr]
            wrong = [d for d in false_det if d["group"] != cls and _dist(d, g) <= thr]
            anyd = [d for d in dets if _dist(d, g) <= 2 * thr]
            if same:
                out["fn"].append((g, "taken", same[0]))
            elif lows:
                out["fn"].append((g, "below_score", max(lows, key=lambda d: d["score"])))
            elif wrong:
                out["fn"].append((g, "wrong_class", wrong[0]))
            else:
                out["fn"].append((g, "missed", None if not anyd else min(anyd, key=lambda d: _dist(d, g))))
    return out


def find_store(run, n, tag="", store=None):
    """the suite store of the evaluation: eval<tag>_n<N>, with the optional _curves / _fusion / _vel parts the flags of `eval` add to its name (or the one named by `store`);
    refuses none or several: with several, the pickle of boxes belongs to the latest evaluation of that name, so say which (--store NAME)"""
    import re
    base = os.path.join(run, "eval", "suite", "quant")
    if store:
        if not os.path.exists(os.path.join(base, store, "frames.jsonl")):
            raise SystemExit(f"no store {os.path.join(base, store)}")
        return os.path.join(base, store)
    pat = re.compile(rf"^eval{re.escape(tag)}(_curves|_fusion|_vel)*_n{n}$")
    found = sorted(d for d in os.listdir(base) if pat.match(d) and os.path.exists(os.path.join(base, d, "frames.jsonl"))) if os.path.isdir(base) else []
    if len(found) != 1:
        raise SystemExit(f"{'no' if not found else 'several'} suite store for --n {n} --tag '{tag}' in {base}: {found or sorted(os.listdir(base)) if os.path.isdir(base) else 'no quant folder'} "
                         f"(evaluate first: bev_suite.py {run} eval --n {n})")
    return os.path.join(base, found[0])


def load_frames(run, n, tag="", store=None):
    """[(scene, index, token, label)] in the order the suite evaluated them, from the suite store of the evaluation"""
    with open(os.path.join(find_store(run, n, tag, store), "frames.jsonl")) as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    return [(r["scene"], r["index"], r["key"], r["label"]) for r in rows]


def load_boxes(run, n, tag="", dt=0.0):
    """({(scene, k): (dets, gts)}, meta): k the position of the frame in its scene's list, from scene_records<tag>_n<N>.pkl; meta = its `frames` (the keys, in order) and GT signature"""
    path = os.path.join(run, "eval", f"scene_records{tag}_n{n}.pkl")
    if not os.path.exists(path):
        raise SystemExit(f"no {path}: the boxes of the frames are stored by `bev_suite.py {run} eval --n {n}` (runs evaluated before 2026-10-31 have only scene_records.pkl)")
    with open(path, "rb") as f:
        rec = pickle.load(f)
    return {(s, k): fr for s, per in rec["det"].items() for k, fr in enumerate(per.get(dt, []))}, rec.get("meta") or {}


def error_rows(run, n, tag="", groups=("vehicle", "pedestrian", "two_wheeler"), thr=2.0, min_score=0.3, dt=0.0, store=None):
    """-> (rows, totals, n_frames): one dict per error (kind FP / FN) with the frame it is in and the command that redraws it; totals {class: {"tp","fp","fn"}}. The records and
    the store must be one evaluation: same number of frames, and when the records carry their frame keys (every evaluation since 2026-10-31) the same keys in the same order."""
    frames, (boxes, meta) = load_frames(run, n, tag, store), load_boxes(run, n, tag, dt)
    seen, rows, tot = Counter(), [], defaultdict(Counter)
    if len(frames) != sum(1 for _ in boxes):
        raise SystemExit(f"{len(frames)} frames in the suite store but {len(boxes)} frames of boxes in the records: they are not from the same evaluation (--tag / --n / --store?)")
    if meta.get("frames") is not None and list(meta["frames"]) != [f[2] for f in frames]:
        raise SystemExit("the frame keys stored with the records differ from the suite store's frame list (another evaluation of the same name: --store NAME, or re-evaluate)")
    if meta.get("frames") is None:
        print("WARNING: the records carry no frame keys (evaluated before 2026-10-31): the frame of an error is taken from the order only", flush=True)
    for scene, index, token, label in frames:
        k = seen[scene]
        if (scene, k) not in boxes:
            raise SystemExit(f"the records hold no frame {k} of {scene} (the suite store lists more frames of it)")
        seen[scene] += 1
        dets, gts = boxes[(scene, k)]
        r = classify_frame([d for d in dets if in_front(d)], [g for g in gts if in_front(g)], groups, thr, min_score)      # the scored window of the metric (x < 60 m, +-32.3 deg)
        for d, g in r["tp"]:
            tot[d["group"]]["tp"] += 1
        for d, kind, other in r["fp"]:
            tot[d["group"]]["fp"] += 1
            rows.append(_row("FP", kind, d, other, scene, index, token, label, run, n))
        for g, kind, other in r["fn"]:
            tot[g["group"]]["fn"] += 1
            rows.append(_row("FN", kind, g, other, scene, index, token, label, run, n))
    return rows, {c: dict(v) for c, v in tot.items()}, len(frames)


def _row(what, kind, b, other, scene, index, token, label, run, n):
    size = (round(b.get("length", 2 * b.get("radius", 0.0)), 1), round(b.get("width", 2 * b.get("radius", 0.0)), 1))
    return {"what": what, "kind": kind, "class": b["group"], "scene": scene, "index": index, "token": token, "x": round(b["cx"], 2), "y": round(b["cy"], 2),
            "range_m": round(math.hypot(b["cx"], b["cy"]), 1), "score": round(b["score"], 3) if "score" in b else "", "size": f"{size[0]}x{size[1]}",
            "explained_by": ("" if other is None else f"{other['group']} at ({other['cx']:.1f}, {other['cy']:.1f})" + (f" score {other['score']:.2f}" if "score" in other else "")),
            "redraw": f"python evaluation/bev_suite.py {run} qual frame {label} --view overview"}


def _bin(x, bins):
    return next((f"{lo:g}-{hi:g} m" for lo, hi in bins if lo <= x < hi), "beyond")


def summaries(rows, totals):
    """the tables as text: by class and kind, by range, by score (FP), by scene (the worst ones first)"""
    out = []
    out.append("by class: " + "; ".join(f"{c}: tp {v.get('tp', 0)}, fp {v.get('fp', 0)}, fn {v.get('fn', 0)}" for c, v in totals.items()))
    kinds = Counter((r["class"], r["what"], r["kind"]) for r in rows)
    out.append("by kind:")
    out += [f"  {c:12s} {w} {k:12s} {v:5d}" for (c, w, k), v in sorted(kinds.items())]
    rng = Counter((r["what"], _bin(r["x"], RANGE_BINS)) for r in rows)
    out.append("by range (x ahead of the car, the bins of the metric): " + "; ".join(f"{w} {b}: {v}" for (w, b), v in sorted(rng.items())))
    sc = Counter(next((f"{lo:g}-{min(hi, 1):g}" for lo, hi in SCORE_BINS if lo <= r["score"] < hi), "<0.3") for r in rows if r["what"] == "FP")
    out.append("false positives by score: " + "; ".join(f"{b}: {v}" for b, v in sorted(sc.items())))
    by_scene = Counter(r["scene"] for r in rows)
    out.append("scenes with the most errors (watch these videos): " + "; ".join(f"{s} {v}" for s, v in by_scene.most_common(8)))
    return "\n".join(out)


def check_against_card(run, n, tag="", groups=("vehicle", "pedestrian", "two_wheeler"), thr=2.0, store=None):
    """the proof that this table is the metric's: with every detection counted (min-score 0) the true positives, the missed boxes and the false positives per class equal
    the card's n_tp, n_gt - n_tp and n_det - n_tp (dt 0, the scored window). -> list of (class, what, here, card) that differ (empty = identical)"""
    card = os.path.join(run, "eval", f"card{tag}.json")
    if not os.path.exists(card):
        raise SystemExit(f"no {card} to check against")
    with open(card) as f:
        m = json.load(f)["metrics"]
    _, tot, _ = error_rows(run, n, tag, groups, thr, 0.0, store=store)
    bad = []
    for c in groups:
        t, k = tot.get(c, {}), f"det/dt+0.00/{c}/all/"
        for what, here, want in (("tp", t.get("tp", 0), m.get(k + "n_tp")), ("fn", t.get("fn", 0), None if m.get(k + "n_gt") is None else m[k + "n_gt"] - m.get(k + "n_tp", 0)),
                                 ("fp", t.get("fp", 0), None if m.get(k + "n_det") is None else m[k + "n_det"] - m.get(k + "n_tp", 0))):
            if want is None or int(want) != here:
                bad.append((c, what, here, want))
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?")
    ap.add_argument("--n", default="1000", help="the n of the evaluation (a number, or all)")
    ap.add_argument("--tag", default="", help="the tag of the evaluation (eval --warm writes _warm)")
    ap.add_argument("--store", default=None, help="the suite store folder name when several fit (eval_curves_fusion_n1000, ...)")
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--thr", type=float, default=2.0)
    ap.add_argument("--top", type=int, default=20, help="how many rows to print per kind (the CSV has them all)")
    ap.add_argument("--class", dest="cls", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true", help="also prove the table against the card: with every detection counted, tp / fn / fp per class equal the card's numbers")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if not a.run:
        ap.error("RUN is required (or --selftest)")
    groups = (a.cls,) if a.cls else ("vehicle", "pedestrian", "two_wheeler")
    if a.check:
        bad = check_against_card(a.run.rstrip("/"), a.n, a.tag, groups, a.thr, a.store)
        print("check against the card (every detection counted): " + ("IDENTICAL for tp / fn / fp of every class" if not bad else f"DIFFERENT: {bad}"))
        if bad:
            return 1
    if a.check and a.thr != 2.0:
        ap.error("--check compares with the card's n_tp, which is counted at 2.0 m")
    rows, totals, nf = error_rows(a.run.rstrip("/"), a.n, a.tag, groups, a.thr, a.min_score, store=a.store)
    out = a.out or os.path.join(a.run, "eval", f"errors{a.tag}_n{a.n}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["what"])
        w.writeheader()
        w.writerows(rows)
    print(f"{a.run}: {nf} frames, {len(rows)} errors (score >= {a.min_score}, match within {a.thr} m) -> {out}\n")
    print(summaries(rows, totals))
    for what, key, rev in (("FP", lambda r: -r["score"], False), ("FN", lambda r: r["range_m"], False)):
        sel = sorted([r for r in rows if r["what"] == what], key=key)[:a.top]
        print(f"\n{'highest-scoring false positives' if what == 'FP' else 'nearest missed boxes'} ({len(sel)} of {sum(1 for r in rows if r['what'] == what)}):")
        for r in sel:
            print(f"  {r['kind']:12s} {r['class']:11s} {r['scene']}/{r['index']:03d} ({r['x']:6.1f}, {r['y']:6.1f}) {r['range_m']:5.1f} m  "
                  f"{'score ' + str(r['score']) if what == 'FP' else 'size ' + r['size']}  {r['explained_by']}")
        if sel:
            print(f"    redraw the first: {sel[0]['redraw']}")
    return 0


# ------------------------------------------------------------------------------------------------ self-test
def selftest():
    import random
    import tempfile

    def box(g, x, y, s=None):
        b = {"group": g, "cx": float(x), "cy": float(y), "length": 4.0, "width": 2.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0}
        if s is not None:
            b["score"] = s
        return b
    gts = [box("vehicle", 10, 0), box("vehicle", 20, 5), box("vehicle", 30, 0), box("vehicle", 40, 0), box("vehicle", 50, 3), box("pedestrian", 15, 8)]
    dets = [box("vehicle", 10.3, 0, 0.9),                      # TP of gts[0]
            box("vehicle", 10.9, 0.4, 0.8),                    # duplicate on gts[0]
            box("vehicle", 20.2, 5.1, 0.7),                    # TP of gts[1]
            box("pedestrian", 30.1, 0.0, 0.8),                 # wrong class on a vehicle: FP wrong_class, and gts[2] a FN wrong_class
            box("vehicle", 40.4, 0.2, 0.2),                    # under min_score: gts[3] is a FN below_score
            box("vehicle", 55.0, 20.0, 0.95),                  # a ghost: FP isolated
            box("pedestrian", 15.2, 8.1, 0.6)]                 # TP of the pedestrian
    r = classify_frame(dets, gts, ("vehicle", "pedestrian"), thr=2.0, min_score=0.3)
    kinds = lambda lst: sorted(k for _, k, _ in lst)           # noqa: E731
    assert len(r["tp"]) == 3 and kinds(r["fp"]) == ["duplicate", "isolated", "wrong_class"], (len(r["tp"]), kinds(r["fp"]))
    assert kinds(r["fn"]) == ["below_score", "missed", "wrong_class"], kinds(r["fn"])
    fn = {k: g for g, k, _ in r["fn"]}
    assert fn["below_score"]["cx"] == 40 and fn["wrong_class"]["cx"] == 30 and fn["missed"]["cx"] == 50
    # 'taken': two boxes 1 m apart, one detection between them
    r2 = classify_frame([box("vehicle", 10.4, 0, 0.9)], [box("vehicle", 10, 0), box("vehicle", 11, 0)], ("vehicle",))
    assert len(r2["tp"]) == 1 and kinds(r2["fn"]) == ["taken"], r2
    # a ghost pedestrian next to a car that WAS found is a ghost, not a class confusion; next to a car nobody found it is the confusion; a duplicate wins over the confusion
    car = box("vehicle", 20, 0)
    r3 = classify_frame([box("vehicle", 20.1, 0, 0.9), box("pedestrian", 20.8, 0.3, 0.8)], [car], ("vehicle", "pedestrian"))
    assert kinds(r3["fp"]) == ["isolated"] and not r3["fn"], r3
    r4 = classify_frame([box("pedestrian", 20.8, 0.3, 0.8)], [car], ("vehicle", "pedestrian"))
    assert kinds(r4["fp"]) == ["wrong_class"] and kinds(r4["fn"]) == ["wrong_class"], r4
    r5 = classify_frame([box("vehicle", 20.1, 0, 0.9), box("vehicle", 20.6, 0.2, 0.7), box("pedestrian", 20.8, 0.3, 0.8)], [car], ("vehicle", "pedestrian"))
    assert sorted(kinds(r5["fp"])) == ["duplicate", "isolated"], r5                  # the pedestrian: the car is found, so a ghost; the second car: a duplicate
    # the counts equal the metric's matching on random worlds (same function, applied per class and frame)
    rng = random.Random(3)
    for _ in range(40):
        g = [box(rng.choice(("vehicle", "pedestrian")), rng.uniform(0, 60), rng.uniform(-20, 20)) for _ in range(rng.randint(0, 9))]
        d = [dict(box(rng.choice(("vehicle", "pedestrian")), rng.uniform(0, 60), rng.uniform(-20, 20), round(rng.uniform(0, 1), 1))) for _ in range(rng.randint(0, 9))]
        res = classify_frame(d, g, ("vehicle", "pedestrian"), 2.0, 0.3)
        want_tp = sum(sum(j >= 0 for j in BX.match([x for x in d if x["group"] == c and x["score"] >= 0.3], [x for x in g if x["group"] == c], 2.0)) for c in ("vehicle", "pedestrian"))
        n_det = sum(1 for x in d if x["score"] >= 0.3)
        assert len(res["tp"]) == want_tp and len(res["tp"]) + len(res["fp"]) == n_det and len(res["tp"]) + len(res["fn"]) == len(g), (res, d, g)
    # through the files: a run folder with records and a store
    with tempfile.TemporaryDirectory() as d_:
        run = os.path.join(d_, "runs", "x")
        os.makedirs(os.path.join(run, "eval", "suite", "quant", "eval_n3"))
        with open(os.path.join(run, "eval", "suite", "quant", "eval_n3", "frames.jsonl"), "w") as f:
            for sc, i in (("scene-1", 0), ("scene-2", 0), ("scene-1", 1)):
                f.write(json.dumps({"label": f"{sc}/{i:03d}", "scene": sc, "index": i, "key": f"tok{sc}{i}"}) + "\n")
        outside = [box("vehicle", 10, 15, s=0.9), box("vehicle", 65, 0, s=0.9)]                    # 56 deg off the axis, and beyond 60 m: outside the scored window
        keys = ["tokscene-10", "tokscene-20", "tokscene-11"]
        with open(os.path.join(run, "eval", "scene_records_n3.pkl"), "wb") as f:
            pickle.dump({"det": {"scene-1": {0.0: [(dets + outside, gts + [box("vehicle", 10, -15), box("vehicle", 70, 0)]), ([], [box("vehicle", 25, 2)])]},
                                 "scene-2": {0.0: [(dets, gts)]}}}, f)
        rows, totals, nf = error_rows(run, 3)
        assert nf == 3 and totals["vehicle"]["tp"] == 4 and sum(1 for x in rows if x["what"] == "FN") == 3 + 3 + 1, (totals, len(rows))
        assert all(x["x"] < 60 and abs(math.atan2(x["y"], x["x"])) <= 0.5637 for x in rows), "only the scored window"
        miss = [x for x in rows if x["scene"] == "scene-1" and x["index"] == 1]
        assert len(miss) == 1 and miss[0]["kind"] == "missed" and miss[0]["token"] == "tokscene-11" and "qual frame scene-1/001" in miss[0]["redraw"], miss
        assert not any(x["x"] in (10.0, 65.0, 70.0) for x in rows), "boxes outside the scored window are no errors"
        text = summaries(rows, totals)
        assert "scene-1" in text and "by range" in text and "false positives by score" in text
        with open(os.path.join(run, "eval", "scene_records_n3.pkl"), "rb") as f:                   # the keys of the records must be the store's: another order is refused
            rec = pickle.load(f)
        rec["meta"] = {"frames": keys}
        with open(os.path.join(run, "eval", "scene_records_n3.pkl"), "wb") as f:
            pickle.dump(rec, f)
        assert error_rows(run, 3)[2] == 3
        rec["meta"] = {"frames": keys[::-1]}
        with open(os.path.join(run, "eval", "scene_records_n3.pkl"), "wb") as f:
            pickle.dump(rec, f)
        try:
            error_rows(run, 3)
        except SystemExit:
            pass
        else:
            raise AssertionError("records of another frame order must be refused")
        rec["det"]["scene-2"][0.0].append(([], []))                                              # one more frame of boxes than the store lists
        with open(os.path.join(run, "eval", "scene_records_n3.pkl"), "wb") as f:
            pickle.dump(rec, f)
        try:
            error_rows(run, 3)
        except SystemExit:
            pass
        else:
            raise AssertionError("a count mismatch must be refused")
        os.rename(os.path.join(run, "eval", "suite", "quant", "eval_n3"), os.path.join(run, "eval", "suite", "quant", "eval_curves_fusion_n3"))
        assert len(load_frames(run, 3)) == 3, "the store name may carry the --curves / --fusion parts"
        assert len(load_frames(run, 3, store="eval_curves_fusion_n3")) == 3 and find_store(run, 3, store="eval_curves_fusion_n3").endswith("eval_curves_fusion_n3")
        try:
            find_store(run, 3, "_warm")
        except SystemExit:
            pass
        else:
            raise AssertionError("another tag must not find the store")
        os.makedirs(os.path.join(run, "eval", "suite", "quant", "eval_n4"))
        try:
            error_rows(run, 4)
        except SystemExit:
            pass
        else:
            raise AssertionError("a store without records must be refused")
    print("bev_errors selftest ok: FP duplicate / wrong class / isolated, FN below score / wrong class / missed / taken, counts equal the metric's matching, frame trace through the files")


if __name__ == "__main__":
    raise SystemExit(main())
