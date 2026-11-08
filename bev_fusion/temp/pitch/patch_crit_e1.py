p = "evaluation/bev_errors.py"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


# ---- docstring: precedence, the missed text, the window, tables vs pictures, no bev_suite subcommand
rep('''    python evaluation/bev_errors.py RUN [--n 1000] [--tag ""] [--min-score 0.3] [--thr 2.0] [--top 20] [--class vehicle] [--out FILE.csv]     (also: bev_suite.py RUN errors ...)''',
    '''    python evaluation/bev_errors.py RUN [--n 1000|all] [--tag ""] [--store NAME] [--min-score 0.3] [--thr 2.0] [--top 20] [--class vehicle] [--check] [--out FILE.csv]''')
rep('''Kinds (the reason an error is what it is):
  FP  duplicate       a second detection on a ground-truth box that another detection already has
      wrong_class     a ground-truth box of ANOTHER class lies within `thr` m (class confusion)
      isolated        nothing of the ground truth within `thr` m (a ghost; the nearest box is given)
  FN  below_score     a detection of its class within `thr` m exists but its score is under `min-score` (a threshold effect, not a miss of the network)
      wrong_class     a detection of another class sits on it
      missed          no detection within 2 x `thr` m of it at all
      taken           its neighbour detection went to another box (two objects, one detection)
The table also lists''', '''Kinds (the reason an error is what it is; the first that applies wins, in the order listed):
  FP  duplicate       a ground-truth box of its class lies within `thr` m but another detection has it
      wrong_class     a ground-truth box of ANOTHER class within `thr` m that no detection of its own class found (the class confusion: that box is a miss of its class)
      isolated        nothing of the ground truth within `thr` m (a ghost; the nearest box is given)
  FN  taken           a detection of its class lies within `thr` m but went to another box (two objects, one detection)
      below_score     a detection of its class within `thr` m exists but its score is under `min-score` (a threshold effect, not a miss of the network)
      wrong_class     a detection of another class sits on it and is itself a false positive (the class confusion)
      missed          nothing of the kinds above (`explained_by` then names the nearest detection within 2 x `thr`, if any)
The tables count every annotated box of the scored window, like the plain `det` numbers; the pictures (bev_views) apply the D38 'no care' rule on top (boxes with less than half of
their footprint in the observed area are neither right nor false nor missed there), so a count on a picture can be smaller. The table also lists''')

# ---- classify_frame: two passes (matched status of the other classes)
a = s.index("def classify_frame(")
b = s.index("def find_store(")
s = s[:a] + '''def classify_frame(dets, gts, groups, thr=2.0, min_score=0.3):
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


''' + s[b:]

# ---- store: --store, n may be 'all'
rep('''def find_store(run, n, tag=""):
    """the suite store of the evaluation: eval<tag>_n<N>, with the optional _curves / _fusion / _vel parts the flags of `eval` add to its name; refuses none or several"""
    import re
    base = os.path.join(run, "eval", "suite", "quant")''', '''def find_store(run, n, tag="", store=None):
    """the suite store of the evaluation: eval<tag>_n<N>, with the optional _curves / _fusion / _vel parts the flags of `eval` add to its name (or the one named by `store`);
    refuses none or several: with several, the pickle of boxes belongs to the latest evaluation of that name, so say which (--store NAME)"""
    import re
    base = os.path.join(run, "eval", "suite", "quant")
    if store:
        if not os.path.exists(os.path.join(base, store, "frames.jsonl")):
            raise SystemExit(f"no store {os.path.join(base, store)}")
        return os.path.join(base, store)''')
rep('''def load_frames(run, n, tag=""):
    """[(scene, index, token, label)] in the order the suite evaluated them, from the suite store of the evaluation"""
    with open(os.path.join(find_store(run, n, tag), "frames.jsonl")) as f:''', '''def load_frames(run, n, tag="", store=None):
    """[(scene, index, token, label)] in the order the suite evaluated them, from the suite store of the evaluation"""
    with open(os.path.join(find_store(run, n, tag, store), "frames.jsonl")) as f:''')
rep('''def load_boxes(run, n, tag="", dt=0.0):
    """{(scene, k): (dets, gts)} with k the position of the frame in its scene's list, from scene_records<tag>_n<N>.pkl"""''', '''def load_boxes(run, n, tag="", dt=0.0):
    """({(scene, k): (dets, gts)}, meta): k the position of the frame in its scene's list, from scene_records<tag>_n<N>.pkl; meta = its `frames` (the keys, in order) and GT signature"""''')
rep('''    with open(path, "rb") as f:
        det = pickle.load(f)["det"]
    return {(s, k): fr for s, per in det.items() for k, fr in enumerate(per.get(dt, []))}''', '''    with open(path, "rb") as f:
        rec = pickle.load(f)
    return {(s, k): fr for s, per in rec["det"].items() for k, fr in enumerate(per.get(dt, []))}, rec.get("meta") or {}''')
rep('''def error_rows(run, n, tag="", groups=("vehicle", "pedestrian", "two_wheeler"), thr=2.0, min_score=0.3, dt=0.0):
    """-> (rows, totals): one dict per error (kind FP / FN) with the frame it is in and the command that redraws it; totals {class: {"tp","fp","fn"}} and n_frames"""
    frames, boxes = load_frames(run, n, tag), load_boxes(run, n, tag, dt)
    seen, rows, tot = Counter(), [], defaultdict(Counter)
    if len(frames) != sum(1 for _ in boxes):
        raise SystemExit(f"{len(frames)} frames in the suite store but {len(boxes)} frames of boxes in the records: they are not from the same evaluation (--tag / --n?)")''', '''def error_rows(run, n, tag="", groups=("vehicle", "pedestrian", "two_wheeler"), thr=2.0, min_score=0.3, dt=0.0, store=None):
    """-> (rows, totals, n_frames): one dict per error (kind FP / FN) with the frame it is in and the command that redraws it; totals {class: {"tp","fp","fn"}}. The records and
    the store must be one evaluation: same number of frames, and when the records carry their frame keys (every evaluation since 2026-10-31) the same keys in the same order."""
    frames, (boxes, meta) = load_frames(run, n, tag, store), load_boxes(run, n, tag, dt)
    seen, rows, tot = Counter(), [], defaultdict(Counter)
    if len(frames) != sum(1 for _ in boxes):
        raise SystemExit(f"{len(frames)} frames in the suite store but {len(boxes)} frames of boxes in the records: they are not from the same evaluation (--tag / --n / --store?)")
    if meta.get("frames") is not None and list(meta["frames"]) != [f[2] for f in frames]:
        raise SystemExit("the frame keys stored with the records differ from the suite store's frame list (another evaluation of the same name: --store NAME, or re-evaluate)")
    if meta.get("frames") is None:
        print("WARNING: the records carry no frame keys (evaluated before 2026-10-31): the frame of an error is taken from the order only", flush=True)''')
rep('''    for scene, index, token, label in frames:
        k = seen[scene]''', '''    for scene, index, token, label in frames:
        k = seen[scene]
        if (scene, k) not in boxes:
            raise SystemExit(f"the records hold no frame {k} of {scene} (the suite store lists more frames of it)")''')

# ---- range by cx like the metric
rep('''    rng = Counter((r["what"], _bin(r["range_m"], RANGE_BINS)) for r in rows)
    out.append("by range (the distance of the box from the car): "''', '''    rng = Counter((r["what"], _bin(r["x"], RANGE_BINS)) for r in rows)
    out.append("by range (x ahead of the car, the bins of the metric): "''')

# ---- CLI
rep('''    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--tag", default="", help="the tag of the evaluation (eval --warm writes _warm)")''', '''    ap.add_argument("--n", default="1000", help="the n of the evaluation (a number, or all)")
    ap.add_argument("--tag", default="", help="the tag of the evaluation (eval --warm writes _warm)")
    ap.add_argument("--store", default=None, help="the suite store folder name when several fit (eval_curves_fusion_n1000, ...)")''')
rep('''        bad = check_against_card(a.run.rstrip("/"), a.n, a.tag, groups, a.thr)''', '''        bad = check_against_card(a.run.rstrip("/"), a.n, a.tag, groups, a.thr, a.store)''')
rep('''    rows, totals, nf = error_rows(a.run.rstrip("/"), a.n, a.tag, groups, a.thr, a.min_score)
    out = a.out or os.path.join(a.run, "eval", f"errors{a.tag}_n{a.n}.csv")''', '''    if a.check and a.thr != 2.0:
        ap.error("--check compares with the card's n_tp, which is counted at 2.0 m")
    rows, totals, nf = error_rows(a.run.rstrip("/"), a.n, a.tag, groups, a.thr, a.min_score, store=a.store)
    out = a.out or os.path.join(a.run, "eval", f"errors{a.tag}_n{a.n}.csv")''')
rep('''def check_against_card(run, n, tag="", groups=("vehicle", "pedestrian", "two_wheeler"), thr=2.0):''', '''def check_against_card(run, n, tag="", groups=("vehicle", "pedestrian", "two_wheeler"), thr=2.0, store=None):''')
rep('''    _, tot, _ = error_rows(run, n, tag, groups, thr, 0.0)
    bad = []''', '''    _, tot, _ = error_rows(run, n, tag, groups, thr, 0.0, store=store)
    bad = []''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
