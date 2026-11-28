"""Tests of evaluation/metrics.py (its former self-test).

    python -P -m bev_fusion.tests.test_metrics      (from the problem folder, with its module path)
"""
import math

import numpy as np

from bev_fusion.evaluation.metrics import DIST_THRESHOLDS, DetMeter, SegMeter, TOLERANCES, TP_THRESHOLD, bin_label, cell_bins, flatten, format_table, gt_boxes_from_meta, in_front, ms_per_frame, range_bin, report, tp_errors, unflatten


def _selftest():
    from bev_fusion.net.model.lift_pd import Grid
    g = Grid()                                            # x 0..60, y -26..26, 0.5 m: 120 x 104 cells
    X, Y = g.nx, g.ny
    eq = lambda a, b, e=1e-9: abs(a - b) <= e                                                 # noqa: E731

    # 1 range bins of cells (by centre x) and boxes
    cb = cell_bins((X, Y), g)
    assert cb[0, 0] == 0 and cb[39, 5] == 0 and cb[40, 5] == 1 and cb[79, 0] == 1 and cb[80, 0] == 2 and cb[119, 0] == 2
    assert [(cb[:, 0] == i).sum() for i in range(3)] == [40, 40, 40]
    assert range_bin(19.99) == 0 and range_bin(20.0) == 1 and range_bin(45.0) == 2
    assert range_bin(-1.0) is None and range_bin(60.0) is None

    # 2 segmentation: perfect -> 1 at every tolerance; a 1-cell shift (0.5 m) -> 90 / 110 at 0, 1 at 0.5 and 1.0 m
    gt = np.zeros((X, Y))
    gt[10:20, 30:40] = 1.0                                # a 10 x 10 cell block in 0-20 m
    gt[60:64, 50:60] = 1.0                                # a 4 x 10 block in 20-40 m
    m = SegMeter(g)
    m.add({"vehicle": gt}, {"vehicle": gt})
    r = m.result()[0.0]["vehicle"]["all"]
    for b in ("all", "0-20", "20-40"):
        assert all(eq(r[b]["iou@%g" % t], 1.0) for t in TOLERANCES), r[b]
    assert math.isnan(r["40-60"]["iou@0"]) and r["all"]["n_gt"] == 140
    shifted = np.roll(gt, 1, axis=0)
    m = SegMeter(g)
    m.add({"vehicle": shifted}, {"vehicle": gt})
    r = m.result()[0.0]["vehicle"]["all"]
    assert eq(r["0-20"]["iou@0"], 90 / 110) and eq(r["0-20"]["iou@0.5"], 1.0) and eq(r["0-20"]["iou@1"], 1.0)
    iou_shift = r["0-20"]["iou@0"]
    # tolerance radius, metres -> cells (0.5 m cells: 0.5 m = 1 cell, 1 m = 2 cells; a DISC, so a diagonal step of 1 cell is outside 0.5 m but inside 0.75 m and
    # 1 m; a 2-cell step is inside 1 m only). A 10 x 10 block moved by (1, 1) and by (2, 0) cells; the expectation is a brute-force count straight from the
    # definition (a predicted cell is a hit if a GT cell centre lies within t metres of its centre, and the other way round) and numbers worked out by hand
    # (diagonal: 99 of 100 cells hit at 0.5 m, the corner pair misses; 2-cell step: 90 of 100 at 0.5 m)
    def tol_iou(P, G, t):
        pc, gc = np.argwhere(P) * g.res, np.argwhere(G) * g.res
        d = np.sqrt(((pc[:, None] - gc[None]) ** 2).sum(-1))
        miss_p, miss_g = np.sum(d.min(1) > t + 1e-9), np.sum(d.min(0) > t + 1e-9)
        tp = (len(pc) - miss_p + len(gc) - miss_g) / 2
        return tp / (tp + miss_p + miss_g)
    blk = np.zeros((X, Y))
    blk[10:20, 30:40] = 1.0
    tols = (0.0, 0.25, 0.5, 0.75, 1.0)
    for shift, hand in (((1, 1), {0.0: 81 / 119, 0.25: 81 / 119, 0.5: 99 / 101, 0.75: 1.0, 1.0: 1.0}),
                        ((2, 0), {0.0: 80 / 120, 0.25: 80 / 120, 0.5: 90 / 110, 0.75: 90 / 110, 1.0: 1.0})):
        moved = np.roll(blk, shift, axis=(0, 1))
        m = SegMeter(g, tolerances=tols)
        m.add({"vehicle": moved}, {"vehicle": blk})
        r = m.result()[0.0]["vehicle"]["all"]["all"]
        for t in tols:
            assert eq(r["iou@%g" % t], tol_iou(moved > 0, blk > 0, t)) and eq(r["iou@%g" % t], hand[t]), (shift, t, r["iou@%g" % t], hand[t])
    # mask: false positives on undefined cells do not count
    fp = gt.copy()
    fp[100:110, 0:10] = 1.0
    mask = np.ones((X, Y), bool)
    mask[95:, :20] = False
    m = SegMeter(g)
    m.add({"vehicle": fp}, {"vehicle": gt}, mask=mask)
    assert eq(m.result()[0.0]["vehicle"]["all"]["all"]["iou@0"], 1.0)
    # known / unknown: the counts split exactly (mass 0 in the far half)
    mass = np.ones((X, Y))
    mass[:, :52] = 0.0
    m = SegMeter(g)
    m.add({"vehicle": shifted}, {"vehicle": gt}, mass=mass, dt=0.5)
    c = {s: m.c[(0.5, "vehicle", s, "all")] for s in ("all", "known", "unknown")}
    assert np.array_equal(c["all"]["cnt"], c["known"]["cnt"] + c["unknown"]["cnt"])
    assert c["known"]["n_gt"] == 32 and c["unknown"]["n_gt"] == 108

    # 3 lines: distance field of a line along x at y = 0.1 m; perfect -> F1 1 and field error 0; a lateral 1-cell
    #   shift -> F1 0 at 0 m, 1 at 0.5 m; field error on the 4 cells with GT distance < 1 m (y = -0.75 .. 0.75):
    #   |1.35 - 0.85|, |0.85 - 0.35|, |0.35 - 0.15|, |0.15 - 0.65| -> 1.7 / 4 = 0.425 m
    yc = g.y_min + (np.arange(Y) + 0.5) * g.res
    field = np.minimum(np.abs(yc - 0.1), 2.0)[None].repeat(X, 0)
    m = SegMeter(g)
    m.add_lines({"lane_divider": field}, {"lane_divider": field})
    r = m.result()[0.0]["lane_divider:zc"]["all"]["all"]
    assert all(eq(r["f1@%g" % t], 1.0) for t in TOLERANCES) and r["field_mae"] == 0 and r["n_gt"] == X
    m = SegMeter(g)
    m.add_lines({"lane_divider": np.minimum(np.abs(yc - 0.6), 2.0)[None].repeat(X, 0)}, {"lane_divider": field})
    r = m.result()[0.0]["lane_divider:zc"]["all"]["all"]
    assert r["f1@0"] == 0 and eq(r["f1@0.5"], 1.0) and eq(r["field_mae"], 0.425), r

    # 4 detection: perfect -> AP 1 (both), TP errors 0
    gts = [[{"group": "car", "cx": 10.0, "cy": 1.0, "length": 4.5, "width": 1.8, "yaw": 0.3, "vx": 5.0, "vy": 0.0},
            {"group": "car", "cx": 30.0, "cy": -2.0, "length": 4.5, "width": 1.8, "yaw": -1.0, "vx": 0.0, "vy": 0.0},
            {"group": "pedestrian", "cx": 15.0, "cy": 3.0, "radius": 0.4, "vx": 0.0, "vy": 1.2}],
           [{"group": "car", "cx": 50.0, "cy": 4.0, "length": 4.5, "width": 1.8, "yaw": 3.0, "vx": 1.0, "vy": 1.0},
            {"group": "car", "cx": 10.0, "cy": 20.0, "length": 4.5, "width": 1.8, "yaw": 0.0, "vx": 0.0, "vy": 0.0}]]
    assert not in_front(gts[1][1])                        # 63 deg off the axis: outside the front field of view
    dm = DetMeter()
    for gs in gts:
        dm.add([dict(b, score=0.9) for b in gs], gs)
    r = dm.result()[0.0]
    for cls in ("car", "pedestrian"):
        row = r[cls]["all"]
        assert all(eq(row["ap@%g" % t], 1.0) and eq(row["ap_all@%g" % t], 1.0) for t in DIST_THRESHOLDS), row
        assert all(eq(row[k], 0.0, 1e-6) for k in ("ate", "ase", "ave")) and eq(row["biou"], 1.0, 1e-6)
    assert r["car"]["all"]["n_gt"] == 3 and r["car"]["0-20"]["n_gt"] == 1 and r["car"]["40-60"]["n_gt"] == 1
    assert eq(r["car"]["all"]["aoe"], 0.0) and math.isnan(r["pedestrian"]["all"]["aoe"]) and eq(r["mean"]["all"]["mAP"], 1.0)
    # empty prediction -> AP 0
    dm = DetMeter()
    dm.add([], gts[0])
    assert all(dm.result()[0.0][c]["all"]["ap@%g" % t] == 0.0 for c in ("car", "pedestrian") for t in DIST_THRESHOLDS)
    # a box rotated by 90 deg: AOE pi / 2; ASE 0 (sizes aligned); BIoU by hand 1.8^2 / (2 * 4.5 * 1.8 - 1.8^2) = 0.25
    g0 = gts[0][0]
    e = tp_errors(dict(g0, yaw=g0["yaw"] + math.pi / 2, vx=g0["vx"] + 1.0), g0)
    assert eq(e["aoe"], math.pi / 2) and eq(e["ase"], 0.0) and eq(e["biou"], 0.25, 1e-6) and eq(e["ave"], 1.0)
    # size 4.0 x 2.0 against 4.5 x 1.8: ASE = 1 - 4.0 * 1.8 / (8.1 + 8.0 - 7.2); circle 0.5 against 0.4: 1 - 0.64
    assert eq(tp_errors(dict(g0, length=4.0, width=2.0), g0)["ase"], 1 - 7.2 / 8.9)
    assert eq(tp_errors(dict(gts[0][2], radius=0.5), gts[0][2])["ase"], 0.36)
    # known two-detection ranking: 2 GT, a false positive (0.9) above a true positive (0.8): precision (0, 0.5),
    #   recall (0, 0.5): all-point AP 0.5 * 0.5 = 0.25; nuScenes p(r) = r up to 0.5: sum_{k=11}^{50} (k / 100 - 0.1)
    #   = 8.2 over 90 bins / 0.9 = 8.2 / 81
    two = [{"group": "car", "cx": 10.0, "cy": 0.0, "length": 4.5, "width": 1.8, "yaw": 0.0, "vx": 0.0, "vy": 0.0},
           {"group": "car", "cx": 25.0, "cy": 0.0, "length": 4.5, "width": 1.8, "yaw": 0.0, "vx": 0.0, "vy": 0.0}]
    dm = DetMeter()
    dm.add([dict(two[0], cx=14.0, score=0.9), dict(two[1], cx=25.3, score=0.8)], two)
    row = dm.result()[0.0]["car"]["all"]
    assert eq(row["ap_all@1"], 0.25) and eq(row["ap@1"], 8.2 / 81) and eq(row["ate"], 0.3) and row["n_tp"] == 1, row
    assert eq(row["ap_all@4"], 1.0) and eq(row["ap@4"], 1.0)  # at 4 m the 4 m-off detection matches too
    ap_rank = row["ap@1"]
    # mean errors: a class with GT and no true positive counts WORST_ERROR (1.0), it is not left out of the mean (a missed class must not improve mATE).
    #   car: found 0.3 m off, sizes / yaw / velocity equal (ATE 0.3, others 0); truck: GT, never detected; pedestrian (circle): found 1.5 m off (ATE 1.5)
    #   -> mATE (0.3 + 1 + 1.5) / 3, mASE and mAVE (0 + 1 + 0) / 3; AOE: car 0, truck 1, the pedestrian's AOE is undefined (circle) -> 0.5
    car = {"group": "car", "cx": 10.0, "cy": 1.0, "length": 4.5, "width": 1.8, "yaw": 0.3, "vx": 5.0, "vy": 0.0}
    truck = {"group": "truck", "cx": 30.0, "cy": -2.0, "length": 8.0, "width": 2.5, "yaw": 0.0, "vx": 0.0, "vy": 0.0}
    ped = {"group": "pedestrian", "cx": 15.0, "cy": 3.0, "radius": 0.4, "vx": 0.0, "vy": 1.2}
    dm = DetMeter(classes=["car", "truck", "pedestrian"])
    dm.add([dict(car, cx=10.3, score=0.9), dict(ped, cx=16.5, score=0.9)], [car, truck, ped])
    mean = dm.result()[0.0]["mean"]["all"]
    assert eq(mean["mATE"], (0.3 + 1.0 + 1.5) / 3) and eq(mean["mASE"], 1 / 3) and eq(mean["mAVE"], 1 / 3) and eq(mean["mAOE"], 0.5), mean
    # the pedestrian is not found either: it counts 1.0 in ATE / ASE / AVE, its AOE stays undefined
    dm = DetMeter(classes=["car", "pedestrian"])
    dm.add([dict(car, cx=10.3, score=0.9)], [car, ped])
    mean = dm.result()[0.0]["mean"]["all"]
    assert eq(mean["mATE"], 0.65) and eq(mean["mASE"], 0.5) and eq(mean["mAVE"], 0.5) and eq(mean["mAOE"], 0.0), mean
    dm = DetMeter(classes=["car", "truck"])                         # a class without any GT is not in the mean (the truck has no GT here)
    dm.add([dict(car, cx=10.3, score=0.9)], [car])
    assert eq(dm.result()[0.0]["mean"]["all"]["mATE"], 0.3)

    # 5 dt-keyed results: three dt, flatten -> unflatten round trip, report and table
    dm, sm = DetMeter(), SegMeter(g)
    for dt in (0.0, 0.5, -0.5):
        dm.add([dict(b, score=0.9) for b in gts[0]], gts[0], dt=dt)
        sm.add({"vehicle": shifted}, {"vehicle": gt}, mass=mass if dt else None, dt=dt)
    nested = {"det": dm.result(), "seg": sm.result()}
    assert sorted(nested["det"]) == [-0.5, 0.0, 0.5] and "unknown" in nested["seg"][0.5]["vehicle"]
    back = unflatten(flatten(nested))
    fa, fb = flatten(nested), flatten(back)
    assert sorted(back["det"]) == [-0.5, 0.0, 0.5] and fa.keys() == fb.keys()
    assert all((fa[k] == fb[k]) or (fa[k] != fa[k] and fb[k] != fb[k]) for k in fa)
    assert "det/dt-0.50/car/0-20/ap@0.5" in fa and "seg/dt+0.50/vehicle/known/0-20/iou@0.5" in fa

    # 5b boxes without any lidar / radar point are not GT (as in the nuScenes detection evaluation); opt-out flag
    mb = [{"group": "vehicle", "size": [1.8, 4.5, 1.5], "center": [10.0, 0.0, 0.0], "vel": [0.0, 0.0], "yaw": 0.0, "observed": True},
          {"group": "vehicle", "size": [1.8, 4.5, 1.5], "center": [30.0, 0.0, 0.0], "vel": [0.0, 0.0], "yaw": 0.0, "observed": False},
          {"group": "vehicle", "size": [1.8, 4.5, 1.5], "center": [40.0, 0.0, 0.0], "vel": [0.0, 0.0], "yaw": 0.0}]
    assert len(gt_boxes_from_meta(mb)) == 2 and len(gt_boxes_from_meta(mb, observed_only=False)) == 3

    # 5c the per-frame rows (frame_rows / reduce_rows) give the numbers of the ORIGINAL global greedy matching (oracle below: the
    #    frozen copy of the former DetMeter code in Evaluation/selftest_comparison.py, independent of the core that DetMeter calls now)
    #    on random frames with ties, empty frames, circles and boxes outside the field of view; the reduction is order-independent;
    #    `only` computes just the asked scopes
    from Evaluation.selftest_comparison import _REF

    def oracle_eval(frames):
        row = {}
        for t in DIST_THRESHOLDS:
            mm, n_gt = _REF.match(frames, t)
            fl = [x[1] for x in mm]
            row["ap@%g" % t], row["ap_all@%g" % t] = _REF.ap_nuscenes(fl, n_gt), _REF.ap_all_points(fl, n_gt)
            if t == TP_THRESHOLD:
                errs = [_REF.tp_errors(d, g) for _, ok, d, g in mm if ok]
                for k in ("ate", "ase", "aoe", "ave", "biou"):
                    v = [e[k] for e in errs if e[k] == e[k]]
                    row[k] = float(np.mean(v)) if v else float("nan")
                row["n_tp"] = len(errs)
        row["n_gt"] = sum(len(gg) for _, gg in frames)
        row["n_det"] = sum(len(dd) for dd, _ in frames)
        return row
    rng = np.random.default_rng(3)
    classes = ["vehicle", "pedestrian"]
    dm = DetMeter(classes=classes)
    for dt in (0.0, 0.5):
        for f in range(30):
            gs = []
            for _ in range(int(rng.integers(0, 6))):
                cls = classes[int(rng.integers(0, 2))]
                b = {"group": cls, "cx": float(rng.uniform(0, 62)), "cy": float(rng.uniform(-30, 30)), "vx": float(rng.normal()),
                     "vy": float(rng.normal())}
                b.update({"radius": 0.4} if cls == "pedestrian" else {"length": 4.5, "width": 1.8, "yaw": float(rng.uniform(-3, 3))})
                gs.append(b)
            ds = [dict(b, cx=b["cx"] + float(rng.normal(0, 1.2)), cy=b["cy"] + float(rng.normal(0, 1.2)),
                       score=float(rng.choice([0.9, 0.9, 0.5, rng.uniform()]))) for b in gs if rng.uniform() < 0.8]
            ds += [dict(gs[0] if gs else {"group": "vehicle", "length": 4.5, "width": 1.8, "yaw": 0.0, "vx": 0.0, "vy": 0.0},
                        cx=float(rng.uniform(0, 60)), cy=float(rng.uniform(-20, 20)), score=float(rng.uniform())) for _ in range(3)]
            dm.add(ds, gs, dt=dt)
    new = dm.result()
    for dt, frames in dm.frames.items():
        for cls in classes:
            fc = [([d for d in ds if d["group"] == cls], [g for g in gs if g["group"] == cls]) for ds, gs in frames]
            want = {"all": oracle_eval(fc)}
            for i in range(3):
                inb = lambda bs: [b for b in bs if range_bin(b["cx"]) == i]                         # noqa: E731
                want[bin_label(i)] = oracle_eval([(inb(ds), inb(gs)) for ds, gs in fc])
            for sc, row in want.items():
                assert list(row) == list(new[dt][cls][sc]), (list(row), list(new[dt][cls][sc]))
                for k, v in row.items():
                    w = new[dt][cls][sc][k]
                    assert (v != v and w != w) or abs(v - w) < 1e-12, (dt, cls, sc, k, v, w)
        rows = [dm.frame_rows(ds, gs, classes) for ds, gs in frames]
        part = dm.reduce_rows(rows, classes, only={("vehicle", "all")})
        assert list(part) == ["vehicle"] and part["vehicle"]["all"] == new[dt]["vehicle"]["all"]
    # without tied scores the frame order does not matter (with ties the greedy order is 'frame order, then detection order')
    uni = DetMeter(classes=classes)
    for ds, gs in dm.frames[0.0]:
        uni.add([dict(d, score=float(rng.uniform())) for d in ds], gs)
    rows = [uni.frame_rows(ds, gs, classes) for ds, gs in uni.frames[0.0]]
    a, b = flatten({"x": uni.reduce_rows(rows, classes)}), flatten({"x": uni.reduce_rows(rows[::-1], classes)})
    assert a.keys() == b.keys() and all((a[k] == b[k]) or (a[k] != a[k] and b[k] != b[k]) or abs(a[k] - b[k]) < 1e-12 for k in a)
    # a detection without a score (nan: a diverged model) is refused at its frame, not ranked last
    nan_det = {"group": "vehicle", "cx": 5.0, "cy": 0.0, "length": 4.5, "width": 1.8, "yaw": 0.0, "vx": 0.0, "vy": 0.0, "score": float("nan")}
    try:
        DetMeter(classes=classes).frame_rows([nan_det], [dict(nan_det, score=1.0)], classes)
    except ValueError:
        pass
    else:
        raise AssertionError("a nan score must be refused")

    # 6 latency helper (CPU callable here; timing itself only on an idle GPU)
    import torch
    x = torch.randn(1, 8, 64, 64)
    conv = torch.nn.Conv2d(8, 8, 3, padding=1)
    lat = ms_per_frame(conv, x, n=10, warmup=2)
    assert lat["median"] > 0 and lat["n"] == 10
    flat = report(seg=sm, det=dm, ms={"conv": lat})
    table = format_table({k: v for k, v in flat.items() if "/dt+0.00/" in k and "/all/" in k or k.startswith("ms/")},
                         columns=["iou@0", "iou@0.5", "iou@1", "ap@1", "ap_all@1", "ate", "ase", "aoe", "ave",
                                  "median", "n_gt"])
    print(table)
    print(f"metrics selftest ok: range bins of cells and boxes, IoU perfect 1 / 1-cell shift {iou_shift:.3f} at 0 m "
          f"and 1 at 0.5 / 1 m, tolerance disc against a brute-force count (diagonal and 2-cell shifts), mean errors of a missed class = 1.0, "
          f"mask, known / unknown split, line zero-crossing F1 (perfect 1, shift 0 / 1, field "
          f"error 0.425 m), detection perfect AP 1 and errors 0, empty AP 0, 90 deg AOE pi/2 ASE 0 BIoU 0.25, ASE "
          f"by hand (box 0.191, circle 0.36), two-detection AP nuScenes {ap_rank:.4f} = 8.2/81 / all-point 0.25, "
          f"front field of view, dt-keyed round trip ({len(fa)} keys), per-frame rows = the original greedy matching on random "
          f"frames (ties, circles, empty frames), latency {lat['median']:.2f} ms (CPU)")


if __name__ == "__main__":
    _selftest()
