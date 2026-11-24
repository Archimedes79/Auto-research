"""Checks of the junction-free rows (bev_junction.py), kept apart so that an edit of a test or of the reproduction does not change the evaluator hash.

    python evaluation/bev_junction_check.py                       synthetic known answers (CPU, no data, seconds)
    python evaluation/bev_junction_check.py --reproduce [--n N]   the real-data check on the 246 stored frames of the recall test (CPU, 20 s)

`--reproduce` reads the UNTRACKED reference folder `runs/junction_ref/` (`rp_frames.json` + `rp_out/`, moved out of temp/ by D140) (the stored dt-0 probability maps and per-frame counts of the t20 network) and the GT cache
`data/cache/gt/`: it cannot run on a clean clone, and says so. It re-counts the lane divider IoU of the recall test with the packed GtGenerator masks and compares with
`records/intersection_iou_result.txt` (made with a PIL raster of the map's intersection polygons, which differs from the exact mask in about 1.6 % of the edge cells).
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in (ROOT, os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from evaluation.bev_junction import MASKS, RASTER, JunctionMasks, junction_free  # noqa: E402

STORED = {"fovj": (0.2204, 0.5091, 0.6410), "fovjb": (0.2223, 0.5141, 0.6472)}      # records/intersection_iou_result.txt, variants "both" and "both+band" (the map raster, PIL), IoU@0 / 0.5 / 1


def _fake_store(root, data="d1", raster=RASTER, borders=tuple(MASKS.values()), cells=None):
    """a lane store folder with packed mask caches as GtGenerator writes them (one frame, token "a"): the mask of border b has the cells cells[b] set"""
    os.makedirs(os.path.join(root, "masks"), exist_ok=True)
    for b in borders:
        m = np.zeros((120, 104), np.uint8)
        for c in (cells or {}).get(b, [(3, 7)]):
            m[c] = 1
        np.savez(os.path.join(root, "masks", f"intersection_b{b:g}.npz"), tokens=np.array(["a"]), bits=np.packbits(m.ravel())[None], shape=np.array([120, 104]),
                 border_m=np.array(b), kind=np.array("intersection"), source=np.array("fake"), raster=np.array(raster), data=np.array(data))
    return root


def check_masks():
    """JunctionMasks: refuses early (before any network is loaded) a missing cache file of either border and a rasteriser that is not the validated one; reads the cell [ix, iy] it was given;
    its mask id (describe) changes with the lane store, the data the cache was made from and the borders"""
    with tempfile.TemporaryDirectory() as t:
        def refused(store, exc, text):
            try:
                JunctionMasks(store)
            except exc as e:
                assert text in str(e), str(e)
                return
            raise AssertionError(f"{store}: must be refused with {text!r}")
        refused(os.path.join(t, "empty"), FileNotFoundError, "generate-bev")
        refused(_fake_store(os.path.join(t, "half"), borders=(0.0,)), FileNotFoundError, "intersection_b0.5.npz")             # one of the two borders missing
        refused(_fake_store(os.path.join(t, "old"), raster="0"), ValueError, "rasteriser 0")
        good = _fake_store(os.path.join(t, "good"), cells={0.0: [(3, 7)], 0.5: [(3, 7), (4, 7)]})
        jm = JunctionMasks(good)
        m = jm({"meta": {"sample": "a"}})
        assert set(m) == set(MASKS) and m["fovj"].dtype == bool and m["fovj"].shape == (120, 104)
        assert [tuple(c) for c in np.argwhere(m["fovj"])] == [(3, 7)] and [tuple(c) for c in np.argwhere(m["fovjb"])] == [(3, 7), (4, 7)], "[ix, iy], the border of each mask"
        try:
            jm({"meta": {"sample": "missing"}})
            raise AssertionError("a keyframe without a mask must be an error")
        except KeyError as e:
            assert "missing" in str(e)
        d = jm.describe()
        assert d["lane_store"] == "good" and d["raster"] == RASTER and d["borders_m"] == dict(MASKS) and d["cache_data"] == {"fovj": "d1", "fovjb": "d1"}, d
        other = JunctionMasks(_fake_store(os.path.join(t, "other"), data="d2")).describe()
        assert other != d and JunctionMasks(good).describe() == d, "the mask id differs with the store and the data of the cache, not between two readers of the same one"


def selftest():
    """synthetic known answers: the rule (what is removed, and from where), the dilation not carrying paint out of the junction in either direction, the add_sample rows, the mask reader"""
    from evaluation.bev_task import add_sample
    from evaluation.metrics import SegMeter
    grid = type("G", (), {"x_min": 0.0, "res": 0.5})()
    X, Y = 40, 30
    gt = np.zeros((X, Y), np.float32)
    gt[:, 10] = 1                                                      # a line along x at iy = 10 ...
    inter = np.zeros((X, Y), bool)
    inter[10:20, 5:15] = True                                          # ... crossing a junction (ix 10 .. 19) ...
    pred = gt.copy()
    pred[10:20, 10] = 0                                                # ... that the network does not draw inside the junction (matches the missing label: nothing there)
    pred[10:20, 12] = 1                                                # ... and a line beside it inside the junction: a false cell per row, where the GT has no label (iy 12 is 2 cells away)
    gt_in = gt.copy()
    gt_in[10:20, 10] = 0                                               # the GT has no paint inside the junction
    fov = np.ones((X, Y), bool)
    m = {"all": SegMeter(grid), "j": SegMeter(grid)}
    m["all"].add({"lane_divider": pred}, {"lane_divider": gt_in}, mask=fov)
    p, g, f = junction_free({"lane_divider": pred, "other": pred}, {"lane_divider": gt_in, "other": gt_in}, fov, inter)
    assert sorted(p) == ["lane_divider"], "only the divider layers get the rows"
    assert not p["lane_divider"][inter].any() and not g["lane_divider"][inter].any() and not f[inter].any() and f[~inter].all(), "the junction cells leave P, G and the scored cells"
    assert np.array_equal(p["lane_divider"][~inter], pred[~inter]) and np.array_equal(g["lane_divider"][~inter], gt_in[~inter]), "outside the junction nothing changes"
    m["j"].add(p, g, mask=f)
    c = m["all"].result()[0.0]["lane_divider"]["all"]["all"], m["j"].result()[0.0]["lane_divider"]["all"]["all"]
    assert c[0]["iou@0"] == 30 / 40, c[0]                              # plain: P = 40 cells (30 right + 10 false inside the junction), G = 30 cells
    assert c[1]["iou@0"] == 1.0 and c[1]["n_gt"] == 30, c[1]           # junction-free: the 10 false cells are inside and leave, the 30 right ones remain
    # the dilation does not carry the inside out: a predicted cell inside the junction 1 cell from the border must not count as a hit for a GT cell outside at tolerance 0.5 m
    gt2, pred2 = np.zeros((X, Y), np.float32), np.zeros((X, Y), np.float32)
    gt2[9, 10] = 1                                                     # a GT cell just outside the junction (ix 9) ...
    pred2[10, 10] = 1                                                  # ... and the prediction just inside it (ix 10)
    p2, g2, f2 = junction_free({"lane_divider": pred2}, {"lane_divider": gt2}, fov, inter)
    mj, mo = SegMeter(grid), SegMeter(grid)
    mj.add(p2, g2, mask=f2)
    mo.add({"lane_divider": pred2}, {"lane_divider": gt2}, mask=fov)
    rj, ro = mj.result()[0.0]["lane_divider"]["all"]["all"], mo.result()[0.0]["lane_divider"]["all"]["all"]
    assert ro["iou@0.5"] == 1.0 and rj["iou@0.5"] == 0.0 and rj["n_gt"] == 1, (ro, rj)       # the plain metric lets the cell inside confirm the GT cell outside; the junction-free row does not
    # and the same the other way round: a GT cell inside must not confirm a predicted cell outside
    p3, g3, f3 = junction_free({"lane_divider": gt2}, {"lane_divider": pred2}, fov, inter)       # prediction just outside (ix 9), GT just inside (ix 10)
    mj, mo = SegMeter(grid), SegMeter(grid)
    mj.add(p3, g3, mask=f3)
    mo.add({"lane_divider": gt2}, {"lane_divider": pred2}, mask=fov)
    rj, ro = mj.result()[0.0]["lane_divider"]["all"]["all"], mo.result()[0.0]["lane_divider"]["all"]["all"]
    assert ro["iou@0.5"] == 1.0 and rj["iou@0.5"] == 0.0 and rj["n_gt"] == 0, (ro, rj)
    # through the real add_sample: the meters of a frame carry the junction rows under their own mask names
    spec = [("lane_divider", "line", "lane_divider"), ("road_divider", "line", "road_divider"), ("drivable_area", "area", "drivable_area")]
    gdict = {"lane_divider": gt_in, "road_divider": np.zeros_like(gt_in), "drivable_area": np.ones_like(gt_in), "camera": np.ones_like(gt_in), "undefined": np.zeros_like(gt_in),
             "ignore": np.zeros_like(gt_in)}
    prob = {"lane_divider": pred, "road_divider": np.zeros_like(pred), "drivable_area": np.ones_like(pred)}
    meters = {k: SegMeter(grid) for k in ("fov", "seen", "out", "obj", "fovj", "fovjb")}
    add_sample(meters, prob, {}, gdict, spec, 0.0, None, junction={"fovj": inter, "fovjb": inter})
    assert {k[1] for k in meters["fovj"].c} == {"lane_divider", "road_divider"} == {k[1] for k in meters["fovjb"].c}, "the two divider layers, no area rows"
    assert meters["fovj"].result()[0.0]["lane_divider"]["all"]["all"]["iou@0"] == 1.0 and meters["fov"].result()[0.0]["lane_divider"]["all"]["all"]["iou@0"] == 30 / 40
    one = {k: SegMeter(grid) for k in ("fov", "seen", "out", "obj")}
    add_sample(one, prob, {}, gdict, spec, 0.0, None)                  # without `junction` the meters are those of before
    assert one["fov"].c.keys() == meters["fov"].c.keys() and all(np.array_equal(one["fov"].c[k]["cnt"], meters["fov"].c[k]["cnt"]) for k in one["fov"].c)
    check_masks()
    return "junction_free rule, dilation barrier both ways, add_sample rows, mask reader (early refusals, [ix, iy], mask id): ok"


def reproduce(n=None):
    """the real-data check against records/intersection_iou_result.txt: the 246 frames of the recall test (stored dt-0 probability maps), the metric's own counting with the
    packed GtGenerator masks instead of the map raster. Prints the base counts against the real metric's stored counts (they differ by at most 1 cell, as in the scratch), the mask
    shares, the orientation check (the mask lies on the GT drivable area, a flipped one does not) and the lane divider IoU at 0 / 0.5 / 1 m of the base and of both junction masks next to
    the stored values. CPU, 2 threads. Needs the untracked reference folder runs/junction_ref/ (rp_frames.json, rp_out/): without it, says so and returns None"""
    import json
    import pickle
    from evaluation.metrics import SegMeter
    here = os.path.join(ROOT, "runs", "junction_ref")
    if not os.path.exists(os.path.join(here, "rp_frames.json")):
        print(f"--reproduce needs the untracked reference folder {here}/rp_frames.json + rp_out/ (the stored predictions of the recall test): not in a clean clone")
        return None
    frames = json.load(open(os.path.join(here, "rp_frames.json")))
    done = [f for f in frames if os.path.exists(os.path.join(here, "rp_out", f"{f['i']}.npz"))][:n]
    jm = JunctionMasks()
    grid = type("G", (), {"x_min": 0.0, "res": 0.5})()
    meters = {m: SegMeter(grid) for m in ("base", *MASKS)}
    share, on_road, worst = np.zeros(4), np.zeros(3), 0.0           # on_road: mask cells on the GT drivable area as drawn, flipped in y, flipped in x
    for f in done:
        g = np.load(os.path.join(ROOT, "data", "cache", "gt", f["file"]), allow_pickle=True)["gt"].item()["cur"]
        prob = np.load(os.path.join(here, "rp_out", f"{f['i']}.npz"))["lane_divider"].astype(np.float32)
        cam, gl = np.asarray(g["camera"], np.float32) > 0, np.asarray(g["lane_divider"], np.float32)
        masks = jm(g)
        meters["base"].add({"lane_divider": prob}, {"lane_divider": gl}, mask=cam)
        for name, inter in masks.items():
            p, gg, ff = junction_free({"lane_divider": prob}, {"lane_divider": gl}, cam, inter)
            meters[name].add(p, gg, mask=ff)
        inter, road = masks["fovj"], np.asarray(g["drivable_area"], np.float32) > 0.5
        share += [cam.sum(), (cam & inter).sum(), ((gl > 0.5) & cam).sum(), ((gl > 0.5) & cam & inter).sum()]
        on_road += [(i & road).sum() / max(inter.sum(), 1) for i in (inter, inter[:, ::-1], inter[::-1])]
        st = pickle.load(open(os.path.join(here, "rp_out", f"{f['i']}.counts.pkl"), "rb"))        # the real metric's stored counts of this frame
        j = [q for q, key in enumerate(st["keys"]) if str(key) == "fov|+0.00|lane_divider|all|all"][0]
        one = SegMeter(grid)
        one.add({"lane_divider": prob}, {"lane_divider": gl}, mask=cam)
        worst = max(worst, float(np.abs(st["cnt"][j].astype(float) - one.c[(0.0, "lane_divider", "all", "all")]["cnt"]).max()))
    res = {k: m.result()[0.0]["lane_divider"]["all"]["all"] for k, m in meters.items()}
    assert on_road[0] / len(done) > max(on_road[1:]) / len(done) + 0.2, "orientation: the mask must lie on the drivable area, a flipped one does not"
    print(f"{len(done)} frames; base counts differ from the stored counts of the real metric by at most {worst:g} cells")
    print(f"share of fov cells inside the polygons {share[1] / share[0]:.3f} (stored 0.210), of the GT divider cells {share[3] / share[2]:.3f} (stored 0.012)")
    print("orientation: share of the mask cells on the GT drivable area %.3f as drawn, %.3f flipped in y, %.3f flipped in x" % tuple(on_road / len(done)))
    print("lane_divider IoU@0 / 0.5 / 1   base " + " / ".join(f"{res['base']['iou@' + t]:.4f}" for t in ("0", "0.5", "1")) + "   (stored 0.1969 / 0.4463 / 0.5613)")
    for name in MASKS:
        print(f"lane_divider IoU@0 / 0.5 / 1   {name:5s}" + " / ".join(f" {res[name]['iou@' + t]:.4f}" for t in ("0", "0.5", "1")) + "   (stored " + " / ".join(f"{v:.4f}" for v in STORED[name]) + ")")
    return res


if __name__ == "__main__":
    if "--reproduce" in sys.argv:
        raise SystemExit(0 if reproduce(int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else None) is not None else 2)
    print(selftest())
