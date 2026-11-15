import re
p = "evaluation/bev_junction.py"
s = open(p, encoding="utf-8").read()
a = s.index("def reproduce(n=None):")
b = s.index('if __name__ == "__main__":')
new = '''STORED = {"fovj": (0.2204, 0.5091, 0.6410), "fovjb": (0.2223, 0.5141, 0.6472)}      # temp/overview/intersection_iou_result.txt, variants "both" and "both+band" (the map raster, PIL), IoU@0 / 0.5 / 1


def reproduce(n=None):
    """the real-data check against temp/overview/intersection_iou_result.txt: the 246 frames of the recall test (stored dt-0 probability maps), the metric's own counting with the
    packed GtGenerator masks instead of the map raster. Prints the base counts against the real metric's stored counts, the mask shares, the orientation check (the mask lies on the
    GT drivable area, a flipped one does not) and the lane divider IoU at 0 / 0.5 / 1 m of the base and of both junction masks next to the stored values. CPU, 2 threads"""
    import json
    import pickle
    from evaluation.metrics import SegMeter
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "temp", "overview")
    frames = json.load(open(os.path.join(here, "rp_frames.json")))
    done = [f for f in frames if os.path.exists(os.path.join(here, "rp_out", f"{f['i']}.npz"))][:n]
    jm = JunctionMasks()
    grid = type("G", (), {"x_min": 0.0, "res": 0.5})()
    meters = {m: SegMeter(grid) for m in ("base", *MASKS)}
    share, on_road, worst = np.zeros(4), np.zeros(3), 0.0           # on_road: mask cells on the GT drivable area as drawn, flipped in y, flipped in x
    for f in done:
        g = np.load(os.path.join(here, "..", "..", "data", "cache", "gt", f["file"]), allow_pickle=True)["gt"].item()["cur"]
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
        st = pickle.load(open(os.path.join(here, "rp_out", f"{f['i']}.counts.pkl"), "rb"))        # the real metric's stored counts of this frame: the base must equal them
        j = [q for q, key in enumerate(st["keys"]) if str(key) == "fov|+0.00|lane_divider|all|all"][0]
        one = SegMeter(grid)
        one.add({"lane_divider": prob}, {"lane_divider": gl}, mask=cam)
        worst = max(worst, float(np.abs(st["cnt"][j].astype(float) - one.c[(0.0, "lane_divider", "all", "all")]["cnt"]).max()))
    res = {k: m.result()[0.0]["lane_divider"]["all"]["all"] for k, m in meters.items()}
    print(f"{len(done)} frames; base counts equal the stored counts of the real metric: max difference {worst:g} cells")
    print(f"share of fov cells inside the polygons {share[1] / share[0]:.3f} (stored 0.210), of the GT divider cells {share[3] / share[2]:.3f} (stored 0.012)")
    print("orientation: share of the mask cells on the GT drivable area %.3f as drawn, %.3f flipped in y, %.3f flipped in x" % tuple(on_road / len(done)))
    print(f"lane_divider IoU@0 / 0.5 / 1   base " + " / ".join(f"{res['base']['iou@' + t]:.4f}" for t in ("0", "0.5", "1")) + "   (stored 0.1969 / 0.4463 / 0.5613)")
    for name in MASKS:
        print(f"lane_divider IoU@0 / 0.5 / 1   {name:5s}" + " / ".join(f" {res[name]['iou@' + t]:.4f}" for t in ("0", "0.5", "1")) + "   (stored " + " / ".join(f"{v:.4f}" for v in STORED[name]) + ")")
    return res


'''
open(p, "w", encoding="utf-8").write(s[:a] + new + s[b:])
