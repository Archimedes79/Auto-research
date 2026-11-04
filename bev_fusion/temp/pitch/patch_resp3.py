p = "evaluation/bev_response.py"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)
a = s.index("def annotated(samples, task):")
b = s.index("def false_in_area")
s = s[:a] + '''def annotated(samples, task):
    """per sample: every annotated box of the scored classes inside the window (observed or not), with its fusion-area class (footprint share from the corners)
    and the sample's area"""
    from evaluation.bev_fusion_area import fusion_area, gt_class, gt_with_shares
    out = []
    for _, s in samples:
        g = s[0.0]
        boxes, shares = gt_with_shares(g, task.groups)
        out.append(([dict(b, cls=gt_class(f)) for b, f in zip(boxes, shares) if in_front(b)], fusion_area(g)))
    return out


def geometry_rows(samples, task):
    """footprint geometry against an INDEPENDENT area: the cached `vehicle` layer is the area, so the footprint of a vehicle box must lie in it (mean share >= 0.95);
    a y flip, swapped length / width or a negated yaw must fall below 0.9. Then the autogt reference implementation (autogt/coverage.py, the GT owner's): the same fusion
    cells, the same corner shares, and the same classes for the dict-based footprint that detections use."""
    from autogt import coverage as AC
    from autogt.grid import GridSpec
    from evaluation.bev_fusion_area import corner_share, footprint_share, fusion_area, gt_class, gt_with_shares
    rows, grid = [], GridSpec(res=0.5)
    shares = {"as is": [], "y flipped": [], "length / width swapped": [], "yaw negated": []}
    differ, maxdiff, n_cls, n_box = 0, 0.0, 0, 0
    for _, s in samples:
        g = s[0.0]
        layer = np.asarray(g["vehicle"], np.float32) > 0.5
        boxes, f_corner = gt_with_shares(g, task.groups)
        for b, fc, rb in zip(boxes, f_corner, [r for r in g["meta"]["boxes"] if r["group"] in task.groups]):
            if b["group"] == CLASS and "length" in b and in_front(b):
                for k, fn in (("as is", lambda b: b), ("y flipped", lambda b: dict(b, cy=-b["cy"])), ("length / width swapped", lambda b: dict(b, length=b["width"], width=b["length"])),
                              ("yaw negated", lambda b: dict(b, yaw=-b["yaw"]))):
                    shares[k].append(footprint_share(fn(b), layer))
            ref = AC.footprint_share(rb["corners"], AC.fusion_cells(g), grid)
            maxdiff = max(maxdiff, abs(fc - ref))
            if "length" in b:
                n_box += 1
                n_cls += gt_class(footprint_share(b, fusion_area(g))) != gt_class(fc)
        differ += int((fusion_area(g) != AC.fusion_cells(g)).sum())
    m = {k: float(np.mean(v)) for k, v in shares.items()}
    lab = f"geometry: {len(shares['as is'])} vehicle footprints against the cached vehicle layer as the area"
    rows += [(lab, "mean share as is >= 0.95 (1 = yes)", float(m["as is"] >= 0.95), 1.0, 0.0)]
    rows += [(lab, f"mean share {k} < 0.9 (1 = yes)", float(m[k] < 0.9), 1.0, 0.0) for k in ("y flipped", "length / width swapped", "yaw negated")]
    lab = "autogt reference (autogt/coverage.py)"
    rows += [(lab, "cells where fusion_area differs from fusion_cells", float(differ), 0.0, 0.0), (lab, "max |corner share - autogt footprint_share|", maxdiff, 0.0, 0.02),
             (lab, "share of box footprints (dict form, as for detections) with another class than the corner form", n_cls / max(n_box, 1), 0.0, 0.01)]
    return rows


''' + s[b:]
rep('''    Nc, Np = cnt["care"], cnt["partial"]
    rows = []''', '''    Nc, Np = cnt["care"], cnt["partial"]
    rows = geometry_rows(samples, task)
    assert Nc > 0 and Np > 0 and cnt["dont_care"] > 0, f"too few frames for the fusion rows (care {Nc}, partial {Np}, don't care {cnt['dont_care']}): use --n 24 or more"''')
rep('''    mi = sum(len(f) for f in fi)''', '''    mi = sum(len(f) for f in fi)
    assert mi > 0, "no false box could be placed inside the area"''')
rep('''    mo = sum(len(f) for f in fo)''', '''    mo = sum(len(f) for f in fo)
    assert mo > 0, "no false box could be placed outside the area"''')
rep('''    m = run(shifted, "detf")''', '''    assert n_sh > 0, "no neutral detection could be nudged into the area"
    m = run(shifted, "detf")''')
open(p, "w", encoding="utf-8").write(s)
