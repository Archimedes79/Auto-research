p = "evaluation/soa_maps.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    return "soa_maps selftest ok:''', '''    # 8. the claim region is CROPPED, not masked: a GT cell beyond the reference array (row 99) must not confirm a predicted cell inside it through the tolerance dilation
    zero = {k: np.zeros((NX, NY), np.float32) for k in LAYERS}
    pred, gtl = {**zero, "lane_divider": np.zeros((NX, NY), np.float32)}, {**zero, "lane_divider": np.zeros((NX, NY), np.float32)}
    pred["lane_divider"][98, 40], gtl["lane_divider"][99, 40] = 1.0, 1.0
    c, _ = frame_counts({"ref": pred}, gtl, np.ones((NX, NY), bool), claim_rows(x_max=50.0))
    ld = LAYERS.index("lane_divider")
    assert list(c[("claim", "ref")][THR.index(0.5), ld, 1]) == [0, 1, 0, 0] and list(c[("old", "ref")][THR.index(0.5), ld, 1]) == [1, 0, 1, 0], (c[("claim", "ref")][1, ld, 1], c[("old", "ref")][1, ld, 1])
    return "soa_maps selftest ok:''')
s = s.replace('old region artefact"', 'crop not mask, old region artefact"')
s = s.replace('''@mutant("the camera cells are not applied (every cell scored)")''', '''@mutant("the claim region masks the far rows instead of cropping the maps")
def _m_mask():
    keep = frame_counts

    def bad(maps, gt, fov, rows):
        f = np.asarray(fov).copy()
        f[rows:] = False
        return keep(maps, gt, f, NX)                                  # all rows reach the dilation; only the counted cells are cut (and the old region is then cut too)
    return {"frame_counts": bad}


@mutant("the camera cells are not applied (every cell scored)")''')
open(p, "w", encoding="utf-8").write(s)
