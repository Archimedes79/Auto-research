p = "training/grid/jgt.py"
s = open(p, encoding="utf-8").read()
a = s.index("def raster_exact():")
b = s.index("def fields(jg")
new = '''def raster_exact():
    """BFLC_RASTER=exact: the polygon masks cover a cell when MORE THAN HALF of its area is inside the polygon (8 x 8 supersampling, no half-cell shift); default: the legacy `np.round` + `cv2.fillPoly` of E285 / E286, kept so the stored numbers stay reproducible"""
    return os.environ.get("BFLC_RASTER", "") == "exact"


SS = 8


def fill_poly(mask, pts_cells, line=False):
    """fill the polygon `pts_cells` ((m, 2) in CELL units, column = y first as cv2 wants: x = column) into `mask` (uint8, row = x). Legacy: integer vertices (cv2 puts integer coordinates on pixel CENTRES while our cell k spans [k, k + 1): the mask is
    shifted by half a cell, critic 2026-11-11); `line`: the outline is drawn too (thin stop lines stay visible). Exact (BFLC_RASTER=exact): a cell is in when more than half of it is inside (`line`: more than a quarter, the thin paint)"""
    import cv2
    pts = np.asarray(pts_cells, float)
    if raster_exact():
        c0, c1 = max(int(np.floor(pts[:, 0].min())), 0), min(int(np.ceil(pts[:, 0].max())) + 1, mask.shape[1])
        r0, r1 = max(int(np.floor(pts[:, 1].min())), 0), min(int(np.ceil(pts[:, 1].max())) + 1, mask.shape[0])
        if c1 <= c0 or r1 <= r0:
            return
        big = np.zeros(((r1 - r0) * SS, (c1 - c0) * SS), np.uint8)
        q = np.round(((pts - np.array([c0, r0])) * SS - 0.5) * 16).astype(np.int32)                     # sub-cell vertices (1/16 of a supersample), supersample centres at integer coordinates
        cv2.fillPoly(big, [q], 1, lineType=cv2.LINE_8, shift=4)
        cov = big.reshape(r1 - r0, SS, c1 - c0, SS).mean((1, 3))
        mask[r0:r1, c0:c1] |= (cov > (0.25 if line else 0.5)).astype(mask.dtype)
    else:
        q = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [q], 1)
        if line:
            cv2.polylines(mask, [q], True, 1, 1)


'''
s = s[:a] + new + s[b:]
# self-test text: add a random convex polygon against shapely
s = s.replace('''        assert m0.sum() > 0 and not (m0 == want).all(), "the legacy raster is the half-cell shifted one"
''', '''        assert m0.sum() > 0 and not (m0 == want).all(), "the legacy raster is the half-cell shifted one"
        from shapely.geometry import Polygon, box as sbox
        rng = np.random.RandomState(0)
        for _ in range(20):                                                                           # random convex polygons: the cells with more than half their area inside agree with shapely
            ang = np.sort(rng.uniform(0, 2 * np.pi, 6))
            poly = np.stack([rng.uniform(-20, 20) + rng.uniform(2, 9) * np.cos(ang), rng.uniform(-20, 20) + rng.uniform(2, 9) * np.sin(ang)], 1)
            sp = Polygon(poly)
            if not sp.is_valid or sp.area < 2:
                continue
            mm = np.zeros((100, 100), np.uint8)
            fill_poly(mm, ((poly + 50.0) / 1.0)[:, ::-1])
            ii, jj = np.where(mm > 0)
            r, c = np.meshgrid(np.arange(100), np.arange(100), indexing="ij")
            lo = np.array([sp.bounds[0] + 50 - 2, sp.bounds[1] + 50 - 2]).astype(int).clip(0, 99)
            hi = np.array([sp.bounds[2] + 50 + 2, sp.bounds[3] + 50 + 2]).astype(int).clip(0, 99)
            ref = np.zeros((100, 100), bool)
            for i in range(lo[0], hi[0] + 1):
                for j in range(lo[1], hi[1] + 1):
                    ref[i, j] = sp.intersection(sbox(i - 50, j - 50, i - 49, j - 49)).area > 0.5
            assert (mm.astype(bool) != ref).sum() <= 0.03 * max(ref.sum(), 1) + 2, ((mm.astype(bool) != ref).sum(), ref.sum())
''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
