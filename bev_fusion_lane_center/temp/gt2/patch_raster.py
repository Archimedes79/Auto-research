def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:80])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, count))


# ---- jgt: one polygon rasteriser with an opt-in exact mode (critic 2026-11-11: cv2.fillPoly puts integer coordinates on pixel CENTRES, our cell k spans [k, k+1): the mask is shifted by half a cell)
sub("training/grid/jgt.py", "def fields(jg, n=100, res=1.0, half=50.0):\n", '''def raster_exact():
    """BFLC_RASTER=exact: the polygon masks are rasterised with sub-cell vertices and the half-cell shift removed (default: the legacy `np.round` + `cv2.fillPoly` of E285 / E286, kept so the stored numbers stay reproducible)"""
    return os.environ.get("BFLC_RASTER", "") == "exact"


def fill_poly(mask, pts_cells, line=False):
    """fill the polygon `pts_cells` ((m, 2) in CELL units, column = y first as cv2 wants: x = column) into `mask`; legacy: integer vertices, shifted by half a cell; exact: 1/16 cell vertices and cell CENTRES (k + 0.5) as the sampling points"""
    import cv2
    pts = np.asarray(pts_cells, float)
    if raster_exact():
        q = np.round((pts - 0.5) * 16).astype(np.int32)
        cv2.fillPoly(mask, [q], 1, lineType=cv2.LINE_8, shift=4)
        if line:
            cv2.polylines(mask, [q], True, 1, 1, lineType=cv2.LINE_8, shift=4)
    else:
        q = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [q], 1)
        if line:
            cv2.polylines(mask, [q], True, 1, 1)


def fields(jg, n=100, res=1.0, half=50.0):
''')
sub("training/grid/jgt.py", "        pts = ((z[\"poly\"] + half) / res)[:, ::-1]\n        cv2.fillPoly(zone[0], [np.round(pts).astype(np.int32)], 1)\n", "        fill_poly(zone[0], ((z[\"poly\"] + half) / res)[:, ::-1])\n")
sub("training/grid/jgt.py", "                cv2.fillPoly(zone[1], [np.round(((np.asarray(q.exterior.coords, float) + half) / res)[:, ::-1]).astype(np.int32)], 1)\n", "                fill_poly(zone[1], ((np.asarray(q.exterior.coords, float) + half) / res)[:, ::-1])\n")

# ---- qjunction._fill through the same rasteriser
s = open("training/grid/qjunction.py", encoding="utf-8").read()
a = s.index("def _fill(mask, poly):")
b = s.index("def junction_targets(")
s = s[:a] + '''def _fill(mask, poly):
    pts = ((np.asarray(poly, float) + HALF) / RES05)[:, ::-1]                                   # (column = y, row = x) for cv2
    if len(pts) >= 3:
        JG.fill_poly(mask, pts, line=True)


''' + s[b:]
open("training/grid/qjunction.py", "w", encoding="utf-8").write(s)

# ---- self-test of the exact mode in qjunction (appended before the final message)
print("ok")

# ---- jgt: import os, drop the local cv2 import of fields, self-test of the two modes
sub("training/grid/jgt.py", "import numpy as np\n\nimport qends as QE", "import os\n\nimport numpy as np\n\nimport qends as QE")
sub("training/grid/jgt.py", '    import cv2\n    zone = np.zeros((2, n, n), np.uint8)', '    zone = np.zeros((2, n, n), np.uint8)')
s = open("training/grid/jgt.py", encoding="utf-8").read()
i = s.index("def _selftest():")
s = s[:i] + '''def _selftest_raster():
    """the exact raster puts a cell in the mask when its CENTRE is inside the polygon; the legacy one is shifted by half a cell (kept for the stored numbers)"""
    box = np.array([[-1.2, 0.3], [3.4, 0.3], [3.4, 5.7], [-1.2, 5.7]])                           # x in [-1.2, 3.4], y in [0.3, 5.7] metres
    cells = ((box + 50.0) / 1.0)[:, ::-1]
    old = os.environ.pop("BFLC_RASTER", None)
    try:
        m0 = np.zeros((100, 100), np.uint8)
        fill_poly(m0, cells)
        os.environ["BFLC_RASTER"] = "exact"
        m1 = np.zeros((100, 100), np.uint8)
        fill_poly(m1, cells)
        xc = -49.5 + np.arange(100)
        want = ((xc[:, None] >= -1.2) & (xc[:, None] <= 3.4) & (xc[None, :] >= 0.3) & (xc[None, :] <= 5.7)).astype(np.uint8)
        assert (m1 == want).all(), (int(m1.sum()), int(want.sum()))
        assert m0.sum() > 0 and not (m0 == want).all(), "the legacy raster is the half-cell shifted one"
    finally:
        os.environ.pop("BFLC_RASTER", None)
        if old is not None:
            os.environ["BFLC_RASTER"] = old


''' + s[i:]
s = s.replace("\nif __name__ == \"__main__\":\n    _selftest()", "\nif __name__ == \"__main__\":\n    _selftest_raster()\n    _selftest()")
open("training/grid/jgt.py", "w", encoding="utf-8").write(s)
print("ok2")
