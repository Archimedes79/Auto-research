"""Targets of the junction maps (E285 / E286, D82): the junction GT of `jgt.py` and the painted evidence of `data/prep/junction_paint.py` as arrays on the grids of the pixel stage.

    t, jg = junction_targets(g, paint)     # g = lane GT graph of a frame, paint = `junction_paint.load(token)` or None
    t["jz"]      (2, 100, 100) float32   junction zone and conflict core (1 m cells, row = x, column = y, origin -50 m)
    t["jheat"]   (2, 100, 100) float32   entry / exit heat: Gaussian (sigma 1.5 m) at the entry / exit points
    t["jdir"]    (4, 100, 100) float32   entry heading (tx, ty), exit heading (tx, ty) at the cells within 2 sigma of an end, 0 elsewhere
    t["jpaint"]  (2, 200, 200) float32   0.5 m masks: stop lines, pedestrian crossings (the map expansion's polygons; thin stop lines are also drawn as a 1 cell line)
All by location: no matching, nothing predicted. A frame without a junction gets zero maps (`jg` is None). Run the self-test: python qjunction.py
"""
import numpy as np

import jgt as JG

N05, RES05, HALF = 200, 0.5, 50.0


def _fill(mask, poly):
    import cv2
    pts = np.round(((np.asarray(poly, float) + HALF) / RES05)[:, ::-1]).astype(np.int32)        # (column = y, row = x) for cv2
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 1)
        cv2.polylines(mask, [pts], True, 1, 1)


def junction_targets(g, paint=None):
    jg = JG.junction_gt(g)
    zone, heat, head = JG.fields(jg)
    stop = np.zeros((N05, N05), np.uint8)
    ped = np.zeros((N05, N05), np.uint8)
    if paint:
        for s in paint["stop_line"]:
            _fill(stop, s["poly"])
        for p in paint["ped_crossing"]:
            _fill(ped, p)
    return dict(jz=zone, jheat=heat, jdir=head, jpaint=np.stack([stop, ped]).astype(np.float32)), jg


def painted_flags(jg, paint, stop_m=3.0, ped_m=4.0):
    """(ent_painted (E,), ext_painted (X,)) bool: an end is PAINTED when a stop line of a type other than the turn stop (turn stops lie inside the junction, they are waiting points, not entries) lies within `stop_m`
    or a pedestrian crossing within `ped_m` (E285: only about half of the entries are painted: the rest has to be inferred from context)"""
    if jg is None:
        return np.zeros(0, bool), np.zeros(0, bool)
    from shapely.geometry import Point, Polygon
    stops = [Polygon(s["poly"]).buffer(0) for s in (paint or {}).get("stop_line", []) if s["type"] != 0 and len(s["poly"]) > 2]
    peds = [Polygon(p).buffer(0) for p in (paint or {}).get("ped_crossing", []) if len(p) > 2]

    def flag(E):
        out = np.zeros(len(E), bool)
        for i, e in enumerate(E):
            pt = Point(float(e[0]), float(e[1]))
            out[i] = any(p.distance(pt) <= stop_m for p in stops) or any(p.distance(pt) <= ped_m for p in peds)
        return out
    return flag(jg["ent"]), flag(jg["ext"])


def _selftest():
    def run(a, b, n=12):
        return np.stack([np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)], 1)
    s, nth = run((0, -30), (0, -8)), run((0, 8), (0, 30))
    conn = run(s[-1], nth[0], 16)
    pts = np.vstack([s, conn, nth])
    g = dict(pieces=[dict(piece="p0", pts=pts, tier=np.full(len(pts), 3),
                          segments=[dict(kind="lane", i0=0, i1=len(s) - 1), dict(kind="connector", i0=len(s), i1=len(s) + len(conn) - 1), dict(kind="lane", i0=len(s) + len(conn), i1=len(pts) - 1)])])
    paint = dict(stop_line=[dict(poly=np.array([[-2, -8.2], [2, -8.2], [2, -7.8], [-2, -7.8]], np.float32), type=3)], ped_crossing=[np.array([[-3, 6], [3, 6], [3, 8], [-3, 8]], np.float32)])
    t, jg = junction_targets(g, paint)
    assert jg is not None and t["jz"].shape == (2, 100, 100) and t["jheat"].shape == (2, 100, 100) and t["jdir"].shape == (4, 100, 100) and t["jpaint"].shape == (2, 200, 200)
    assert t["jz"][0].sum() > 20 and t["jpaint"][0].sum() >= 8 and t["jpaint"][1].sum() > 20
    # the stop line lies at x = 0, y = -8: row 100, column 84 / 85 of the 0.5 m grid
    assert t["jpaint"][0][100, 83:86].sum() > 0
    pe, px = painted_flags(jg, paint)
    assert len(pe) == len(jg["ent"]) and pe[0] and px[0], (pe, px)                         # stop sign line at the entry, crossing at the exit
    t0, jg0 = junction_targets(dict(pieces=[]), None)
    assert jg0 is None and t0["jz"].sum() == 0 and t0["jpaint"].sum() == 0
    print("qjunction selftest ok")


if __name__ == "__main__":
    _selftest()
