q = "training/grid/qjunction.py"
s = open(q, encoding="utf-8").read()
old = "def _selftest():"
new = '''def painted_flags(jg, paint, stop_m=3.0, ped_m=4.0):
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


def _selftest():'''
assert old in s
s = s.replace(old, new, 1)
s = s.replace('    t0, jg0 = junction_targets(dict(pieces=[]), None)', '    pe, px = painted_flags(jg, paint)\n    assert len(pe) == len(jg["ent"]) and pe[0] and not px[0], (pe, px)                         # the entry (0, -8) lies on the stop line, the exit (0, 8) on the crossing: 4 m rule -> painted too\n    t0, jg0 = junction_targets(dict(pieces=[]), None)', 1)
open(q, "w", encoding="utf-8").write(s)
print("ok")
