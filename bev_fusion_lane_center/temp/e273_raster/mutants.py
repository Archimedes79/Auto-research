"""scratch: mutation check of qraster._selftest (every mutant must fail it)"""
import os
ROOT = os.environ["BFLC_ROOT"]
src = open(os.path.join(ROOT, "training", "grid", "qraster.py"), encoding="utf-8").read()
M = [
    ("road >= 0.5 -> > 0.5", ">= 0.5\n    if inter", "> 0.5\n    if inter"),
    ("any coverage by supersampling", "out[1] = _touched(inter, n, res)", "out[1] = _coverage(inter, n, res, ss) > 0"),
    ("edge contact counts", "hit[hit] = shapely.relate_pattern(p, box[hit], \"T********\")        # the interiors meet", "pass"),
    ("event axes swapped", "p[None, None, :, 0]) ** 2 + (c[None, :, None] - p[None, None, :, 1])", "p[None, None, :, 1]) ** 2 + (c[None, :, None] - p[None, None, :, 0])"),
    ("dist not truncated", "np.minimum(d, trunc)", "np.where(np.isinf(d), 99.0, d)"),
    ("corner pieces kept", "(tt[1:] - tt[:-1] > 1e-9)", "(tt[1:] - tt[:-1] >= 0)"),
    ("segment start cell only", "t = sid[:-1][ok], 0.5 * (tt[1:] + tt[:-1])[ok]", "t = sid[:-1][ok], 0.0 * (tt[1:] + tt[:-1])[ok]"),
    ("cross is only a split", "EVENT_OF[\"cross\"] = (0, 1)", "EVENT_OF[\"cross\"] = (0,)"),
    ("window events drawn", "EVENT_OF[\"cross\"] = (0, 1)", "EVENT_OF[\"cross\"] = (0, 1)\nEVENT_OF[\"window\"] = (2,)"),
    ("events add up", ").min(2)\n            out[k]", ").sum(2)\n            out[k]"),
    ("dilate ignored", "if dilate > 0:", "if False:"),
    ("box <=", "k = (np.abs(P) < box).all(1)", "k = (np.abs(P) <= box).all(1)"),
    ("dilation of the query too small", "size=2 * int(np.ceil(trunc / res)) + 1", "size=2 * int(np.ceil(trunc / res)) - 1"),
    ("sigma 1.5 -> 1.0", "sigma=1.5", "sigma=1.0"),
    ("lane tangent flipped", "T.append(QT._tangents(q))", "T.append(-QT._tangents(q))"),
    ("mask cells x/y swapped", "m[c[:, 0], c[:, 1]] = True", "m[c[:, 1], c[:, 0]] = True"),
    ("holes ignored", "[h for h in (np.asarray(h, float).reshape(-1, 2) for h in a.get(\"holes\", [])) if len(h) >= 3]", "[]"),
    ("supersample 2 not 4", "def areas2(g, n=50, res=2.0, dilate=0.0, ss=4)", "def areas2(g, n=50, res=2.0, dilate=0.0, ss=2)"),
]
for name, a, b in M:
    assert a in src, name
    ns = {"__name__": "mut"}
    try:
        exec(compile(src.replace(a, b, 1), "qraster_mut", "exec"), ns)
        ns["_selftest"]()
        print("SURVIVED:", name)
    except Exception as e:
        print("killed  :", name, "|", type(e).__name__, str(e)[:60].replace("\n", " "))
