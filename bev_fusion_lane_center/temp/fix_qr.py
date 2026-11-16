p = "training/grid/qraster.py"
s = open(p, encoding="utf-8").read()
s = s.replace('    r3 = areas2(g, dilate=2.0)\n    assert r3[1].sum() > areas2(g)[1].sum() and r3[0].sum() > r2[0].sum(), "dilate grows the polygons"',
              '    r3 = areas2(g, dilate=2.0)\n    assert r3[1].sum() > areas2(g)[1].sum() and (r3[0] == areas2(g)[0]).all(), "dilate grows the INTERSECTION polygons only"\n    r4 = areas2(g, dilate_road=2.0)\n    assert r4[0].sum() > r2[0].sum() and (r4[1] == areas2(g)[1]).all(), "dilate_road grows the road polygons only"')
open(p, "w", encoding="utf-8").write(s)
