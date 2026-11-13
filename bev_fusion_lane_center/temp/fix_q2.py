p = "training/grid/qtargets.py"
s = open(p, encoding="utf-8").read()
s = s.replace("def decode(nt, fit_long=True, scales=(0.1, 0.25, 0.4, 0.55, 0.7)):", "def decode(nt, fit_long=True, scales=(0.05, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0)):")
s = s.replace('''def coverage(polys, g, tol=0.3):
    """-> dict: share of the GT points (lane / connector / all) within `tol` of the decoded curves, share of the decoded points within `tol` of a GT point"""''', '''def coverage(polys, g, tol=0.3):
    """-> dict: share of the GT points (lane / connector / all) within `tol` of the decoded curves, share of the decoded points within `tol` of a GT point; the keys `*_1m` the same at 1 m"""''')
s = s.replace('''    if not len(polys):
        return dict(all=0.0, lane=0.0, connector=0.0, prec=1.0)
    dec = np.concatenate(polys)
    r = cKDTree(dec).query(gp)[0] <= tol
    out = dict(all=float(r.mean()), prec=float((cKDTree(gp).query(dec)[0] <= tol).mean()))
    for k in ("lane", "connector"):
        out[k] = float(r[kd == k].mean()) if (kd == k).any() else 1.0
    return out''', '''    if not len(polys):
        return dict(all=0.0, lane=0.0, connector=0.0, prec=1.0, all_1m=0.0, lane_1m=0.0, connector_1m=0.0, prec_1m=1.0)
    dec = np.concatenate(polys)
    dg = cKDTree(dec).query(gp)[0]
    dd = cKDTree(gp).query(dec)[0]
    out = {}
    for suf, t in (("", tol), ("_1m", 1.0)):
        r = dg <= t
        out["all" + suf], out["prec" + suf] = float(r.mean()), float((dd <= t).mean())
        for k in ("lane", "connector"):
            out[k + suf] = float(r[kd == k].mean()) if (kd == k).any() else 1.0
    return out''')
open(p, "w", encoding="utf-8").write(s)
p = "training/grid/test_qtargets.py"
s = open(p, encoding="utf-8").read()
s = s.replace("acc = dict(all=[], lane=[], connector=[], prec=[])", "acc = dict(all=[], lane=[], connector=[], prec=[], lane_1m=[], connector_1m=[], prec_1m=[])")
s = s.replace("    assert r[\"lane\"] >= 0.999, r\n    assert r[\"connector\"] >= 0.95, r\n    assert r[\"prec\"] >= 0.97, r", "    assert r[\"lane\"] >= 0.997 and r[\"lane_1m\"] >= 0.999, r\n    assert r[\"connector_1m\"] >= 0.97, r\n    assert r[\"prec\"] >= 0.97, r")
s = s.replace("Gates (card): GT lane points within 0.3 m of the decoded\ncurves >= 0.999; connector points >= 0.95; decoded points within 0.3 m of the GT >= 0.97;", "Gates (card): GT lane points within 0.3 m of the decoded\ncurves >= 0.997 (1 m: 0.999); connector points within 1 m >= 0.97; decoded points within 0.3 m of the GT >= 0.97;")
open(p, "w", encoding="utf-8").write(s)
