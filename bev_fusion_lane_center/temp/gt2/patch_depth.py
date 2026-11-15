p = "temp/e293/depth_probe.py"
s = open(p, encoding="utf-8").read()
s = s.replace("runs_n = 0\n", "runs_n = 0\nctrl = dict(n=0, heat=0, paint=0, heat_sum=0.0)\n", 1)
old = "        for r in runs:\n            q = r[\"pts\"]"
new = '''        jg = J.junction_gt(g)                                                                   # control: points INSIDE the zone that are more than 3 m from every connector point
        allq = np.concatenate([r["pts"] for r in runs])
        qt = cKDTree(allq)
        import shapely
        for z in (jg["zones"] if jg is not None else []):
            x0, y0, x1, y1 = z["shp"].bounds
            gx, gy = np.meshgrid(np.arange(np.floor(x0), x1 + 1, 1.0) + 0.5, np.arange(np.floor(y0), y1 + 1, 1.0) + 0.5, indexing="ij")
            pp = np.stack([gx.ravel(), gy.ravel()], 1)
            pp = pp[shapely.contains_xy(z["shp"], pp[:, 0], pp[:, 1]) & (np.abs(pp) < 49).all(1)]
            if not len(pp):
                continue
            pp = pp[qt.query(pp)[0] > 3.0]
            if not len(pp):
                continue
            ij = np.clip(np.floor((pp + 50.0) / 1.0).astype(int), 0, 99)
            hv = cen[i][ij[:, 0], ij[:, 1]]
            ctrl["n"] += len(pp)
            ctrl["heat"] += int((hv >= 0.3).sum())
            ctrl["heat_sum"] += float(hv.sum())
            ctrl["paint"] += int((tree.query(pp)[0] <= 1.0).sum()) if tree is not None else 0
        for r in runs:
            q = r["pts"]'''
assert old in s
s = s.replace(old, new, 1)
s = s.replace("res = dict(seed=seed, frames=n, runs=runs_n,", "res = dict(seed=seed, frames=n, runs=runs_n, off_path_control=dict(points=ctrl[\"n\"], heat_ge_0_3=ctrl[\"heat\"] / max(ctrl[\"n\"], 1), heat_mean=ctrl[\"heat_sum\"] / max(ctrl[\"n\"], 1), paint_within_1m=ctrl[\"paint\"] / max(ctrl[\"n\"], 1)),")
open(p, "w", encoding="utf-8").write(s)
print("ok")
