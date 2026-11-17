import os, sys, numpy as np
R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", "evaluation", ".."):
    sys.path.insert(0, os.path.join(R, sub))
import lane_map_data as LMD
cases = {
 "none": [],
 "one point": [np.array([[1.0, 2.0]])],
 "two equal": [np.array([[1.0, 2.0], [1.0, 2.0]])],
 "short 1m": [np.array([[0.0, 0.0], [1.0, 0.0]])],
 "dup pts": [np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 0.0], [5.0, 0.0], [10.0, 0.0]])],
 "nan": [np.array([[0.0, 0.0], [np.nan, 1.0], [5.0, 0.0]])],
}
for name, pts in cases.items():
    for fn in ("exact", "error", "drop_all"):
        try:
            if fn == "exact": raw = LMD.exact_map(pts)
            elif fn == "error": raw = LMD.error_map(pts, "tok")
            else:
                sd = LMD.roads_of(pts); sd["roads"] = []; raw = LMD.sample_roads(sd, np.array([1., 1.]), 0.01)
            t = LMD.finish(raw)
            c = LMD.collate_maps([t])
            print(f"{name:10s} {fn:9s} n={len(t['valid'])} finite={np.isfinite(t['pos']).all() and np.isfinite(t['dir']).all()} M={c['valid'].shape[1]}")
        except Exception as e:
            print(f"{name:10s} {fn:9s} EXC {type(e).__name__}: {str(e)[:80]}")
