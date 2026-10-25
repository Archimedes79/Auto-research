"""Loop 17: can the E94 control graphs of the VAL failure series (out/loop2/e116_valfail, loops 13 / 14) be scored with the loop-16 card? Prints the structure and the card per phase."""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_common as C  # noqa: E402
import p8_lib as L  # noqa: E402

OUT = L.OUT
for var in ("single", "maxdecay_0.95", "ema_hold_mon_0.5"):
    f = f"{OUT}/e116_valfail/graphs_dark_{var}.pkl"
    r = pickle.load(open(f, "rb"))
    print(var, list(r.keys()), len(r["graphs"]), type(r["graphs"][0]), list(r["graphs"][0].keys()) if isinstance(r["graphs"][0], dict) else "")
    es_gt = L.cache("gt", "val")
    gts = [es_gt[int(x)] for x in r["rows"]]
    ph = np.asarray([p == "fail" for p in r["phase"]])
    for nm, m in (("cl", ~ph), ("fail", ph)):
        idx = np.where(m)[0]
        c = C.card([r["graphs"][i] for i in idx], [gts[i] for i in idx])
        print("  ", nm, len(idx), {k: round(c[k], 3) for k in ("S", "F1@1.0", "J", "vert100", "lanes_found")})
