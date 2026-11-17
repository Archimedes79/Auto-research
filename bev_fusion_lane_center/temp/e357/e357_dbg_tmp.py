"""E357 throw-away: why the GT links through the writer do not give link P = R = 1 on 3 VAL frames"""
import collections

import numpy as np
import torch

torch.set_num_threads(2)
import jtok_eval as JV  # noqa: E402
import lane_tok_data as TD  # noqa: E402
import qgraph_eval as QE  # noqa: E402

for k in (0, 200, 400):
    f = QE.val_frames()[k] if k == 0 else fr[k]  # noqa: F821
    if k == 0:
        fr = QE.val_frames()
        f = fr[0]
    t = TD.frame_table(f.lane(), 2.5)
    p = JV.gt_prediction(t)
    lk = p["links"]
    n = JV.link_numbers([JV.link_counts(t, lk["src"], lk["dst"], lk["a"], lk["b"])])
    pairs = collections.Counter(zip(t["src"].tolist(), t["dst"].tolist()))
    dup = {pr: c for pr, c in pairs.items() if c > 1}
    print(k, {key: round(v, 4) for key, v in n.items() if not key.startswith("link_c")}, "duplicates", dup, "self", int((t["src"] == t["dst"]).sum()))
    for (a, b) in dup:
        m = (t["src"] == a) & (t["dst"] == b)
        print("   dup", a, b, "long", t["long"][m], "conn", t["conn"][m], "scale", t["scale"][m], "chord", np.linalg.norm(t["pos"][a] - t["pos"][b]))
