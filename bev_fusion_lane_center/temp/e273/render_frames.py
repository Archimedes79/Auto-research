"""one-off: pictures of the best / median / worst VAL frames of seed 0 (by per-frame topo F1 among non-trivial frames) for the E273 report (CPU, about 3 s per frame)

    wsl bash tools/env/_py14.sh render_frames.py        needs temp/e273/report/frame_metrics.json (report_data.py)
"""
import json
import os
import sys

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
for sub in ("tools/viz", "training/grid", "training/data", "training", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import qgraph_view as QV  # noqa: E402
import qpredict as QP  # noqa: E402

OUT = os.path.join(ROOT, "temp", "e273", "report")
rows = json.load(open(os.path.join(OUT, "frame_metrics.json")))["s0"]
ok = [r for r in rows if r["n_pieces"] >= 8 and r["topo_F1"] is not None]
ok.sort(key=lambda r: r["topo_F1"])
pick = {}
pick["worst_a"] = ok[0]
seen = {ok[0]["scene"]}
for r in ok[1:]:
    if r["scene"] not in seen:
        pick["worst_b"] = r
        break
pick["median"] = ok[len(ok) // 2]
pick["best"] = ok[-1]
junc = [r for r in rows if r["n_split"] + r["n_merge"] >= 3 and r["junction_F1"] is not None]
junc.sort(key=lambda r: r["junction_F1"])
if junc:
    pick["junction_miss"] = junc[0]
fn = QP.make_pred_fn("e273_p_s0", "error", "cpu", keep_maps=True)
d = os.path.join(OUT, "frames")
os.makedirs(d, exist_ok=True)
sel = {}
for k, r in pick.items():
    path = os.path.join(d, f"{k}_{r['token'][:8]}.png")
    QV.render_frame(r["token"], fn(r["token"]), path, cams=True)
    sel[k] = dict(r, file=os.path.basename(path))
    print(k, r["token"][:8], "topo", r["topo_F1"], "F1@1", r["F1_1"], flush=True)
json.dump(sel, open(os.path.join(OUT, "frame_selection.json"), "w"), indent=1, default=float)
