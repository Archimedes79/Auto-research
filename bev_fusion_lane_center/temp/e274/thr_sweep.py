"""one-off (E274): the read-out thresholds of the E273 head on the 873 VAL frames: node existence e >= t_e and link p >= t_l, no retraining (the predictions keep every node with e >= 0.02)

    wsl bash tools/env/_py14.sh thr_sweep.py e273_p_s0
"""
import json
import os
import sys

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/train", "evaluation", "tools/viz"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402
import qgraph_run_eval as QR  # noqa: E402

run = sys.argv[1]
frames = QE.val_frames()
ds, preds = QR.predict_val(run, "exact", 0, "cuda")
out = {}
for te, tl in ((0.5, 0.5), (0.4, 0.5), (0.3, 0.5), (0.2, 0.5), (0.3, 0.4), (0.3, 0.3), (0.4, 0.4), (0.5, 0.4), (0.5, 0.3)):
    r = QE.score_frames(preds, frames, thr_e=te, thr_link=tl)
    g, n = r["graph"], r["nodes"]
    out[f"e{te}_l{tl}"] = dict(topo_F1=g["topo_F1"], F1_1=g["F1@1.0"], junction_F1=g["junction_F1"], lanes_found=g["lanes_found"], Q=g["Q"], dup=g["dup_frac"], vert100_dp=g["vert100_dp"],
                               node_P=n["node_P"], node_R=n["node_R"], link_short_P=n["link_short_P"], link_short_R=n["link_short_R"], link_long_P=n["link_long_P"], link_long_R=n["link_long_R"])
    print(f"e>={te} p>={tl}: topo {g['topo_F1']:.3f} F1@1 {g['F1@1.0']:.3f} junction {g['junction_F1']:.3f} lanes {g['lanes_found']:.3f} Q {g['Q']:.3f} dup {g['dup_frac']:.3f} vert {g['vert100_dp']:.1f} | "
          f"node P {n['node_P']:.3f} R {n['node_R']:.3f} | short P {n['link_short_P']:.2f} R {n['link_short_R']:.2f} long P {n['link_long_P']:.2f} R {n['link_long_R']:.2f}", flush=True)
os.makedirs(os.path.join(ROOT, "temp", "e274"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "temp", "e274", f"thr_sweep_{run}.json"), "w"), indent=1)
