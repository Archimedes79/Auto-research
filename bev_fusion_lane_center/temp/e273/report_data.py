"""one-off: numbers for the E273 report: per-frame metrics of both seeds (error map), situation breakdown by the number of GT junction events, and the node-recall ceiling on the oracle graph

    wsl bash tools/env/_py14.sh report_data.py      (GPU for the predictions, CPU for the rest; about 10 min)
writes temp/e273/report/{frame_metrics.json,node_drop.json}
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation", "tools/viz"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qformat as QF  # noqa: E402
import qgraph_eval as QE  # noqa: E402
import qgraph_run_eval as QR  # noqa: E402

OUT = os.path.join(ROOT, "temp", "e273", "report")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
frames = QE.val_frames()
print("frames", len(frames), flush=True)

# GT situation of every frame: split / merge events and pieces of the lane GT
sit = []
for f in frames:
    g = f.lane()
    ev = [e["type"] for e in g.get("events", [])]
    sit.append(dict(n_split=ev.count("split"), n_merge=ev.count("merge"), n_cross=ev.count("cross"), n_pieces=len(g["pieces"]),
                    n_conn=sum(1 for p in g["pieces"] if p.get("kind") == "connector"), has_inter=bool(len(g.get("areas", {}).get("intersection", [])) if isinstance(g.get("areas"), dict) else False)))
    f._lane = f._ref = None
print("situations", time.time() - t0, flush=True)

res = {}
for seed in (0, 1):
    run = f"e273_p_s{seed}"
    ds, preds = QR.predict_val(run, "error", 0, "cuda")
    out, rec = QE.score_frames(preds, frames, records=True)
    rows = []
    for i, f in enumerate(frames):
        gr = QE.graph_numbers([rec["graph"][i]])
        row = dict(i=i, token=f.token, scene=int(f.scene), F1_1=gr.get("F1@1.0"), topo_F1=gr.get("topo_F1"), junction_F1=gr.get("junction_F1"), lanes_found=gr.get("lanes_found"),
                   Q=gr.get("Q"), vert100=gr.get("vert100"), dup=gr.get("dup_frac"))
        try:
            row["node_F1"] = QE.node_numbers(QE.pool_counts([rec["node"][i]])).get("node_F1")
            row["lane_acc"] = QE.assign_numbers(QE.pool_counts([rec["assign"][i]])).get("lane_acc") if len(rec["assign"]) == len(frames) else None
        except Exception as e:  # noqa: BLE001
            row["err"] = str(e)[:80]
        row.update(sit[i])
        rows.append(row)
    res[f"s{seed}"] = rows
    print("seed", seed, "scored", time.time() - t0, "pooled topo", out["graph"].get("topo_F1"), flush=True)
json.dump(res, open(os.path.join(OUT, "frame_metrics.json"), "w"), default=float)

# node-recall ceiling: the oracle graph with a share of the nodes removed (event nodes only, or any node)
rng = np.random.RandomState(0)
sub = frames[::3]
drop = {}
for mode in ("any", "events"):
    for rate in (0.0, 0.05, 0.10, 0.17, 0.25, 0.35):
        preds = []
        for f in sub:
            g = f.lane()
            p = QF.oracle_prediction(g, with_assign=False)
            e = p["nodes"]["e"]
            cand = np.arange(len(e)) if mode == "any" else np.flatnonzero(np.asarray(p["nodes"]["flags"]).max(1) > 0.5)
            k = int(round(rate * len(cand)))
            if k:
                e[rng.choice(cand, k, replace=False)] = 0.0
            preds.append(p)
            f._lane = f._ref = None
        o = QE.score_frames(preds, sub)["graph"]
        drop[f"{mode}_{rate}"] = {k: o.get(k) for k in ("topo_F1", "F1@1.0", "junction_F1", "lanes_found")}
        print(mode, rate, drop[f"{mode}_{rate}"], flush=True)
json.dump(drop, open(os.path.join(OUT, "node_drop.json"), "w"), default=float)
print("done", time.time() - t0)
