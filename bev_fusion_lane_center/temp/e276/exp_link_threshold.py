"""E276 P10 (video note 8, read-out only): what do a higher link threshold and a separate threshold for the long links do to the GRAPH numbers of baseline B? From the FULL prediction dump
(`temp/e276/predfull_<run>.pkl`, exact map, nodes e >= 0.2, links p >= 0.05), CPU only, no model:
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh exp_link_threshold.py <dump pkl> <run name for the reference card> [--rows N]
Every variant is (threshold of the short links, threshold of the long links); nodes are read at e >= 0.5 as in the card of B. The variant (0.5, 0.5) must reproduce the graph numbers of the scoring card
runs/e276/<run>/card.json (reproduction gate, 0.002). A diagnostic by hand, in-sample on VAL 873, no verdict; a hard selection on the output (D55 stop-gap).
Output: temp/e276/linkthr_<run>.json and a table on stdout."""
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402

dump_path, run = sys.argv[1], sys.argv[2]
dump_path = dump_path if os.path.isabs(dump_path) else os.path.join(ROOT, dump_path)
n_rows = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 0
dump = pickle.load(open(dump_path, "rb"))
frames = QE.val_frames()[: len(dump)]
if n_rows:
    dump, frames = dump[:n_rows], frames[:n_rows]
VARIANTS = [(0.5, 0.5), (0.7, 0.7), (0.9, 0.9), (0.5, 0.7), (0.5, 0.9), (0.5, 1.01)]   # (short, long); 1.01 = no long link (a smoke on 20 frames showed 11 variants take 33 min per seed: six are enough)
rec = {v: [] for v in VARIANTS}
for d, f in zip(dump, frames):
    assert int(d["row"]) == int(f.row), (d["row"], f.row)
    nd = dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"])
    if d.get("s") is not None:
        nd["s"] = d["s"]
    p = np.asarray(d["link_p"], float)
    lg = np.asarray(d["link_long"], bool)
    cached = getattr(f, "_ref", None) is not None
    for (ts, tl) in VARIANTS:
        keep_link = np.where(lg, p >= tl, p >= ts)
        lk = dict(src=d["link_src"], dst=d["link_dst"], p=np.where(keep_link, p, 0.0), long=lg, a=d["link_a"], b=d["link_b"])
        graph, _ = QE.readout(dict(nodes=nd, links=lk), 0.5, 0.04)
        rec[(ts, tl)].append(QE.graph_counts(graph, f))
    if not cached:
        f._lane = f._ref = None
out = {}
keys = ("topo_F1", "topo_P", "topo_R", "junction_F1", "junction_P", "junction_R", "F1@1.0", "lanes_found", "Q", "dup_frac", "vert100_dp")
print("short long | " + " ".join(k.rjust(11) for k in keys))
for v in VARIANTS:
    g = QE.graph_numbers(rec[v])
    out[f"{v[0]}_{v[1]}"] = {k: g.get(k) for k in keys}
    print(f"{v[0]:5} {v[1]:4} | " + " ".join(("%11.4f" % g[k]) if g.get(k) is not None else "          -" for k in keys))
ref = None
card = os.path.join(ROOT, "runs", "e276", run, "card.json")
if os.path.exists(card):
    ref = json.load(open(card))["p"]["graph"]
    diff = {k: out["0.5_0.5"][k] - ref[k] for k in ("topo_F1", "junction_F1", "F1@1.0", "lanes_found", "Q", "vert100_dp")}
    print("reproduction gate (variant 0.5 / 0.5 minus the scoring card):", {k: round(v, 5) for k, v in diff.items()}, "OK" if (not n_rows and max(abs(x) for x in diff.values()) < 0.002) else "(not applicable with --rows or off)")
json.dump(dict(run=run, frames=len(dump), variants=out, ref_card=ref), open(os.path.join(ROOT, "temp", "e276", f"linkthr_{run}.json"), "w"), indent=1)
