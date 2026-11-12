"""E276 pre-experiment P1 (diagnostic, CPU, no model): what do the NODE TARGETS themselves allow? A perfect network ("oracle": the GT nodes and links of a target decoded back to Hermite curves, read out
as the head's output is) is scored by the scorer on the 873 VAL frames for the E273 target (a node in every 2 m cell) and for sparse targets (Douglas-Peucker deviation eps, longest chord). Numbers:
vertices per 100 m after Douglas-Peucker (`vert100_dp`, the budget basis 14.3), nodes per 100 m of GT lane, topo F1, F1@1.0, junction F1, lanes found, Q. An oracle is a sanity check, not an upper bound.
Answers: is the budget reachable with a node target at all, and does the sparse target lose geometry (card risk 6, the G0 gate of E273)?
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh exp_density_oracle.py [n_frames]
"""
import json
import os
import sys
import time

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qformat  # noqa: E402
import qgraph_eval as QE  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
frames = QE.val_frames()[: n or None]
variants = {"dense (E273)": None, "sparse 0.3 / 7.5 m": (0.3, 7.5), "sparse 0.3 / 15 m": (0.3, 15.0), "sparse 0.6 / 7.5 m": (0.6, 7.5)}
out = {}
t0 = time.time()
for name, sp in variants.items():
    QE.NODE_SPARSE = sp                                                                    # the scorer's GT nodes follow the target
    preds = []
    for f in frames:
        g = f.lane()
        preds.append(qformat.oracle_prediction(g, with_assign=False, sparse=sp))
        for p in preds[-1:]:
            p.pop("table", None)
        f._lane = f._ref = None
    for f in frames:
        f._lane = f._ref = None
    res = QE.score_frames(preds, frames)
    out[name] = dict(graph={k: res["graph"][k] for k in ("vert100_dp", "topo_F1", "F1@1.0", "junction_F1", "lanes_found", "Q", "dup_frac")}, nodes_per_100m=res["nodes"].get("nodes_per_100m"),
                     node_R=res["nodes"].get("node_R"))
    print(f"{name}: {time.time() - t0:.0f} s", flush=True)
print(f"\nVAL {len(frames)} frames, an oracle decoded from each node target, scored by qgraph_eval (no model; not an upper bound)")
print("| node target | nodes per 100 m GT lane | vert100_dp (budget 14.3) | topo F1 | F1@1.0 | junction F1 | lanes found | Q | dup_frac |")
print("|---|---|---|---|---|---|---|---|---|")
for name, r in out.items():
    g = r["graph"]
    print(f"| {name} | {r['nodes_per_100m']:.1f} | {g['vert100_dp']:.1f} | {g['topo_F1']:.4f} | {g['F1@1.0']:.4f} | {g['junction_F1']:.4f} | {g['lanes_found']:.4f} | {g['Q']:.4f} | {g['dup_frac']:.4f} |")
json.dump(out, open(os.path.join(ROOT, "temp", "e276", "density_oracle.json"), "w"), indent=1, default=float)
