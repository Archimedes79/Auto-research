"""E276 P8b (video note 2): which GT map pieces (>= 10 m inside the window) are not claimed by the delivered graph, and why; split by tier and camera visibility. From the FULL prediction dump of B
(exact map), CPU only:
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh exp_coverage.py <dump pkl> <l4 store folder> [--rows N]
Per piece, from the delivered nodes (e >= 0.5) that lie on a lane (within LANE_NEAR of a GT point; a node belongs to its nearest piece and to every piece that ties with it, QE.TIE):
  no node      no delivered node has the piece as its nearest (or tied) piece: the head delivered nothing there,
  claimed      at least one of these nodes has this very piece as its most probable map piece,
  shared       none claims this piece but one is assigned to a piece that ties with it (an overlapping piece took the node: the scorer counts that node right),
  misassigned  nodes lie on it but all go to other pieces or the dustbin (a real assignment failure).
The piece's tier is the best (smallest) tier of its points, its visibility the share of its points that a camera sees (vis > 0). Output: temp/e276/coverage_<tag>.json and a table."""
import gzip
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_eval as QE  # noqa: E402

dump_path, store = sys.argv[1], sys.argv[2]
dump_path = dump_path if os.path.isabs(dump_path) else os.path.join(ROOT, dump_path)
n_rows = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 0
dump = pickle.load(open(dump_path, "rb"))
frames = QE.val_frames()[: len(dump)]
if n_rows:
    dump, frames = dump[:n_rows], frames[:n_rows]
tag = os.path.basename(dump_path).replace("predfull_", "").replace(".pkl", "")
CATS = ("no node", "claimed", "shared", "misassigned")
rec = []                                                              # (tier, vis share, category index, length)
for f, d in zip(frames, dump):
    assert d["row"] == f.row
    g = pickle.load(gzip.open(os.path.join(store, f.token + ".pkl.gz"), "rb"))
    ref = QE.LaneRef(g)
    keep = d["e"] >= 0.5
    P = d["pos"][keep]
    ids = d["map_ids"]
    Kn = len(ids)
    amax = d["assign_P"][keep].argmax(1) if keep.any() else np.zeros(0, int)
    gt_ids = ref.ids
    dist, tied, _ = ref.near(P) if len(P) else (np.zeros(0), [], None)
    near_of = {}                                                      # piece index -> the nodes that lie on it
    for n in range(len(P)):
        if dist[n] <= QE.LANE_NEAR:
            for j in tied[n]:
                near_of.setdefault(int(j), []).append(n)
    for j, p in enumerate(g["pieces"]):
        if p["length"] < 10.0:
            continue
        nodes = near_of.get(j, [])
        if not nodes:
            cat = 0
        else:
            claimed = any(amax[n] < Kn and ids[amax[n]] == gt_ids[j] for n in nodes)
            if claimed:
                cat = 1
            else:
                shared = any(amax[n] < Kn and ids[amax[n]] in {gt_ids[t] for t in tied[n]} for n in nodes)
                cat = 2 if shared else 3
        rec.append((int(np.asarray(p["tier"]).astype(int).min()), float((np.asarray(p["vis"]) > 0).mean()), cat, float(p["length"])))
rec = np.array(rec, float)
print("pieces >= 10 m:", len(rec))


def line(name, m):
    if not m.any():
        return dict(group=name, n=0)
    c = rec[m, 2].astype(int)
    return dict(group=name, n=int(m.sum()), **{CATS[k]: float((c == k).mean()) for k in range(4)})


groups = [("all", np.ones(len(rec), bool))]
for t in (1, 2, 3):
    groups.append((f"tier {t}", rec[:, 0] == t))
    groups.append((f"tier {t}, seen by a camera (vis >= 0.5)", (rec[:, 0] == t) & (rec[:, 1] >= 0.5)))
    groups.append((f"tier {t}, not seen (vis < 0.5)", (rec[:, 0] == t) & (rec[:, 1] < 0.5)))
groups.append(("seen by a camera, any tier", rec[:, 1] >= 0.5))
groups.append(("not seen, any tier", rec[:, 1] < 0.5))
out = [line(n, m) for n, m in groups]
print("| group | pieces | " + " | ".join(CATS) + " |")
print("|---|---|---|---|---|---|")
for o in out:
    if o["n"]:
        print(f"| {o['group']} | {o['n']} | " + " | ".join(f"{o[c]:.3f}" for c in CATS) + " |")
json.dump(dict(dump=tag, frames=len(dump), groups=out), open(os.path.join(ROOT, "temp", "e276", f"coverage_{tag}.json"), "w"), indent=1)
