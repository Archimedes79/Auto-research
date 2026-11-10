"""E276 P11: where is the remaining gap of baseline B, in the links or in the nodes? Oracle link selections on the FULL dump (exact map), CPU only, a diagnostic (an oracle is a sanity check, not an upper bound):
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh exp_oracle_links.py <dump pkl> <run name for the reference card> [--rows N]
Every delivered link of B (p >= 0.5, both ends e >= 0.5) is, with the matching of the P9 analysis (the nodes matched one-to-one to GT nodes within 2 m; a link is RIGHT when its two GT nodes are joined by a GT link):
  right (0), wrong between two matched nodes (1), an end delivered but unmatched (2: a false node, or a surplus node of a denser row: the one-to-one match leaves one of two close nodes unmatched), an end below e 0.5 (3). Variants scored by the E273 scorer (graph numbers), nodes unchanged:
  B            all links (reproduces the card),
  only right   keep category 0 (every kept link is right: the links among the matched nodes that B has),
  no wrong     drop category 1 (the wrong links between matched nodes), keep 0 and 2,
  no unmatched-end drop category 2, keep 0, 1 and 3.
Output: temp/e276/oraclelinks_<run>.json and a table."""
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
from lane_task import measures as MS  # noqa: E402

dump_path, run = sys.argv[1], sys.argv[2]
dump_path = dump_path if os.path.isabs(dump_path) else os.path.join(ROOT, dump_path)
n_rows = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 0
dump = pickle.load(open(dump_path, "rb"))
frames = QE.val_frames()[: len(dump)]
if n_rows:
    dump, frames = dump[:n_rows], frames[:n_rows]
VARIANTS = {"B": (0, 1, 2, 3), "only right": (0,), "no wrong": (0, 2, 3), "no unmatched-end": (0, 1, 3)}
rec = {v: [] for v in VARIANTS}
cats = np.zeros(4)
for d, f in zip(dump, frames):
    assert int(d["row"]) == int(f.row)
    ref = f.ref()
    nt = ref.nt
    keep = d["e"] >= 0.5
    P = d["pos"][keep]
    kidx = np.flatnonzero(keep)
    local = -np.ones(len(d["e"]), int)
    local[kidx] = np.arange(len(kidx))
    g_of_p = -np.ones(len(P), int)
    if len(P) and len(nt["pos"]):
        pairs = MS.match_1to1(P, nt["pos"], QE.NODE_TOL)
        if len(pairs):
            g_of_p[pairs[:, 0]] = pairs[:, 1]
    gt_link = {(int(a), int(b)) for a, b in zip(nt["src"], nt["dst"])}
    p = np.asarray(d["link_p"], float)
    cat = np.full(len(p), 2, int)
    for i, (s_, t_) in enumerate(zip(d["link_src"], d["link_dst"])):
        a, b = local[s_], local[t_]
        if a < 0 or b < 0:
            cat[i] = 3                                                    # an end below e 0.5: not delivered (the read-out drops the link anyway)
            continue
        if g_of_p[a] < 0 or g_of_p[b] < 0:
            continue
        cat[i] = 0 if (int(g_of_p[a]), int(g_of_p[b])) in gt_link else 1
    on = p >= 0.5
    for k in range(4):
        cats[k] += int(((cat == k) & on).sum())
    nd = dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"])
    if d.get("s") is not None:
        nd["s"] = d["s"]
    cached = getattr(f, "_ref", None) is not None
    for name, ks in VARIANTS.items():
        use = on & np.isin(cat, ks)
        lk = dict(src=d["link_src"], dst=d["link_dst"], p=np.where(use, p, 0.0), long=np.asarray(d["link_long"], bool), a=d["link_a"], b=d["link_b"])
        graph, _ = QE.readout(dict(nodes=nd, links=lk), 0.5, 0.04)
        rec[name].append(QE.graph_counts(graph, f))
    if not cached:
        f._lane = f._ref = None
keys = ("topo_F1", "topo_P", "topo_R", "junction_F1", "junction_P", "junction_R", "F1@1.0", "lanes_found", "Q", "dup_frac", "vert100_dp")
out = {}
print(f"links p >= 0.5: right {int(cats[0])}, wrong between matched nodes {int(cats[1])}, an end delivered but unmatched {int(cats[2])}, an end not delivered (e < 0.5, dropped by the read-out) {int(cats[3])}")
print("variant      | " + " ".join(k.rjust(11) for k in keys))
for name in VARIANTS:
    g = QE.graph_numbers(rec[name])
    out[name] = {k: g.get(k) for k in keys}
    print(f"{name:12} | " + " ".join(("%11.4f" % g[k]) if g.get(k) is not None else "          -" for k in keys))
ref_card = None
card = os.path.join(ROOT, "runs", "e276", run, "card.json")
if os.path.exists(card) and not n_rows:
    ref_card = json.load(open(card))["p"]["graph"]
    diff = {k: out["B"][k] - ref_card[k] for k in ("topo_F1", "junction_F1", "F1@1.0", "lanes_found", "Q", "vert100_dp")}
    print("reproduction gate (B minus the scoring card):", {k: round(v, 5) for k, v in diff.items()}, "OK" if max(abs(x) for x in diff.values()) < 0.002 else "FAILED")
json.dump(dict(run=run, frames=len(dump), link_categories=dict(right=int(cats[0]), wrong_matched=int(cats[1]), unmatched_end=int(cats[2]), not_delivered_end=int(cats[3])), variants=out, ref_card=ref_card),
          open(os.path.join(ROOT, "temp", "e276", f"oraclelinks_{run}.json"), "w"), indent=1)
