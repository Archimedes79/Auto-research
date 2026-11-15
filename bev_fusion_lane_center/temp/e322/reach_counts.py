"""E322 scratch: how many delivered piece ends lie in reach (4 m / 45 degrees, the wiring gate of add_connectors) of the connector ends of a run, i.e. how often the one-piece rule has a choice. usage: reach_counts.py <run> <seed> [--attach]"""
import os, pickle, sys
import numpy as np
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import qgraph_eval as QE
import qgraph_readout as QR
from qends_eval import _tan, _ang, WIRE_POS, WIRE_ANG
run, seed = sys.argv[1], int(sys.argv[2])
attach = "--attach" in sys.argv
full = os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{seed}.pkl")
dump = pickle.load(open(full, "rb"))
ends = {f["row"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "rb"))}
frames = QE.val_frames()
spec = QR.parse("close:1.5@0.5")
hist = {"in": np.zeros(6, int), "out": np.zeros(6, int)}
gaps = []                                                                                                   # distance of the second nearest minus the nearest, ends with >= 2 candidates
for f, d in zip(frames, dump):
    pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                         links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
    graph, bk = QE.readout(pred, 0.5, 0.5)
    L = [np.asarray(P, float) for P in graph.lines if len(P)]
    ends_ = [(P[-1], _tan(P, True), P[0], _tan(P, False)) for P in L]
    e = ends[f.row]
    k = e["kp_pos"].shape[0] // 2
    p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
    ia, ib = np.nonzero(p >= 0.1)
    qp, qd = (e["att_pos"], e["att_dir"]) if attach else (e["kp_pos"], e["kp_dir"])
    for i, j in zip(ia, ib):
        for key, idx, pos_i, tan_i in (("in", i, 0, 1), ("out", k + j, 2, 3)):
            ds = sorted(float(np.linalg.norm(en[pos_i] - qp[idx])) for en in ends_ if np.linalg.norm(en[pos_i] - qp[idx]) <= WIRE_POS and _ang(en[tan_i], qd[idx]) <= WIRE_ANG)
            hist[key][min(len(ds), 5)] += 1
            if len(ds) >= 2:
                gaps.append(ds[1] - ds[0])
n = {k: hist[k].sum() for k in hist}
print(run, "attached ends" if attach else "keypoint ends", "pairs", n["in"])
for k in hist:
    print(f"  {k:3s} candidates in reach 0 / 1 / 2 / 3 / 4 / 5+: " + " / ".join(f"{100 * h / n[k]:.1f} %" for h in hist[k]))
g = np.asarray(gaps)
print(f"  ends with >= 2 candidates: {len(g)}; second minus nearest distance: median {np.median(g):.2f} m, share under 0.5 m {100 * np.mean(g < 0.5):.0f} %, under 1 m {100 * np.mean(g < 1.0):.0f} %")
