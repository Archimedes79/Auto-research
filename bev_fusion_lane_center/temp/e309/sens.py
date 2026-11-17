"""E309: sensitivity of the topo metric (pooled over the 873 VAL frames, seed given) to its tolerances: does the order B / +E284 / +E287 p2 change?  usage: sens.py <seed>"""
import json, os, pickle, sys, time
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training/grid", "training", "training/data", "evaluation/sparse_lanes", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import qends_eval as QEE, qgraph_eval as QE, qgraph_readout as QR, topo_metrics as TM
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
RUNS = {"e312": f"e312_p_s{seed}"}
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{seed}.pkl"), "rb"))
ends = {k: {f["row"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", r, "ends_val.pkl"), "rb"))} for k, r in RUNS.items()}
frames = QE.val_frames()
spec = QR.parse("close:1.5@0.5")
graphs = []
for f, d in zip(frames, dump):
    pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")), links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
    graph, bk = QE.readout(pred, 0.5, 0.5)
    v = {"B": graph}
    for k in ends:
        e = ends[k][f.row]; kk = e["kp_pos"].shape[0] // 2
        p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
        ia, ib = np.nonzero(p >= 0.1)
        pairs = [(e["kp_pos"][i], e["kp_dir"][i], e["kp_pos"][kk + j], e["kp_dir"][kk + j]) + (tuple(e["scale"][i, j]) if e.get("scale") is not None else ()) for i, j in zip(ia, ib)]
        raw, _, _ = QEE.add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), pairs, snap=True)
        v[k] = raw
    graphs.append(v)
CFG = {"ref (snap 2, tol 1, dir 60deg, reach 30)": dict(),
       "tol 2 (snap 3)": dict(TOL=2.0, SNAP=3.0),
              "tol 2 + no direction": dict(TOL=2.0, SNAP=3.0, COS_SNAP=-1.0),
       "tol 3 (snap 4)": dict(TOL=3.0, SNAP=4.0)}
res = {}
for name, over in CFG.items():
    saved = {k: getattr(TM, k) for k in over}
    for k, x in over.items(): setattr(TM, k, x)
    res[name] = {}
    for var in ("B", "e312"):
        cs = [QE.graph_counts(g[var], f) for g, f in zip(graphs, frames)]
        for f in frames: f._lane = f._ref = None
        n = QE.graph_numbers(cs)
        res[name][var] = {k: float(n[k]) for k in ("topo_F1", "topo_P", "topo_R", "F1@1.0", "lanes_found", "junction_F1", "dup_frac")}
    for k, x in saved.items(): setattr(TM, k, x)
    print(name, {v: round(res[name][v]["topo_F1"], 4) for v in res[name]}, flush=True)
json.dump(res, open(os.path.join(ROOT, "temp", "e309", f"sens_e312_s{seed}.json"), "w"), indent=1)
print("done")
