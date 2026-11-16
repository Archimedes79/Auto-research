import os, pickle, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("BFLC_ROOT", ROOT)
for sub in ("evaluation", "training", "training/grid", "training/data", ".."):
    sys.path.insert(0, os.path.join(ROOT, sub))
sys.path.insert(0, os.path.join(ROOT, "temp", "jr"))
import qends_eval as QEV
import snapconn as SC
import qgraph_eval as QGE
import qgraph_readout as QR
frames = QGE.val_frames()
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", "predfull_e273_p_s0.pkl"), "rb"))
ends = {f["row"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", "e308_t_s0", "ends_val.pkl"), "rb"))}
spec = QR.parse("close:1.5@0.5")
n = diff = 0
for f, d in list(zip(frames, dump))[:150]:
    pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")), links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
    graph, bk = QGE.readout(pred, 0.5, 0.5)
    e = ends[f.row]
    k = e["kp_pos"].shape[0] // 2
    p = 1 / (1 + np.exp(-e["logit"])) * e["allowed"]
    ia, ib = np.nonzero(p >= 0.2)
    prs = [(e["kp_pos"][i], e["kp_dir"][i], e["kp_pos"][k + j], e["kp_dir"][k + j]) for i, j in zip(ia, ib)]
    a = QEV.add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), prs, snap=True)
    b = SC.add_connectors_snap(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), prs)
    n += 1
    same = a[1:] == b[1:] and a[0]["edges"] == b[0]["edges"] and len(a[0]["pts"]) == len(b[0]["pts"]) and all(np.array_equal(x, y) for x, y in zip(a[0]["pts"], b[0]["pts"]))
    diff += int(not same)
print("frames", n, "different", diff)
