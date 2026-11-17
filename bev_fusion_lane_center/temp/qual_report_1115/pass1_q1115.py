"""scratch (2026-11-15, qualitative report): the delivered graphs of B (plain), E312 (snapped, p >= 0.1) and E324 s (snapped, p >= 0.1) on the 873 VAL frames of seed 0, read exactly as
`qends_eval.cmd_score` reads them (B's full dump + the run's ends_val.pkl, make_pairs, add_connectors snap=True), the per-frame counts checked against the stored counts pickles of the cards, the maneuver
counts of `jmaneuver.frame_counts` per frame, the lane GT pieces (with tier / visibility) and the scene descriptions.  CPU only, one process.
usage: wsl bash tools/env/_py14.sh pass1_q1115.py   -> temp/qual_report_1115/pass1.pkl  (never touches runs/)"""
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training/grid", "training", "training/data", "evaluation/sparse_lanes", "tools/viz", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jmaneuver as JM  # noqa: E402
import qends as QENDS  # noqa: E402
import qends_eval as QEE  # noqa: E402
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as QR  # noqa: E402

OUT = os.path.join(ROOT, "temp", "qual_report_1115")
RUN = {"E312": "e312_p_s0", "E324s": "e324_s_s0"}
dump = pickle.load(open(os.path.join(ROOT, "temp", "e276", "predfull_e273_p_s0.pkl"), "rb"))
ends = {k: {f["row"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", r, "ends_val.pkl"), "rb"))} for k, r in RUN.items()}
stored = {k: pickle.load(open(os.path.join(ROOT, "runs", r, "ends_card_snap_thr0.1.counts.pkl"), "rb")) for k, r in RUN.items()}
frames = QE.val_frames()
assert [d["row"] for d in dump] == [f.row for f in frames]
spec = QR.parse("close:1.5@0.5")
out = []
mism = []
t0 = time.time()
for b, (f, d) in enumerate(zip(frames, dump)):
    pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                         links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
    graph, bk = QE.readout(pred, 0.5, 0.5)
    g = f.lane()
    mans = JM.maneuvers(g, QENDS.cluster)
    rt = f.ref().tree
    rec = dict(k=b, token=f.token, row=f.row, scene=f.scene,
               gt=[dict(pts=np.asarray(p["pts"], np.float32), tier=np.asarray(p.get("tier", np.zeros(len(p["pts"]))), np.int8), vis=np.asarray(p.get("vis", np.ones(len(p["pts"]))), bool),
                        kinds=[(s["kind"], int(s["i0"]), int(s["i1"])) for s in p["segments"]]) for p in g["pieces"] if len(p["pts"]) > 1],
               n_man=len(mans), man_pts=[np.asarray(m[2], np.float32) for m in mans], sys={})
    raws = {"B": QE.to_raw(graph)}
    for k in RUN:
        e = ends[k][f.row]
        kk = e["kp_pos"].shape[0] // 2
        p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
        ia, ib = np.nonzero(p >= 0.1)
        pairs = QEE.make_pairs(e, ia, ib, kk, attach=False, chord=False)
        raws[k], _, _ = QEE.add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), pairs, snap=True)
    for k, raw in raws.items():
        c = QE.graph_counts(graph if k == "B" else raw, f)
        ref = stored["E324s"]["base"][b] if k == "B" else stored[k]["head"][b]
        if not np.allclose(c["topo"]["topo"], ref["topo"]["topo"]):
            mism.append((b, k, c["topo"]["topo"].tolist(), ref["topo"]["topo"].tolist()))
        n_ok, n, hit, n_near, real = JM.frame_counts(raw, mans, rt) if mans else (0, 0, 0, 0, [])
        rec["sys"][k] = dict(lines=[np.asarray(l, np.float32) for l in raw["pts"]], man_ok=n_ok, man_n=n, man_real=[bool(x) for x in real], junc_hit=hit, junc_near=n_near)
    f._lane = f._ref = None
    out.append(rec)
    if (b + 1) % 50 == 0:
        print(f"{b + 1}/{len(frames)} {time.time() - t0:.0f} s  mismatches so far {len(mism)}", flush=True)
# scene descriptions of the VAL scenes (nuScenes scene.json)
import qgraph_view as QV  # noqa: E402
nsc = os.path.expanduser("~/data/nuscenes_full/v1.0-trainval")
scn = {s["name"]: s for s in json.load(open(os.path.join(nsc, "scene.json")))}
info = {}
for sid, (name, toks) in QV.val_scenes().items():
    info[sid] = dict(name=name, description=scn[name]["description"], n=len(toks), tokens=toks)
pickle.dump(dict(frames=out, mismatch=mism, scenes=info), open(os.path.join(OUT, "pass1.pkl"), "wb"))
print("done; count mismatches:", len(mism), mism[:3])
