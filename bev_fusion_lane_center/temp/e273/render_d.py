"""one-off: the best graph so far (d = recall_sp0_dp, its stored VAL graphs) in the layout of the E273 pictures (CPU)

    wsl bash tools/env/_py14.sh render_d.py test              one frame, numbers compared with the raw graph
    wsl bash tools/env/_py14.sh render_d.py scenes 11 39 ...  every frame of the scenes -> results/E273.report/d_frames/<scene>/ + mp4
The graph is adapted to the prediction format of qformat: nodes = the vertices of d, links = its edges drawn as straight segments, node flags from the degrees (split: 2 successors, merge: 2
predecessors, start / end), a section = a chain between junction vertices. d has no map assignment (the ids panel then shows the sections).
"""
import os
import sys

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
for sub in ("tools/viz", "training/grid", "training/data", "training", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import lane_suite as LS  # noqa: E402
import qgraph_eval as QE  # noqa: E402
import qgraph_view as QV  # noqa: E402


def raw_to_pred(g):
    """g: pts = list of polylines (m, 2), nodes = the global vertex ids of every polyline, edges = (a, b): polyline a continues into polyline b"""
    pos_of, sec_of = {}, {}
    for pi, (pl, ids) in enumerate(zip(g["pts"], g["nodes"])):
        for v, xy in zip(ids, np.asarray(pl, float)):
            pos_of.setdefault(int(v), xy)
            sec_of.setdefault(int(v), pi)
    ids_sorted = sorted(pos_of)
    idx = {v: i for i, v in enumerate(ids_sorted)}
    pts = np.array([pos_of[v] for v in ids_sorted]).reshape(-1, 2)
    n = len(pts)
    links = set()
    for ids in g["nodes"]:
        for a_, b_ in zip(ids[:-1], ids[1:]):
            if a_ != b_:
                links.add((idx[int(a_)], idx[int(b_)]))
    for a_, b_ in g["edges"]:
        u, w = int(g["nodes"][a_][-1]), int(g["nodes"][b_][0])
        if u != w:
            links.add((idx[u], idx[w]))
    ed = np.array(sorted(links), int).reshape(-1, 2)
    out_deg, in_deg = np.zeros(n, int), np.zeros(n, int)
    succ = [[] for _ in range(n)]
    pred = [[] for _ in range(n)]
    for a_, b_ in ed:
        out_deg[a_] += 1
        in_deg[b_] += 1
        succ[a_].append(b_)
        pred[b_].append(a_)
    dirv = np.zeros((n, 2))
    for i in range(n):
        v = sum((pts[j] - pts[i] for j in succ[i]), np.zeros(2)) if succ[i] else sum((pts[i] - pts[j] for j in pred[i]), np.zeros(2))
        dirv[i] = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-9 else (0.0, 1.0)
    flags = np.zeros((n, 8))
    flags[out_deg >= 2, 2] = 1.0                                              # split
    flags[in_deg >= 2, 3] = 1.0                                               # merge
    flags[in_deg == 0, 4] = 1.0                                               # start
    flags[out_deg == 0, 5] = 1.0                                              # end
    sec = np.array([sec_of[v] for v in ids_sorted], int)
    cell = np.clip(np.floor((pts + 50.0) / 2.0).astype(int), 0, 49)
    L = len(ed)
    ln = np.linalg.norm(pts[ed[:, 1]] - pts[ed[:, 0]], axis=1) if L else np.zeros(0)
    nodes = dict(pos=pts, dirv=dirv, e=np.ones(n), flags=flags, s=(in_deg != 1).astype(float), cell=cell, section=sec)
    lk = dict(src=ed[:, 0].copy() if L else np.zeros(0, int), dst=ed[:, 1].copy() if L else np.zeros(0, int), p=np.ones(L), long=ln > 8.0, a=np.full(L, 1 / 3), b=np.full(L, 1 / 3),
              curves=[np.stack([pts[a_], pts[b_]]) for a_, b_ in ed])
    return dict(nodes=nodes, links=lk, assign=None, maps=None)


def main():
    mode = sys.argv[1]
    sub = LS.GraphSubject("recall_sp0_dp")
    frames = QE.val_frames()
    by_tok = {f.token: f for f in frames}
    fn = lambda token: raw_to_pred(sub.graphs[by_tok[token].row])  # noqa: E731
    if mode == "test":
        for k in (150, 400, 700):
            f = frames[k]
            raw = QE.graph_numbers([QE.graph_counts(sub.graphs[f.row], f)])
            ad = QE.score_frames([fn(f.token)], [f])["graph"]
            print(k, "raw", {m: round(raw[m], 4) for m in ("F1@1.0", "topo_F1", "junction_F1", "lanes_found")}, "adapted", {m: round(ad[m], 4) for m in ("F1@1.0", "topo_F1", "junction_F1", "lanes_found")}, flush=True)
        QV.render_frame(frames[150].token, fn(frames[150].token), os.path.join(ROOT, "temp", "e273", "report", "d_test.png"), cams=True)
        return
    out = os.path.join(ROOT, "results", "E273.report", "d_frames")
    for sc in sys.argv[2:]:
        sc = int(sc) if sc.isdigit() else sc
        tmp = os.path.join(out, str(sc))
        os.makedirs(tmp, exist_ok=True)
        QV.render_scene(sc, tmp, pred_fn=fn, kind="recall_sp0_dp", video=True, every=1)
        print("scene", sc, "done", flush=True)


if __name__ == "__main__":
    main()
