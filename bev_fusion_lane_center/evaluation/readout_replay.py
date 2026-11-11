"""Replay of node read-out steps on the stored full prediction dump of the lane head B (E288): CPU only, no model, no training.

    wsl bash tools/env/_py14.sh readout_replay.py score --seed 0 --specs plain@0.5 close:1.5@0.5 mnms:1.5@0.5 [--full temp/e276/predfull_e273_p_s0.pkl] [--out runs/e288/replay_s0.json]
    wsl bash tools/env/_py14.sh readout_replay.py selftest

Every spec of `qgraph_readout` (plain / close / mnms) is applied to the candidate nodes of every VAL frame (the dump holds e >= 0.2) and the graph is read out and scored exactly as `qends_eval.cmd_score` scores the plain graph
(`qgraph_eval.readout`, `graph_counts`, `graph_numbers`, VAL 873 frames, link threshold 0.5). The dump is the one written by `qgraph_run_eval.py score --dump-full` (scratch, `temp/e276/predfull_e273_p_s<seed>.pkl`).
Output: one row per spec with the lane suite numbers (Q, F1@1.0, lanes found, topo F1 / P / R, junction F1 / P / R, dup_frac, vertices per 100 m) and the number of delivered nodes.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))

KEYS = ("Q", "F1@1.0", "lanes_found", "topo_F1", "topo_P", "topo_R", "junction_F1", "junction_P", "junction_R", "dup_frac", "vert100_dp")


def pred_of(d):
    return dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"]))


def score(a):
    import qgraph_eval as QE
    import qgraph_readout as QR
    full = a.full or os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{a.seed}.pkl")
    dump = pickle.load(open(full, "rb"))
    frames = QE.val_frames()
    assert [d["row"] for d in dump] == [f.row for f in frames], "the full dump is not in the order of the VAL frames"
    res = []
    for text in a.specs:
        spec = QR.parse(text)
        rec, n_nodes, t0 = [], 0, time.time()
        for f, d in zip(frames, dump):
            pred = QR.apply(pred_of(d), spec)
            n_nodes += int((np.asarray(pred["nodes"]["e"], float) >= spec.thr).sum())
            graph, _ = QE.readout(pred, 0.5, 0.5)
            rec.append(QE.graph_counts(graph, f))
            f._lane = f._ref = None
        num = QE.graph_numbers(rec)
        row = dict(spec=text, nodes=n_nodes, seconds=round(time.time() - t0), **{k: float(num[k]) for k in KEYS if k in num})
        res.append(row)
        print(f"{text:18s} nodes {n_nodes:6d} " + " ".join(f"{k} {row[k]:.4f}" for k in KEYS if k in row), flush=True)
    out = a.out or os.path.join(ROOT, "runs", "e288", f"replay_s{a.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(seed=a.seed, full=full, frames=len(frames), rows=res), open(out, "w"), indent=1)
    print("written", out)


def selftest():
    import qgraph_readout as QR
    s = QR.parse("mnms:1.5@0.5")
    pos = np.array([[0, 0], [0.3, 0], [5, 5]], float)
    e = np.array([0.95, 0.9, 0.8])
    q = QR.apply(dict(nodes=dict(pos=pos, e=e)), s)
    assert q["nodes"]["e"][1] == 0.0 and q["nodes"]["e"][0] == 0.95 and q["nodes"]["e"][2] == 0.8
    print("readout_replay selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("score")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--specs", nargs="+", required=True)
    s.add_argument("--full", default=None)
    s.add_argument("--out", default=None)
    sub.add_parser("selftest")
    a = ap.parse_args()
    selftest() if a.cmd == "selftest" else score(a)
