"""E363 scratch: the FAITHFUL dump and its node check (the critic's blocking finding 1: the Matrix-NMS layer of B' (`QGraph.node_stage`, args soft_nms = ['mnms', radius, sigma]) is NOT inside `nodes.*`, so a pair model with the swapped node head sees the RAW existence logit
in `Attach`; the delivered graph of B' (`predfull_e292_a`) is post-layer).  `LayerNodes` wraps `model.nodes` of the pair model with the same decay (`qmnms.matrix_nms_logit(no['e'], no['pos'], radius, sigma)`, `e_raw` kept), in-process only: no frozen file is edited.
  dump  --run e363_<g>_s<S>      -> `jends_eval.dump` (unchanged function) on the wrapped model, writes runs/<run>/ends_val.pkl + ends_head.json (move an existing pair away first: this script refuses to overwrite)
  check --seed S --g a|c2 [--n 16]  -> the pair model's node outputs (`e`, `pos`) on the first n val_clean frames against `QGraph.node_stage` of B' (`runs/e292_<g>_s<S>`) on the same batch: max absolute differences; exit 1 when above 1e-5
usage (problem folder, GPU via run.py): bash tools/env/_pyr.sh faithful.py dump --run e363_a_s0     |     bash tools/env/_pyr.sh faithful.py check --seed 0 --g a
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace as NS

import torch
import torch.nn as nn

ROOT = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jends_eval as JE  # noqa: E402
import qmnms as QMN  # noqa: E402


class LayerNodes(nn.Module):
    """the node head followed by the Matrix-NMS decay of B' (as `QGraph.node_stage`): `e` is the damped logit, `e_raw` the head's own"""
    def __init__(self, nodes, radius, sigma):
        super().__init__()
        self.nodes, self.radius, self.sigma = nodes, float(radius), float(sigma)

    def forward(self, *a):
        no = dict(self.nodes(*a))
        no["e_raw"] = no["e"]
        no["e"] = QMN.matrix_nms_logit(no["e"], no["pos"], self.radius, self.sigma)
        return no


def layer_params(g, seed):
    args = torch.load(f"{ROOT}/runs/e292_{g}_s{seed}/model.pt", map_location="cpu")["args"]
    sn = args.get("soft_nms")
    if g == "a":
        assert sn and sn[0] == "mnms" and len(sn) == 3, f"runs/e292_a_s{seed}: soft_nms {sn}"
        return float(sn[1]), float(sn[2])
    assert not sn, f"runs/e292_{g}_s{seed} has a layer {sn}"
    return None


def model_for(run, dev):
    """the pair model of the run folder `e363_<g>_s<S>` with the layer of B' in front of Attach (arm a only)"""
    g = {"c2": "c2", "a": "a", "b": None}[run.split("_")[1]]
    seed = int(run.rsplit("_s", 1)[1])
    m, args = JE._orig_load_model(run, dev)
    if g is not None and layer_params(g, seed) is not None:
        r, sg = layer_params(g, seed)
        m.nodes = LayerNodes(m.nodes, r, sg).to(dev).eval()
    return m, args


JE._orig_load_model = JE.load_model
JE.load_model = model_for                                                                                  # `jends_eval.dump` calls `load_model(a.run, dev)` of its own module: in-process patch


def cmd_dump(a):
    p = f"{ROOT}/runs/{a.run}/ends_val.pkl"
    assert not os.path.exists(p), f"{p} exists: refused (move it away by hand first)"
    JE.dump(NS(run=a.run, thr=0.5, max=0))


def cmd_check(a):
    import lane_gt_data as LD
    import qpredict as QP
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run = f"e363_{a.g}_s{a.seed}"
    m, _ = model_for(run, dev)
    B = QP.load_model(f"{ROOT}/runs/e292_{a.g}_s{a.seed}", dev)
    LD.QData.DEFAULT_JUNCTION = True
    ds, dl = LD.loader("val_clean", bs=8, workers=0, shuffle=False, map_mode="error", pixels=False)
    worst = dict(e=0.0, pos=0.0, e_raw=0.0)
    n = 0
    with torch.no_grad():
        for b in dl:
            x = b["x"].to(dev).float()
            P, _ = m.jmodel(x, joint=False)
            nd = m.nodes(P["feat2"], P["feat1"], P["feat05"], P["road2"], P["centre1"])
            nb = B.node_stage(B.pyr(x))
            worst["e"] = max(worst["e"], float((nd["e"].float() - nb["e"].float()).abs().max()))
            worst["pos"] = max(worst["pos"], float((nd["pos"].float() - nb["pos"].float()).abs().max()))
            if "e_raw" in nb:
                worst["e_raw"] = max(worst["e_raw"], float((nd["e_raw"].float() - nb["e_raw"].float()).abs().max()))
            n += x.shape[0]
            if n >= a.n:
                break
    ok = worst["e"] <= 1e-5 and worst["pos"] <= 1e-5 and worst["e_raw"] <= 1e-5
    layered = layer_params(a.g, a.seed) is not None
    print(f"NODE CHECK {run}: {n} frames, layer {'mnms %s' % (layer_params(a.g, a.seed),) if layered else 'none'}, max |difference| e {worst['e']:.2e}, pos {worst['pos']:.2e}, e_raw {worst['e_raw']:.2e}: {'PASS' if ok else 'FAIL'}")
    json.dump(dict(run=run, frames=n, layer=layer_params(a.g, a.seed), max_abs=worst, passed=bool(ok)), open(f"{ROOT}/temp/e363/nodecheck_{a.g}_s{a.seed}.json", "w"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--run", required=True)
    c = sub.add_parser("check")
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--g", choices=("a", "c2"), required=True)
    c.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    {"dump": cmd_dump, "check": cmd_check}[a.cmd](a)
