"""ms per frame of the lane head (E273) per stage: batch 1, cached features, bf16 autocast as in training, CUDA events after warm-up; run it on an IDLE GPU (nothing else training).

    wsl bash tools/env/_py14.sh qgraph_timing.py --run e273_p_s0 [--n 60]

Stages: pyr (pixel maps), nodes (node head), links (candidates, set attention, Hermite evidence, link MLP), sections (the linear solve), assign (map encoder + assignment, 2 registration rounds), total.
The map and the section solve are part of the head; the read-out (CPU, numpy) is not timed here. Writes `runs/e273/<run>/timing.json` and prints the medians.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_gt_data as LD  # noqa: E402
import qpredict as QP  # noqa: E402
import sections as SE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    dev = "cuda"
    m = QP.load_model(os.path.join(ROOT, "runs", a.run), dev)
    ds = LD.QData("val_clean", map_mode="error", pixels=False)
    batches = [LD.collate([ds[i]]) for i in np.linspace(0, len(ds) - 1, 12).astype(int)]
    names = ("pyr", "nodes", "links", "sections", "assign")
    t = {k: [] for k in names + ("total",)}
    ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
    with torch.no_grad():
        for it in range(a.n + 10):
            b = batches[it % len(batches)]
            marks = [ev() for _ in range(7)]
            x = b["x"].to(dev).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                marks[0].record()
                P = m.pyr(x)
                marks[1].record()
                no = m.node_stage(P)                                                              # node head + the soft close-node step of E277 when the model has it
                marks[2].record()
                L = m.links(no, P["centre1"][:, :1].sigmoid(), gt=None)
                marks[3].record()
            M, _ = SE.section_membership({k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in L.items()}, no["s"].float())
            marks[4].record()
            mp = dict(map_pos=b["map_pos"].to(dev), map_tan=b["map_tan"].to(dev), map_frac=b["map_frac"].to(dev), map_pidx=b["map_pidx"].to(dev), map_pfeat=b["map_pfeat"].to(dev),
                      map_adj=b["map_adj"].to(dev), kmask=b["kmask"].to(dev), tmask=b["tmask"].to(dev), map_road=b["map_road"].to(dev))
            enc = m.menc(mp["map_pos"], mp["map_tan"], mp["map_frac"], mp["map_pidx"], mp["map_pfeat"], mp["map_adj"], mp["kmask"], mp["tmask"])
            e_act = no["e"].float().flatten(1).gather(1, L["act"]).sigmoid()
            m.asg(L["tok"].float(), L["pos"].float(), L["dirv"].float(), e_act, L["valid"], mp, enc, rounds=2)
            marks[5].record()
            torch.cuda.synchronize()
            if it >= 10:
                for k, (i, j) in zip(names, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))):
                    t[k].append(marks[i].elapsed_time(marks[j]))
                t["total"].append(marks[0].elapsed_time(marks[5]))
    out = {k: dict(median=float(np.median(v)), p90=float(np.percentile(v, 90))) for k, v in t.items()}
    out["gpu_other_processes"] = "see the run record: measure only on an idle GPU"
    print({k: round(v["median"], 2) if isinstance(v, dict) else v for k, v in out.items()})
    d = os.path.join(ROOT, "runs", "e273", a.run)                                       # the E273 scoring folder when it exists, else the run's own folder (an E276 arm has none in e273: it used to fail here after the timing loop)
    if not os.path.isdir(d):
        d = os.path.join(ROOT, "runs", a.run)
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
