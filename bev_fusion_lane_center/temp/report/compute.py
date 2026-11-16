"""Report: the compute table (CPU; Mirko 2026-11-15: 'little compute should be part of the goal'): per system parameters, GMACs per frame (torch FlopCounterMode on one real VAL frame, batch 1, convolutions and matrix products, as `evaluation/jmodel_cost.py`),
ms per frame (from the idle-GPU timing `runs/eval/jtiming/e342b_20261115.json`: batch 1, bf16, a LOADED host: the paired differences are the sound part) and the snapped topo F1 (VAL, both seeds, from the stored cards).
usage: wsl bash tools/env/_py14.sh compute.py   -> temp/report/compute.json"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.getcwd()
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jends_eval as JE  # noqa: E402
import lane_gt_data as LD  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

torch.set_num_threads(2)
RUNS = {"E312": "e312_p_s0", "E341 (E312 recipe, no 0.5 m branch)": "e341_p_s0", "E324 s (system of record)": "e324_s_s0", "E348 (E324 s recipe, no 0.5 m branch)": "e348_w_s0"}
tim = json.load(open(os.path.join(ROOT, "runs", "eval", "jtiming", "e342b_20261115.json")))


def topo(run, card):
    r = []
    for s in (0, 1):
        r.append(json.load(open(os.path.join(ROOT, "runs", run.replace("_s0", f"_s{s}"), card)))["numbers"]["head"]["topo_F1"])
    return r


LD.QData.DEFAULT_JUNCTION = True
ds = LD.QData("val_clean", map_mode="error", pixels=False)
x = LD.collate([ds[len(ds) // 2]])["x"].float()
out = {}
for label, run in RUNS.items():
    m, _ = JE.load_model(run, "cpu")
    pr = {"total": sum(p.numel() for p in m.parameters()) / 1e6, "pyramid": sum(p.numel() for p in m.jmodel.pyr.parameters()) / 1e6, "maps_head": sum(p.numel() for p in m.jmodel.jm.parameters()) / 1e6,
          "pair_head": sum(p.numel() for p in m.je.parameters()) / 1e6, "node_head": (sum(p.numel() for p in m.nodes.parameters()) / 1e6 if m.nodes is not None else 0.0)}
    fl = {}
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            m(x, gt=None)
        fl["total_gmacs"] = fc.get_total_flops() / 2e9
        for nm, mod in (("pyramid", m.jmodel.pyr), ("maps_head", m.jmodel.jm), ("pair_head", m.je)):
            P = None
    # per stage with hooks is simplest through separate counted calls
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc1:
            P = m.jmodel.pyr(x)
        fl["pyramid_gmacs"] = fc1.get_total_flops() / 2e9
        with FlopCounterMode(display=False) as fc2:
            jo = m.jmodel.jm(P)
        fl["maps_head_gmacs"] = fc2.get_total_flops() / 2e9
        if m.nodes is not None:
            with FlopCounterMode(display=False) as fc3:
                m.nodes(P["feat2"], P["feat1"], P["feat05"], P["road2"], P["centre1"])
            fl["node_head_gmacs"] = fc3.get_total_flops() / 2e9
    ms = tim["plain_median_ms"][run]
    sn = topo(run, "ends_card_snap_thr0.1.json")
    out[label] = dict(run=run, params_m=pr, gmacs=fl, ms_per_frame=ms, ms_note="batch 1, bf16, idle GPU, LOADED host (load about 5.6 on 12 cores), median of 250 forwards, runs/eval/jtiming/e342b_20261115.json", snapped_topo_F1_s0_s1=sn)
    print(label, {k: round(v, 3) for k, v in pr.items()}, {k: round(v, 3) for k, v in fl.items()}, round(ms, 2), [round(v, 4) for v in sn])
b = [json.load(open(os.path.join(ROOT, "runs", f"e312_p_s{s}", "ends_card_snap_thr0.1.json")))["numbers"]["base"]["topo_F1"] for s in (0, 1)]
out["B alone (E273 + close-node step, delivered plain graph)"] = dict(params_m=dict(pyramid=2.097, node_head=0.099, links=0.117, assignment_and_map_encoder=0.123, total=2.436), gmacs=dict(pyramid_gmacs=4.677, heads="not counted (E330 section 1)"),
                                                                       ms_per_frame=24.24, ms_note="runs/e273/e273_p_s0/timing.json: B total with the pyramid 4.3 ms (nodes 2.9, links 5.5, sections 2.8, assignment 8.3 ms), batch 1, bf16", snapped_topo_F1_s0_s1=b)
d = {}
for a, b_ in (("E324 s (system of record)", "E312"), ("E341 (E312 recipe, no 0.5 m branch)", "E312"), ("E348 (E324 s recipe, no 0.5 m branch)", "E324 s (system of record)")):
    A, B = out[a], out[b_]
    dt = float(np.mean(A["snapped_topo_F1_s0_s1"]) - np.mean(B["snapped_topo_F1_s0_s1"]))
    dm = A["ms_per_frame"] - B["ms_per_frame"]
    d[f"{a} - {b_}"] = dict(delta_topo_F1=dt, delta_ms=dm, delta_gmacs=A["gmacs"]["total_gmacs"] - B["gmacs"]["total_gmacs"], delta_params_m=A["params_m"]["total"] - B["params_m"]["total"], topo_F1_per_ms=(dt / dm if dm else None))
out["price_lines"] = d
json.dump(out, open(os.path.join(ROOT, "temp", "report", "compute.json"), "w"), indent=1)
print(json.dumps(d, indent=1))
