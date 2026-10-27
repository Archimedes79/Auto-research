"""One-off audit (2026-10-26): are the BatchNorm statistics of the EMA weights consistent with them? The EMA of the weights and
of the running mean / variance mixes statistics of many different weight states. Test: val metrics (cold, dt 0) with the stored
statistics against the same EMA weights with the BatchNorm statistics recomputed from 400 train samples (train mode, no gradient,
cumulative average).
Usage (bev_fusion folder): wsl ~/miniconda3/envs/gcar/bin/python temp/audit/diag_bn.py runs/t1_r3_combined --n 300
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.argv, _argv = [sys.argv[0], next(a for a in sys.argv[1:] if not a.startswith("-"))], sys.argv
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--recal", type=int, default=400)
a = ap.parse_args(_argv[1:])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evaluation"))
sys.argv = [sys.argv[0], a.run]
import evaluate_run as E  # noqa: E402
from evaluation.metrics import DetMeter, SegMeter, report  # noqa: E402


def evaluate(model, loss, loader, cfg, spec, grid, device):
    meters = {"fov": SegMeter(grid), "obj": SegMeter(grid)}
    det = DetMeter(classes=[g for g, _ in E.NUSC_GROUPS])
    tot, nb = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            out, res, gts = E.forward_loss(model, loss, batch, 1.0, device, cfg.amp, cfg, train_terms=False)
            tot += float(out["total"])
            nb += 1
            p = res["out"][0.0]
            for b in range(len(batch["tokens"])):
                g = gts[0.0][b]
                prob, dist = E.predictions(p, b, spec)
                E.add_sample(meters, prob, dist, g, spec, 0.0, None)
                dets = E.decode({k: v[b:b + 1].float().cpu() for k, v in p["obj"].items()}, grid, E.NUSC_GROUPS, k=100,
                                threshold=0.05)[0]
                det.add(dets, E.gt_boxes(g), dt=0.0)
    f = report(seg=meters["fov"], det=det)
    f.update({"obj/" + k: v for k, v in report(seg=meters["obj"]).items()})
    row = {"val_total": tot / max(nb, 1)}
    for name, key in (("drivable", "seg/dt+0.00/drivable_area/all/all/iou@0"), ("walkway", "seg/dt+0.00/walkway/all/all/iou@0"),
                      ("lane_mask@0.5", "seg/dt+0.00/lane_divider/all/all/iou@0.5"),
                      ("veh_AP2", "det/dt+0.00/vehicle/all/ap@2"), ("ped_AP2", "det/dt+0.00/pedestrian/all/ap@2"),
                      ("mAP", "det/dt+0.00/mean/all/mAP")):
        row[name] = f.get(key)
    return row


def main():
    device = "cuda"
    os.chdir(E.ROOT)
    cfg, model, loss, step, ckpt = E.load_run(a.run, device)
    cfg.batch_size, cfg.workers, cfg.seq_len = 4, 3, 1
    if not cfg.radar_cache and os.path.isdir("data/cache/radar"):
        cfg.radar_cache = "data/cache/radar"
    train_ds, val_ds = E.make_datasets(cfg)
    spec, grid = model.spec, model.grid
    vl = E.make_loader(E.val_subset(val_ds, a.n), cfg, False)
    res = {"stored": evaluate(model, loss, vl, cfg, spec, grid, device)}
    print("stored  ", res["stored"], flush=True)
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                                         # cumulative average over the recalibration batches
    model.train()
    tl = E.make_loader(E.val_subset(train_ds, a.recal), cfg, True)
    with torch.no_grad():
        for batch in tl:
            E.forward_loss(model, loss, batch, 1.0, device, cfg.amp, cfg, train_terms=False)
    model.eval()
    res["recalibrated"] = evaluate(model, loss, vl, cfg, spec, grid, device)
    print("recalib.", res["recalibrated"], f"({len(bns)} BatchNorm layers)", flush=True)
    out = os.path.join("temp", "audit", os.path.basename(a.run.rstrip("/")) + "_bn.json")
    with open(out, "w") as f:
        json.dump({"run": a.run, "step": step, "res": res}, f, indent=1)


main()
