"""One-off audit (2026-10-26): what does a trained run really use and where are its errors?
For val and train samples (cold, dt 0, EMA weights): IoU of drivable / walkway / lane-divider mask / vehicle coverage in the camera
cells split by distance band (0-20, 20-40, 40-60 m), vehicle AP, and the same with the radar branch zeroed or the camera branch
zeroed in front of the fusion (how much each sensor is used).
Usage (bev_fusion folder): wsl ~/miniconda3/envs/gcar/bin/python temp/audit/diag_modalities.py runs/t1_r3_combined --n 300
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.argv, _argv = [sys.argv[0], next(a for a in sys.argv[1:] if not a.startswith("-"))], sys.argv
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--splits", default="val,train")
ap.add_argument("--variants", default="base,no_radar,no_camera")
a = ap.parse_args(_argv[1:])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evaluation"))
sys.argv = [sys.argv[0], a.run]
import evaluate_run as E  # noqa: E402
from evaluation.metrics import DetMeter, report  # noqa: E402

LAYERS = [("drivable_area", "fov"), ("walkway", "fov"), ("lane_divider", "fov"), ("vehicle", "obj_fov")]
BANDS = [(0, 40), (40, 80), (80, 120)]                          # rows of the grid: 0-20 m, 20-40 m, 40-60 m ahead
MODE = {"v": "base"}


def fusion_hook(mod, args):
    cam, rad = args
    if MODE["v"] == "no_radar":
        return cam, torch.zeros_like(rad)
    if MODE["v"] == "no_camera":
        return torch.zeros_like(cam), rad
    return None


def run():
    device = "cuda"
    os.chdir(E.ROOT)
    cfg, model, loss, step, ckpt = E.load_run(a.run, device)
    cfg.batch_size, cfg.workers, cfg.seq_len = 4, 3, 1
    if not cfg.radar_cache and os.path.isdir("data/cache/radar"):
        cfg.radar_cache = "data/cache/radar"
    model.fusion.register_forward_pre_hook(fusion_hook)
    train_ds, val_ds = E.make_datasets(cfg)
    spec = model.spec
    k = E.layer_index(spec)
    res = {}
    for split in a.splits.split(","):
        ds = E.val_subset(val_ds if split == "val" else train_ds, a.n)
        loader = E.make_loader(ds, cfg, False)
        variants = a.variants.split(",") if split == "val" else ["base"]
        cnt = {v: {l: np.zeros((len(BANDS) + 1, 3)) for l, _ in LAYERS} for v in variants}
        det = {v: DetMeter(classes=[g for g, _ in E.NUSC_GROUPS]) for v in variants}
        with torch.no_grad():
            for batch in loader:
                for v in variants:
                    MODE["v"] = v
                    out, r, gts = E.forward_loss(model, loss, batch, 1.0, device, cfg.amp, cfg, train_terms=False)
                    p = r["out"][0.0]
                    for b in range(len(batch["tokens"])):
                        g = gts[0.0][b]
                        m = E.masks(g)
                        prob, _ = E.predictions(p, b, spec)
                        for layer, mk in LAYERS:
                            gt = np.asarray(g[k[layer][2]], np.float32) > 0.5
                            pr = prob[layer] > 0.5
                            mm = m[mk]
                            for bi, (r0, r1) in enumerate(BANDS):
                                s = np.zeros_like(mm)
                                s[r0:r1] = True
                                s &= mm
                                cnt[v][layer][bi] += [(pr & gt & s).sum(), (pr & ~gt & s).sum(), (~pr & gt & s).sum()]
                            cnt[v][layer][-1] += [(pr & gt & mm).sum(), (pr & ~gt & mm).sum(), (~pr & gt & mm).sum()]
                        dets = E.decode({kk: vv[b:b + 1].float().cpu() for kk, vv in p["obj"].items()}, model.grid,
                                        E.NUSC_GROUPS, k=100, threshold=0.05)[0]
                        det[v].add(dets, E.gt_boxes(g), dt=0.0)
        MODE["v"] = "base"
        for v in variants:
            flat = report(seg=None, det=det[v])
            row = {}
            for layer, _ in LAYERS:
                c = cnt[v][layer]
                row[layer] = [float(x[0] / max(x.sum(), 1)) for x in c]            # IoU = tp / (tp + fp + fn) per band + all
            row["veh_AP2"] = flat.get("det/dt+0.00/vehicle/all/ap@2")
            row["ped_AP2"] = flat.get("det/dt+0.00/pedestrian/all/ap@2")
            row["mAP"] = flat.get("det/dt+0.00/mean/all/mAP")
            res[f"{split}/{v}"] = row
            print(f"{split:5s} {v:10s} " + "  ".join(f"{l[:8]} {row[l][-1]:.3f} ({row[l][0]:.2f}/{row[l][1]:.2f}/{row[l][2]:.2f})"
                                                     for l, _ in LAYERS) + f"  vehAP2 {row['veh_AP2']:.3f} pedAP2 {row['ped_AP2']:.3f} mAP {row['mAP']:.3f}",
                  flush=True)
    out = os.path.join("temp", "audit", os.path.basename(a.run.rstrip("/")) + "_modalities.json")
    with open(out, "w") as f:
        json.dump({"run": a.run, "step": step, "n": a.n, "bands_m": [(r0 / 2, r1 / 2) for r0, r1 in BANDS], "res": res}, f, indent=1)
    print("written", out)


run()
