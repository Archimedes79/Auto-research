"""Does the carried world memory help? Metric-level paired test of a clip-trained run (TrainConfig.seq_len > 1).

    val clips of seq_len consecutive keyframes of one scene (training.train.ClipDataset), spread over all val scenes
      -> the model runs through the clip with its state carried from keyframe to keyframe (the ego step between them, the state
         detached as in training): the LAST keyframe is "warm"
      -> the same last keyframe is run again from an empty memory: "cold" (what evaluation/evaluate_run.py scores for all samples)
      -> the same SegMeter / DetMeter numbers (evaluation/metrics.py via evaluate_run helpers), per dt = -0.5 / 0 / +0.5 s, for
         warm and cold on the SAME frames, and their difference.
Output: runs/<name>/eval/memory.json (+ table in memory.txt). EMA weights, VAL.
Usage: wsl ~/miniconda3/envs/gcar/bin/python evaluation/eval_memory.py runs/<name> [--clips 600] [--workers 4]
"""
import argparse
import functools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from evaluation import evaluate_run as ev                        # noqa: E402  (puts the run's code snapshot on sys.path)
from evaluation.metrics import DetMeter, SegMeter, report         # noqa: E402
from feature_cache import collate_cached                          # noqa: E402
from losses import NUSC_GROUPS                                    # noqa: E402
from object_head import decode                                    # noqa: E402
from train import (ClipDataset, collate_clips, clip_step, detach_tree, forward_loss, make_datasets,   # noqa: E402
                   val_subset)

KEYS = (("seg_fov", "drivable_area", "iou@0"), ("seg_fov", "walkway", "iou@0"), ("seg_fov", "lane_divider", "iou@1"),
        ("seg_obj", "vehicle", "iou@0"), ("seg_obj", "pedestrian", "iou@0"))


def run(runp, n_clips=600, workers=4, device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    os.chdir(ev.ROOT)
    cfg, model, loss, step, ckpt = ev.load_run(runp, device)
    L = cfg.seq_len
    if L < 2:
        raise SystemExit(f"{runp}: seq_len = {L}, nothing carried")
    if not cfg.radar_cache and os.path.isdir("data/cache/radar"):
        cfg.radar_cache = "data/cache/radar"
    cfg.workers = workers
    _, val_ds = make_datasets(cfg)
    clips = ClipDataset(val_ds, L)
    ds = val_subset(clips, n_clips)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=workers,
                                         collate_fn=functools.partial(collate_clips, fn=collate_cached))
    spec, grid = model.spec, model.grid
    use_ref = bool(cfg.model.get("refine")) and hasattr(model, "boxes")
    M = {w: {"seg": {"fov": SegMeter(grid), "seen": SegMeter(grid), "obj": SegMeter(grid)},
             "det": DetMeter(classes=[g for g, _ in NUSC_GROUPS]),
             "detr": DetMeter(classes=[g for g, _ in NUSC_GROUPS]) if use_ref else None} for w in ("warm", "cold")}
    t0, n = time.time(), 0
    with torch.no_grad():
        for clip in loader:
            state, prev = None, None
            for t, batch in enumerate(clip):
                st = clip_step(prev, device) if prev is not None else None
                _, res, gts = forward_loss(model, loss, batch, 1.0, device, cfg.amp, cfg, train_terms=False, state=state,
                                           step=st)
                state, prev = detach_tree(res["state"]), batch
            _, cold, _ = forward_loss(model, loss, clip[-1], 1.0, device, cfg.amp, cfg, train_terms=False)
            for name, r in (("warm", res), ("cold", cold)):
                refined = {dt: model.boxes(r, dt, k=100, threshold=0.05, rescore=True) for dt in r["out"]} if use_ref else {}
                for dt, p in r["out"].items():
                    for b in range(len(clip[-1]["tokens"])):
                        g = gts[dt][b]
                        mass = p["mass"][b, 0].float().cpu().numpy() if (p.get("mass") is not None and abs(dt) > 1e-6) else None
                        prob, dist = ev.predictions(p, b, spec)
                        ev.add_sample(M[name]["seg"], prob, dist, g, spec, dt, mass)
                        dets = decode({k: v[b:b + 1].float().cpu() for k, v in p["obj"].items()}, grid, NUSC_GROUPS,
                                      k=100, threshold=0.05)[0]
                        M[name]["det"].add(dets, ev.gt_boxes(g), dt=dt)
                        if use_ref:
                            M[name]["detr"].add(refined[dt][b], ev.gt_boxes(g), dt=dt)
            n += len(clip[-1]["tokens"])
            if n % 100 == 0:
                print(f"{n}/{len(ds)} clips, {time.time() - t0:.0f} s", flush=True)
    flat = {}
    for w in ("warm", "cold"):
        for sname, m in M[w]["seg"].items():
            flat.update({f"{w}/seg_{sname}/" + k: v for k, v in report(seg=m).items()})
        flat.update({f"{w}/det/" + k[len("det/"):]: v for k, v in report(det=M[w]["det"]).items()})
        if use_ref:
            flat.update({f"{w}/detr/" + k[len("det/"):]: v for k, v in report(det=M[w]["detr"]).items()})
    lines = [f"{runp} step {step}: warm vs cold memory on the last keyframe of {n} val clips of {L} keyframes (EMA, VAL)"]
    for dt in ("dt+0.00", "dt+0.50", "dt-0.50"):
        rows = [(f"drivable IoU@0", f"seg_fov/seg/{dt}/drivable_area/all/all/iou@0"),
                ("walkway IoU@0", f"seg_fov/seg/{dt}/walkway/all/all/iou@0"),
                ("lane-divider mask IoU@1", f"seg_fov/seg/{dt}/lane_divider/all/all/iou@1"),
                ("vehicle coverage IoU", f"seg_obj/seg/{dt}/vehicle/all/all/iou@0"),
                ("vehicle AP@2", f"det/{dt}/vehicle/all/ap@2"), ("pedestrian AP@2", f"det/{dt}/pedestrian/all/ap@2"),
                ("vehicle ATE", f"det/{dt}/vehicle/all/ate"), ("vehicle AVE", f"det/{dt}/vehicle/all/ave")]
        if use_ref:
            rows += [("vehicle AP@2 refined", f"detr/{dt}/vehicle/all/ap@2"), ("vehicle ATE refined", f"detr/{dt}/vehicle/all/ate")]
        lines.append(dt)
        for label, key in rows:
            kw = f"warm/{key}"
            kc = f"cold/{key}"
            if kw in flat and kc in flat:
                lines.append(f"  {label:26s} warm {flat[kw]:8.4f}  cold {flat[kc]:8.4f}  warm - cold {flat[kw] - flat[kc]:+8.4f}")
    txt = "\n".join(lines)
    out = os.path.join(runp, "eval")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "memory.json"), "w") as f:
        json.dump({"run": runp, "checkpoint": os.path.basename(ckpt), "n_clips": n, "seq_len": L, "metrics": flat}, f,
                  indent=1, default=float)
    with open(os.path.join(out, "memory.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--clips", type=int, default=600)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run(a.run, a.clips, a.workers)
