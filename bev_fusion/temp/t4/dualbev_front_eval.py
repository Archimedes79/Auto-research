"""t4_soa_front: the pre-trained DualBEV (front camera only) on the same 1000 cold-VAL frames as the study arms.
Writes runs/t4_dualbev_front/eval/scene_records.pkl in the format of evaluate_run (det[scene][dt] = [(dets, gt boxes)]) and
metrics.json (front field of view up to 60 m, as every other arm). Resumable (partial records every 100 frames).
Usage (bev_fusion folder): python temp/t4/dualbev_front_eval.py [--device cpu|cuda] [--n 1000] [--threads 2]
"""
import argparse
import json
import os
import pickle
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "training")]
from evaluation import evaluate_run as ev  # noqa: E402
from evaluation.metrics import DetMeter  # noqa: E402
from dualbev_front import DualBEVFront, decode  # noqa: E402
from losses import NUSC_GROUPS  # noqa: E402
from train import TrainConfig, make_datasets, val_subset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--out", default="runs/t4_dualbev_front")
ap.add_argument("--bgr", action="store_true", help="feed the channels swapped (the original pipeline may have swapped RGB to BGR)")
a = ap.parse_args()
torch.set_num_threads(a.threads)
out_dir = os.path.join(ROOT, a.out, "eval")
os.makedirs(out_dir, exist_ok=True)
part = os.path.join(out_dir, "partial.pkl")

cfg = TrainConfig(**json.load(open(os.path.join(ROOT, "runs/t2_ref_s0/config.json"))))
cfg.workers, cfg.feature_cache, cfg.n_cam = 0, "", 1
_, val_ds = make_datasets(cfg)
val_ds._radar = lambda *args, **kw: {}                          # the camera-only reference needs no radar
sub = val_subset(val_ds, a.n)
dev = torch.device(a.device)
model = DualBEVFront().eval().to(dev)
print(model.load_pretrained(os.path.join(ROOT, "models/pretrained/dualbev-r50-cbgs.pth")), flush=True)

records = pickle.load(open(part, "rb")) if os.path.exists(part) else {}       # token -> (scene, dets, gt boxes)
t0, ms = time.time(), []
for i in range(len(sub)):
    tok = sub.dataset.samples[sub.indices[i]] if hasattr(sub, "indices") else sub.samples[i]
    if tok in records:
        continue
    item = sub[i]
    f = item["frames"][0]
    with torch.no_grad():
        t1 = time.time()
        img = f["img"].flip(0) if a.bgr else f["img"]
        outs = model(img[None].to(dev), f["cam_vec"][None].to(dev), f["K"][None].to(dev),
                     f["T_cam_car"][None].to(dev), item["ida"].to(dev))
        outs = [{k: v.float().cpu() for k, v in o.items()} for o in outs]
        ms.append((time.time() - t1) * 1e3)
    dets = decode(outs, k=100, threshold=0.05)[0]
    scene = val_ds.nusc.get("scene", val_ds.nusc.get("sample", tok)["scene_token"])["name"]
    records[tok] = (scene, dets, ev.gt_boxes(item["gt"]["cur"]))
    if len(records) % 100 == 0:
        pickle.dump(records, open(part, "wb"))
        print(f"{len(records)}/{len(sub)} frames, {time.time() - t0:.0f} s, {sum(ms) / len(ms):.0f} ms/frame (model)", flush=True)

det, scenes = DetMeter(classes=[g for g, _ in NUSC_GROUPS]), {}
for tok in (sub.dataset.samples[j] for j in sub.indices) if hasattr(sub, "indices") else sub.samples:
    scene, dets, gb = records[tok]
    det.add(dets, gb, dt=0.0)
    scenes.setdefault(scene, {}).setdefault(0.0, []).append((dets, gb))
flat = ev.report(seg=None, det=det)
flat["info/n_samples"], flat["info/n_scenes"] = len(records), len(scenes)
json.dump({"run": a.out, "checkpoint": "dualbev-r50-cbgs.pth (EMA)", "split": cfg.val_split, "weights": "EMA",
           "memory": "none (single frame)", "device": a.device, "metrics": flat,
           "latency": {"model_ms_median": sorted(ms)[len(ms) // 2] if ms else None}},
          open(os.path.join(out_dir, "metrics.json"), "w"), indent=1, default=float)
pickle.dump({"seg": {}, "det": {s: r for s, r in scenes.items()}, "detr": {}}, open(os.path.join(out_dir, "scene_records.pkl"), "wb"))
if os.path.exists(part):
    os.remove(part)
mAP2 = sum(flat[f"det/dt+0.00/{c}/all/ap@{t}"] for c in ("vehicle", "pedestrian") for t in (0.5, 1, 2, 4)) / 8
print(f"done: {len(records)} frames; mAP2 {mAP2:.3f} (front FOV, <= 60 m)")
