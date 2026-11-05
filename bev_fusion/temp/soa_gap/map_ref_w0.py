"""SOA map reference on OUR metric (2026-10-30): the published camera + radar map model GaussianCaR (6 cameras, radar, 360 degrees,
trained for map segmentation on the official nuScenes train split; its logits are cached by the lane project for 1706 frames, all in
43 of OUR 82 official val scenes) scored with evaluation/metrics.SegMeter on the same frames, the same camera-cell mask, the same
tolerances as our model (t1_r4_soa, EMA weights, dt 0, cold) and a per-cell prior map (null).
Layers: drivable_area = max(sigmoid(lane), sigmoid(road_segment)); ped_crossing, walkway, carpark_area, lane_divider, road_divider
directly (GaussianCaR has no stop line). Mapping of the arrays: row = -y_left, column = forward + OFFSET (temp/soa_gap/align_gcar.py).
Thresholds: 0.5 (as our evaluation) and, to be fair to a model trained with another loss, the best of {0.3, 0.5, 0.7} per layer (in-sample).
Usage (bev_fusion folder, CPU, 2 threads): python temp/soa_gap/map_ref.py <run> <out.json> [limit] [offset]"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "training")
sys.path.insert(0, "evaluation")
sys.path.insert(0, ".")
os.environ.setdefault("OMP_NUM_THREADS", "2")
torch.set_num_threads(2)

from evaluation.bev_probe import add_sample, layer_index, load_run, masks, predictions  # noqa: E402
from metrics import SegMeter  # noqa: E402
from train import forward_loss, make_datasets, make_loader  # noqa: E402

HOME = os.path.expanduser("~")
run, out_path = sys.argv[1], sys.argv[2]
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 0
OFFSET = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
THR = (0.3, 0.5, 0.7)
LAYERS = ("drivable_area", "ped_crossing", "walkway", "carpark_area", "lane_divider", "road_divider")
RES = 0.5
XS = (np.arange(120) + 0.5) * RES
YS = -26.0 + (np.arange(104) + 0.5) * RES
X, Y = np.meshgrid(XS + OFFSET, YS, indexing="ij")
R = np.clip(np.floor((-Y + 50.0) / RES).astype(int), 0, 199)           # row of GaussianCaR = -y_left
C = np.clip(np.floor((X + 50.0) / RES).astype(int), 0, 199)            # column = forward
GC_IDX = {"lane": 0, "road_segment": 1, "road_divider": 2, "lane_divider": 3, "ped_crossing": 4, "walkway": 5, "carpark_area": 6}


def sig(a):
    return 1.0 / (1.0 + np.exp(-np.clip(a.astype(np.float32), -30, 30)))


def gcar_maps(L):
    """cached logits [7, 200, 200] -> our layers on our grid (probabilities)"""
    p = {k: sig(L[i])[R, C] for k, i in GC_IDX.items()}
    p["drivable_area"] = np.maximum(p["lane"], p["road_segment"])
    return {k: p[k] for k in LAYERS}


tok = json.load(open(f"{HOME}/exp_gcar_b123/feats/val_tokens.json"))
lg = np.load(f"{HOME}/exp_gcar_b123/feats/val_logit7.npy", mmap_mode="r")
device = "cpu"
cfg, model, loss, step, ckpt = load_run(run, device)
cfg.batch_size, cfg.workers, cfg.seq_len = 4, 0, 1
if not cfg.radar_cache and os.path.isdir("data/cache/radar"):
    cfg.radar_cache = "data/cache/radar"
train_ds, val_ds = make_datasets(cfg)
pos = {t: i for i, t in enumerate(val_ds.samples)}
rows = [(i, t) for i, t in enumerate(tok) if t in pos]
if LIMIT:
    rows = rows[:: max(1, len(rows) // LIMIT)][:LIMIT]
print(f"{len(rows)} frames (cached GaussianCaR logits inside our val set), model {run} step {step}", flush=True)
spec = model.spec
grid = model.grid
ds = torch.utils.data.Subset(val_ds, [pos[t] for _, t in rows])
loader = make_loader(ds, cfg, False)

# prior map (null): per-cell mean of the GT over 200 train samples spread evenly
idx = np.linspace(0, len(train_ds) - 1, 200).round().astype(int)
acc = {n: 0.0 for n in LAYERS}
for i in idx:
    g = train_ds._gt(train_ds.samples[i])["cur"]
    for n in LAYERS:
        acc[n] = acc[n] + np.asarray(g[n], np.float32)
prior = {n: acc[n] / len(idx) for n in LAYERS}

M = {}                                    # system -> threshold -> SegMeter(fov)
for sysname in ("ours", "gcar", "prior"):
    M[sysname] = {t: SegMeter(grid) for t in THR}
row_of = {t: i for i, t in rows}
t0, done = time.time(), 0
with torch.no_grad():
    for batch in loader:
        out, res, gts = forward_loss(model, loss, batch, 1.0, device, cfg.amp, cfg, train_terms=False)
        for b, t in enumerate(batch["tokens"]):
            g = gts[0.0][b]
            m = masks(g)["fov"]
            prob, _ = predictions(res["out"][0.0], b, spec)
            gc = gcar_maps(np.asarray(lg[row_of[t]]))
            gt = {n: np.asarray(g[n], np.float32) for n in LAYERS}
            for thr in THR:                              # only the PREDICTION threshold varies: p - (thr - .5) > .5 <=> p > thr; GT at 0.5
                sh = thr - 0.5
                M["ours"][thr].add({n: prob[n] - sh for n in LAYERS if n in prob}, {n: gt[n] for n in LAYERS if n in prob}, mask=m)
                M["gcar"][thr].add({n: p - sh for n, p in gc.items()}, gt, mask=m)
                M["prior"][thr].add({n: p - sh for n, p in prior.items()}, gt, mask=m)
            done += 1
        if done % 40 < 4:
            print(f"{done}/{len(rows)} frames, {time.time() - t0:.0f} s", flush=True)

summary = {"run": run, "step": step, "n_frames": done, "offset_m": OFFSET, "systems": {}}
for sysname in M:
    s = {}
    for thr in THR:
        r = M[sysname][thr].result()[0.0]
        for layer in r:
            for b in ("all", "0-20", "20-40", "40-60"):
                row = r[layer]["all"][b]
                s.setdefault(layer, {}).setdefault(b, {})[str(thr)] = {k: row[k] for k in ("iou@0", "iou@0.5", "iou@1") if k in row} | {"n_gt": row["n_gt"]}
    summary["systems"][sysname] = s
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
json.dump(summary, open(out_path, "w"), indent=1)
print("done", out_path, f"{time.time() - t0:.0f} s")
