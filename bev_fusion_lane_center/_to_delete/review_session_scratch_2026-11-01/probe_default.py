"""throw-away: a default-off config trains bit for bit as before the t11 change. argv: <training dir> <out .pt>.
3 optimiser steps on the 4 cached mini samples (the train.py self-test's setup), t10-like model (object_direction, refine off)."""
import os
import sys

TRAIN_DIR, OUT = sys.argv[1], sys.argv[2]
ROOT = "/mnt/c/Auto-research/bev_fusion"
sys.path[:0] = [TRAIN_DIR, ROOT]
os.chdir(ROOT)
import json  # noqa: E402

import torch  # noqa: E402
torch.set_num_threads(2)
import train as tr  # noqa: E402
assert os.path.dirname(os.path.abspath(tr.__file__)) == os.path.abspath(TRAIN_DIR), tr.__file__
from feature_cache import CachedFrontRadarDataset, FeatureStore, used_frames  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402

S = os.path.dirname(os.path.abspath(OUT))
GT = "/mnt/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/6e9638e0-a42f-4660-b7d5-cac910d9084d/scratchpad/gt_mini"
cache = "data/cache/cam_feats_mini"
store = FeatureStore(cache)
cfg = tr.TrainConfig(name="probe", out_root=os.path.join(S, "runs_" + os.path.basename(OUT)[:-3]), data_root="~/data/nuscenes",
                     version="v1.0-mini", feature_cache=cache, gt_cache=GT, batch_size=2, accum=1, workers=0,
                     steps=3, warmup=2, log_every=1, eval_every=3, eval_batches=1, save_every=0, flip_aug=0.5, device="cpu",
                     model={"trunk_width": 0.25, "object_direction": True}, loss={"w_obj_vel": 1.0})
nusc = NuScenes(cfg.version, dataroot=os.path.expanduser(cfg.data_root), verbose=False)
base = CachedFrontRadarDataset(os.path.expanduser(cfg.data_root), cfg.version, store=store, n_cam=3, cam_stride=2, with_gt=True,
                               nusc=nusc, cache_dir=cfg.gt_cache)
ok = [t for t in base.samples if all(store.has(x) for v in used_frames(base, [t]).values() for x in v)]
base.samples = ok[:4]
model, loss, ema, last = tr.train(cfg, datasets=(base, base))
rows = [json.loads(x) for x in open(os.path.join(cfg.out_root, "probe", "log.jsonl"))]
torch.save({"model": model.state_dict(), "loss": loss.state_dict(), "ema": ema.shadow, "rows": rows}, OUT)
print("saved", OUT, len(rows), "log rows")
