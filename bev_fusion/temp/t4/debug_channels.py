"""One-off (t4_soa_front): does the pre-trained DualBEV expect RGB or BGR? Top heatmap scores of the same frames, both orders."""
import json
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, "training", "legacy")]
from dualbev_front import DualBEVFront, decode  # noqa: E402
from train import TrainConfig, make_datasets, val_subset  # noqa: E402

torch.set_num_threads(2)
cfg = TrainConfig(**json.load(open(os.path.join(ROOT, "runs/t2_ref_s0/config.json"))))
cfg.workers, cfg.feature_cache, cfg.n_cam = 0, "", 1
_, val_ds = make_datasets(cfg)
val_ds._radar = lambda *a, **k: {}
sub = val_subset(val_ds, 6)
m = DualBEVFront().eval()
m.load_pretrained(os.path.join(ROOT, "models/pretrained/dualbev-r50-cbgs.pth"))
for i in (0, 1, 3):
    item = sub[i]
    f = item["frames"][0]
    for name, img in (("RGB", f["img"]), ("BGR", f["img"].flip(0))):
        with torch.no_grad():
            outs = m(img[None], f["cam_vec"][None], f["K"][None], f["T_cam_car"][None], item["ida"])
        hm = [torch.sigmoid(o["heatmap"]).amax().item() for o in outs]
        d = decode(outs, k=5, threshold=0.0)[0]
        print(i, name, "max score per task", [round(x, 2) for x in hm], "top3", [(b["cls"], round(b["cx"], 1), round(b["cy"], 1), round(b["score"], 2)) for b in d[:3]], flush=True)
