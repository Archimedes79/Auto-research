"""One-off: why is the zero-crossing line F1 near 0? On val samples of a run's last checkpoint: the predicted distance field
on the GT line cells (GT distance <= 0.25 m) and elsewhere, the share of GT line cells where the prediction is <= 0.25 m
(the decoder's threshold), and the line-mask probability on GT line cells vs other camera cells."""
import sys

import numpy as np
import torch

sys.path.insert(0, "training")
sys.path.insert(0, ".")
from evaluation.bev_probe import load_run, predictions  # noqa: E402
from train import forward_loss, make_datasets, make_loader, val_subset  # noqa: E402

cfg, model, loss, step, _ = load_run(sys.argv[1], "cuda")
cfg.workers = 2
_, val_ds = make_datasets(cfg)
acc = {}
with torch.no_grad():
    for batch in make_loader(val_subset(val_ds, 64), cfg, False):
        _, res, gts = forward_loss(model, loss, batch, 1.0, "cuda", cfg.amp, cfg, train_terms=False)
        for b in range(len(batch["tokens"])):
            prob, dist = predictions(res["out"][0.0], b, model.spec)
            g = gts[0.0][b]
            cam = np.asarray(g["camera"], np.float32) > 0
            for ln in ("lane_divider", "road_divider"):
                gd = np.asarray(g[ln + "_dist"], np.float32)
                on, off = cam & (gd <= 0.25), cam & (gd > 1.0)
                a = acc.setdefault(ln, {"d_on": [], "d_off": [], "p_on": [], "p_off": []})
                a["d_on"].append(dist[ln][on])
                a["d_off"].append(dist[ln][off])
                a["p_on"].append(prob[ln][on])
                a["p_off"].append(prob[ln][off])
print(f"step {step}")
for ln, a in acc.items():
    d_on, d_off = np.concatenate(a["d_on"]), np.concatenate(a["d_off"])
    p_on, p_off = np.concatenate(a["p_on"]), np.concatenate(a["p_off"])
    print(f"{ln}: {len(d_on)} GT line cells; predicted distance on them median {np.median(d_on):.2f} m (p10 {np.percentile(d_on, 10):.2f}),"
          f" share <= 0.25 m {np.mean(d_on <= 0.25):.3f}; off-line cells median {np.median(d_off):.2f} m. Mask prob on line "
          f"cells mean {p_on.mean():.3f} (share > 0.5 {np.mean(p_on > 0.5):.3f}), off-line mean {p_off.mean():.3f}")
