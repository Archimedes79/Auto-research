"""End-to-end gate of the whole chain on real data (plan.md section 10.1, last gate before training runs).

One nuScenes mini keyframe (front camera now + 2 past frames, 5 radars, GT of k-1 / k / k+1) through
training/model.py with training/losses.py BEVLoss, overfitted for N steps (Adam, lr 5e-4, clipping). The chain
can learn if: the total loss falls by at least half, the drivable-area IoU (evaluation/metrics.py, camera cells, 0 m
tolerance) reaches 0.8 at dt 0 and 0.7 at dt +0.5 s, and the motion and object terms fall. A failure names the block.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_end_to_end.py [--steps 150] [--cpu]
GPU when one is free (>= 4 GB, at most 2-3 jobs; CLAUDE.md), else CPU (slow).
"""
import argparse
import os
import sys
import time

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, ROOT)
from evaluation.metrics import SegMeter                              # noqa: E402
from losses import BEVLoss, sample_targets                           # noqa: E402
from model import BEVFusionModel, ModelConfig                        # noqa: E402
from nuscenes_dataset import FrontRadarDataset, collate              # noqa: E402


def to_device(batch, dev):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(dev)
        elif k == "radar":
            out[k] = {kk: vv.to(dev) if torch.is_tensor(vv) else vv for kk, vv in v.items()}
        elif k == "steps":
            out[k] = [[tuple(t.to(dev) for t in s) for s in f] for f in v]
        else:
            out[k] = v
    return out


def drivable_iou(model, batch, gts, dts):
    model.eval()
    with torch.no_grad():
        res = model(batch, dts=dts)
    model.train()
    out = {}
    for dt in dts:
        m = SegMeter(model.grid, tolerances=(0.0,))
        g = gts[dt][0]
        k = model.spec.index(("drivable_area", "area", "drivable_area"))
        p = torch.sigmoid(res["out"][dt]["sem"][0, k]).cpu()
        m.add({"drivable_area": p}, {"drivable_area": g["drivable_area"]},
              mask=(g["camera"] > 0) & (g["undefined"] == 0), dt=dt)
        out[dt] = m.result()[round(float(dt), 3)]["drivable_area"]["all"]["all"]["iou@0"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    os.chdir(ROOT)
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)                                  # a busy CPU: few threads are faster
    dev = "cpu"
    if not a.cpu and torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 4 * 2 ** 30:
        dev = "cuda"
    ds = FrontRadarDataset(os.path.expanduser("~/data/nuscenes"), "v1.0-mini", n_cam=3, cam_stride=2, with_gt=True)
    batch = collate([ds[a.sample]])
    gts = sample_targets(batch["gt"])
    dts = tuple(sorted(gts))
    batch = to_device(batch, dev)
    model = BEVFusionModel(ModelConfig()).to(dev)
    model.train()
    if model.cfg.freeze_camera:                                      # frozen heads: their output is a constant of
        f = model.camera.image_features                              # the sample, computed once (exact, saves the
        with torch.no_grad():                                        # ResNet-50 in every step)
            memo = f(batch["img"].flatten(0, 1), batch["cam_vec"].flatten(0, 1))
        model.camera.image_features = lambda *_: memo
    crit = BEVLoss(grid=model.grid).to(dev)
    params = [p for p in model.parameters() if p.requires_grad] + list(crit.parameters())
    opt = torch.optim.Adam(params, lr=5e-4)
    iou0 = drivable_iou(model, batch, gts, dts)
    hist, t0 = [], time.time()
    for it in range(a.steps):
        L = crit(model(batch, dts=dts)["out"], gts, step_frac=1.0)
        opt.zero_grad(set_to_none=True)
        L["total"].backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        hist.append({k: float(v) for k, v in L.items() if k.startswith("group.") or k in ("total", "aux")})
        if it % 25 == 0 or it == a.steps - 1:
            print(f"step {it:4d}  total {hist[-1]['total']:8.3f}  " +
                  "  ".join(f"{k[6:]} {v:.3f}" for k, v in hist[-1].items() if k.startswith("group.")), flush=True)
    ms = (time.time() - t0) / a.steps * 1000
    iou1 = drivable_iou(model, batch, gts, dts)
    first, last = hist[0], sum((h["total"] for h in hist[-10:])) / 10
    checks = {
        "total loss falls by half": last <= 0.5 * first["total"] if first["total"] > 0 else last < first["total"],
        "drivable IoU dt 0 >= 0.8": iou1[0.0] >= 0.8,
        "drivable IoU dt +0.5 >= 0.7": iou1.get(0.5, 1.0) >= 0.7,
        "objects term falls": hist[-1]["group.objects"] < first["group.objects"],
        "motion term falls": hist[-1]["group.motion"] < first["group.motion"],
    }
    print(f"device {dev}, {a.steps} steps, {ms:.0f} ms per training step; drivable IoU before "
          + ", ".join(f"{dt:+.1f}: {iou0[dt]:.3f}" for dt in dts) + "; after "
          + ", ".join(f"{dt:+.1f}: {iou1[dt]:.3f}" for dt in dts))
    for k, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + k)
    if not all(checks.values()):
        sys.exit(1)
    print("end-to-end gate ok")


if __name__ == "__main__":
    main()
