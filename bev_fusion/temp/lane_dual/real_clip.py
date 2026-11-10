"""scratch (t20): one REAL clip of the large split (val) through the t20 model and loss, forward + backward, no optimiser step beyond the zero-start probe:
the dataset carries lane_img (uint8, 256 x 704), the batch's ida is the DualBEV one, the loss is finite, the 1 m terms exist, the gradient reaches the branch
(mix at step 0, the rest after one SGD step on mix). Not a training run. Run: wsl bash -c 'cd bev_fusion && PYTHONPATH=/mnt/c/Auto-research $HOME/miniconda3/envs/gcar/bin/python temp/lane_dual/real_clip.py'"""
import math
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]
os.chdir(ROOT)
import train as T  # noqa: E402
from camera_encoder import nuscenes_front_ida  # noqa: E402
from nuscenes_dataset import DST_HW  # noqa: E402

t0 = time.time()
cfg = T.read_config("temp/compose/t20_check.preview.json")
cfg.workers = 0
model, loss, cached = T.build(cfg)
model, loss = model.cuda().train(), loss.cuda()
_, val_ds = T.make_datasets(cfg)
clips = T.ClipDataset(val_ds, cfg.seq_len)
loader = T.make_loader(torch.utils.data.Subset(clips, [100, 900, 1700]), cfg, False, workers=0)
clip = next(iter(loader))
print(f"built and loaded in {time.time() - t0:.0f} s; clip of {len(clip)} keyframes, batch {len(clip[0]['tokens'])}", flush=True)
b = clip[0]
ida_ok = torch.equal(b["ida"], nuscenes_front_ida(dst_hw=DST_HW))
print("lane_img", tuple(b["lane_img"].shape), b["lane_img"].dtype, "mean", float(b["lane_img"].float().mean()), "| ida is the DualBEV one:", ida_ok,
      "| ida2 differs:", not torch.equal(b["ida"], b["ida2"]), "| prob", tuple(b["prob"].shape), "prob2", tuple(b["prob2"].shape), flush=True)
assert b["lane_img"].dtype == torch.uint8 and tuple(b["lane_img"].shape) == (cfg.batch_size, 3, *DST_HW) and ida_ok


def run():
    model.zero_grad(set_to_none=True)
    loss.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    acc, ok, _, _ = T.run_clip(model, loss, clip, 1.0, "cuda", "off", cfg, backward=True)
    return acc, ok, torch.cuda.max_memory_allocated() / 1e9


acc, ok, peak = run()
print(f"step 0: finite {ok}, total {acc['total']:.3f}, groups {', '.join(f'{k[6:]} {v:.3f}' for k, v in acc.items() if k.startswith('group.'))}, peak {peak:.2f} GB", flush=True)
g = {n: (q.grad.abs().sum().item() if q.grad is not None else None) for n, q in model.named_parameters() if n.startswith("lane.") and q.requires_grad}
print("lane grads at step 0: mix", g["lane.mix.weight"], "| others nonzero:", [n for n, v in g.items() if v and not n.startswith("lane.mix")], flush=True)
with torch.no_grad():
    model.lane.mix.weight.add_(model.lane.mix.weight.grad, alpha=-0.1 / max(model.lane.mix.weight.grad.abs().max().item(), 1e-12))
acc, ok, peak = run()
g = {n: (q.grad.abs().sum().item() if q.grad is not None else None) for n, q in model.named_parameters() if n.startswith("lane.") and q.requires_grad}
dead = [n for n, v in g.items() if not v]
print(f"after one step on mix: finite {ok}, total {acc['total']:.3f}, lane tensors with a gradient {len(g) - len(dead)} of {len(g)}, dead {dead}", flush=True)
h = {n: q.grad.abs().sum().item() for n, q in model.named_parameters() if n.startswith("head_1m.") and q.grad is not None}
print("head_1m tensors with a gradient:", len(h), "of", sum(1 for n, _ in model.named_parameters() if n.startswith("head_1m.")), flush=True)
assert ok and math.isfinite(acc["total"]) and not dead
