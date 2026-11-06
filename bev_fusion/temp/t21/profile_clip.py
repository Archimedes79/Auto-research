"""One training clip of the composed config (default t21_rec_s0): wall time of forward / backward per keyframe and the top CUDA / CPU ops of
torch.profiler. Scratch of the t21 diagnosis (the rec arm ran at 53 s/step, the window arm at 3.3). Run from bev_fusion/:
    bash tools/env/_py14.sh temp/t21/profile_clip.py temp/compose/t21_rec_s0.json [--mode rec|ctl] [--prof]
The GPU must be free (the run was stopped)."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training"))
import torch  # noqa: E402

import train  # noqa: E402

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "temp/compose/t21_rec_s0.json"
use_prof = "--prof" in sys.argv
n_clip = 2
cfg = train.read_config(cfg_path)
cfg = train.dataclasses.replace(cfg, workers=0)
device = train.resolve_device(cfg.device)
train.seed_all(cfg.seed)
train.apply_speed_flags(cfg)
model, loss, _ = train.build(cfg)
model, loss = model.to(device), loss.to(device)
if cfg.init_from:
    train.warm_start(cfg.init_from, model, loss)
train_ds, _ = train.make_datasets(cfg)
if cfg.seq_len > 1:
    train_ds = train.ClipDataset(train_ds, cfg.seq_len)
loader = train.make_loader(train_ds, cfg, True)
it = iter(loader)
model.train()
loss.train()
opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)     # nothing is updated: lr 0, only the graph is run


def sync():
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def one(item, tag):
    sync()
    t0 = time.time()
    out, ok, _, _ = train.run_clip(model, loss, item, 0.5, device, cfg.amp, cfg, backward=True, augment=False)
    sync()
    opt.zero_grad(set_to_none=True)
    print(f"{tag}: clip {time.time() - t0:.2f} s, finite {ok}, peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)


items = [next(it) for _ in range(n_clip + 1)]
one(items[0], "warm-up")
t0 = time.time()
one(items[1], "timed")
if use_prof:
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        one(items[2], "profiled")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=60), flush=True)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12, max_name_column_width=60), flush=True)
