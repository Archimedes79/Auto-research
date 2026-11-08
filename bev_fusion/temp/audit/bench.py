"""One-off: seconds per optimiser step and peak GPU memory of a config (real cached data, real clips). Usage: python temp/audit/bench.py <cfg.json> [n_steps]"""
import functools, json, sys, time
import torch
sys.path.insert(0, "training")
from train import ClipDataset, TrainConfig, build, collate_clips, make_datasets, run_clip   # noqa: E402
cfg = TrainConfig(**json.load(open(sys.argv[1])))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
cfg.workers = 6
torch.manual_seed(0)
model, loss, _ = build(cfg)
model, loss = model.cuda(), loss.cuda()
print("trainable params (M):", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, flush=True)
train_ds, _ = make_datasets(cfg)
ds = ClipDataset(train_ds, cfg.seq_len) if cfg.seq_len > 1 else train_ds
loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers,
                                     collate_fn=functools.partial(collate_clips, fn=__import__("feature_cache").collate_cached) if cfg.seq_len > 1 else None, drop_last=True)
model.train()
params = [p for p in model.parameters() if p.requires_grad] + list(loss.parameters())
it = iter(loader)
ts = []
for i in range(n + 2):
    clip = next(it)
    torch.cuda.synchronize(); t0 = time.time()
    for p in params: p.grad = None
    run_clip(model, loss, clip, 0.5, "cuda", cfg.amp, cfg, backward=True, augment=True)
    torch.cuda.synchronize(); ts.append(time.time() - t0)
print(f"micro-batch {cfg.batch_size} clips x {cfg.seq_len}: {sum(ts[2:]) / n:.2f} s each; optimiser step = x{cfg.accum} = {cfg.accum * sum(ts[2:]) / n:.2f} s; peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
