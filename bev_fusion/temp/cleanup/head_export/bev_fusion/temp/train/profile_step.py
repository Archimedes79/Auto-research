"""One-off: where does the main process spend its time in one micro-batch of the combined training? cProfile of run_clip
(forward + loss + backward) on 1 clip x 3 keyframes, GPU, after 2 warm-up clips; prints the top functions by tottime
and cumtime, and the wall time per clip split into 'cuda busy' (synchronize) and the rest."""
import cProfile
import json
import pstats
import sys
import time

import torch

sys.path.insert(0, "training")
from train import ClipDataset, TrainConfig, build, collate_clips, make_datasets, run_clip   # noqa: E402
import functools                                                                           # noqa: E402
from feature_cache import collate_cached                                                  # noqa: E402

cfg = TrainConfig(**json.load(open("training/configs/t1_r3_combined.json")))
cfg.batch_size, cfg.accum, cfg.workers = 1, 1, 2
torch.manual_seed(0)
model, loss, _ = build(cfg)
model, loss = model.cuda(), loss.cuda()
train_ds, _ = make_datasets(cfg)
ds = ClipDataset(train_ds, cfg.seq_len)
loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=2,
                                     collate_fn=functools.partial(collate_clips, fn=collate_cached))
it = iter(loader)
model.train()
clips = [next(it) for _ in range(8)]                          # loading is excluded: only the step is timed
params = [p for p in model.parameters() if p.requires_grad] + list(loss.parameters())


def step(clip):
    for p in params:
        p.grad = None
    run_clip(model, loss, clip, 0.5, "cuda", cfg.amp, cfg, backward=True)
    torch.cuda.synchronize()


for c in clips[:2]:
    step(c)
t0 = time.time()
pr = cProfile.Profile()
pr.enable()
for c in clips[2:]:
    step(c)
pr.disable()
dt = (time.time() - t0) / len(clips[2:])
print(f"wall time per clip (3 keyframes x 1 sample x 3 dt): {dt:.2f} s; GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
st = pstats.Stats(pr)
st.sort_stats("tottime").print_stats(18)
st.sort_stats("cumtime").print_stats(22)
