"""scratch (2026-11-17, report chat, NOT a run that feeds a verdict): where do the 0.1 s of a token-head training step go?
A loader (getitem / densify / collate, workers 0 and 2), B eager step with phases, C sync count and places (sync debug mode), D torch.profiler (device busy share, kernel count, top kernels),
E CUDA-graph feasibility (fwd+bwd and the whole step incl. clip + AdamW capturable) with a surrogate loss, F GPU-resident frames: the batch assembled on the GPU.
2 CPU threads, one GPU job of about 2 GB; nothing is written except the json next to this file."""
import json
import os
import sys
import time
import warnings
import collections

import numpy as np
import torch

torch.set_num_threads(2)
R = os.environ["BFLC_ROOT"]
for sub_ in ("training", "training/grid", "training/data", "training/train", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.join(R, sub_))
import jtoken as JK  # noqa: E402
import jtoken_train as JT  # noqa: E402
import lane_ptok_data as LP  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prof_tokstep.json")
res = {}
dev = "cuda"


def save():
    json.dump(res, open(OUT, "w"), indent=1, default=str)


def sync():
    torch.cuda.synchronize()


def ms(t):
    return round(t * 1e3, 2)


# ---------------------------------------------------------------- A: the loader
ds, dl = LP.loader("train", 0, 8, 0)
idx = np.random.default_rng(0).permutation(len(ds))[:48]
t = time.time()
items = [ds[int(i)] for i in idx[:24]]
res["A_getitem_cold_ms_per_frame"] = ms((time.time() - t) / 24)
t = time.time()
items = [ds[int(i)] for i in idx[:24]]
res["A_getitem_warm_ms_per_frame"] = ms((time.time() - t) / 24)
f = ds.c.f
i0 = int(idx[0])
t = time.time()
for _ in range(24):
    ypair, ypl, yps = np.array(f["ypair"][i0]), np.array(f["ypair_long"][i0]), np.array(f["ypair_scale"][i0])
res["A_read_pairs_ms_per_frame"] = ms((time.time() - t) / 24)
t = time.time()
for _ in range(24):
    LP.densify(ypair, ypl, yps, ds.K)
res["A_densify_ms_per_frame"] = ms((time.time() - t) / 24)
t = time.time()
for _ in range(24):
    b = LP.collate(items[:8])
res["A_collate_ms_per_batch"] = ms((time.time() - t) / 24)
nb = sum(v.numel() * v.element_size() for v in b.values() if torch.is_tensor(v))
res["A_batch_MB"] = round(nb / 1e6, 1)
res["A_batch_MB_by_field"] = {k: round(v.numel() * v.element_size() / 1e6, 2) for k, v in b.items() if torch.is_tensor(v)}
t = time.time()
g = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
sync()
res["A_h2d_ms_per_batch"] = ms(time.time() - t)
ds2, dl2 = LP.loader("train", 0, 8, 2)
it = iter(dl2)
for _ in range(5):
    next(it)
t = time.time()
for _ in range(30):
    next(it)
res["A_loader_workers2_ms_per_batch"] = ms((time.time() - t) / 30)
del it
save()

# ---------------------------------------------------------------- B: the eager step
args = dict(JT.ARCH_DEFAULTS)
model = JT.build_head(args).to(dev)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.98), eps=1e-6)


class NoRun:
    def log(self, *a, **k):
        pass


loss_fn = JT.make_loss(NoRun(), 0.0, 0, 0.25, "none", "sum", None, (), 1.0)
JK.SWITCH_ORIG = JK.SWITCH
model._step_py = 10 ** 6                                   # past the key-weight switch: the learned path, as in most of a run
gb = JT.to_device(b, dev)


def eager_step(batch, phases=None):
    p = phases
    t0 = time.time()
    with torch.autocast("cuda", torch.bfloat16):
        loss, info = loss_fn(model, batch)
    if p is not None:
        sync(); p["fwd+loss"] += time.time() - t0; t0 = time.time()
    if not torch.isfinite(loss):
        return
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if p is not None:
        sync(); p["backward"] += time.time() - t0; t0 = time.time()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
    float(gn)
    opt.step()
    float(loss)
    if p is not None:
        sync(); p["clip+opt"] += time.time() - t0


for _ in range(8):
    eager_step(gb)
sync()
t = time.time()
for _ in range(40):
    eager_step(gb)
sync()
res["B_eager_fixed_gpu_batch_ms_per_step"] = ms((time.time() - t) / 40)
ph = collections.defaultdict(float)
for _ in range(20):
    eager_step(gb, ph)
res["B_phases_ms_per_step"] = {k: ms(v / 20) for k, v in ph.items()}
# the whole step as the trainer sees it: a batch from the loader (workers 0), host -> device
t = time.time()
it0 = iter(dl)
for _ in range(20):
    eager_step(next(it0))
sync()
res["B_eager_with_loader_w0_ms_per_step"] = ms((time.time() - t) / 20)
res["B_params"] = JK.count_params(model)
res["B_peak_gpu_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
save()

# ---------------------------------------------------------------- C: syncs
torch.cuda.set_sync_debug_mode("warn")
with warnings.catch_warnings(record=True) as W:
    warnings.simplefilter("always")
    eager_step(gb)
torch.cuda.set_sync_debug_mode(0)
cnt = collections.Counter((os.path.basename(w.filename), w.lineno) for w in W)
res["C_syncs_per_step"] = len(W)
res["C_sync_places"] = {f"{k[0]}:{k[1]}": v for k, v in cnt.most_common()}
save()

# ---------------------------------------------------------------- D: profiler
from torch.profiler import profile, ProfilerActivity  # noqa: E402

sync()
t = time.time()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(15):
        eager_step(gb)
    sync()
wall = time.time() - t
ka = prof.key_averages()
dev_total = sum(e.self_device_time_total for e in ka) / 1e3
res["D_wall_ms_per_step_profiled"] = ms(wall / 15)
res["D_device_busy_ms_per_step"] = round(dev_total / 15, 2)
res["D_kernel_launches_per_step"] = int(sum(e.count for e in ka if e.self_device_time_total > 0) / 15)
top = sorted(ka, key=lambda e: -e.self_device_time_total)[:14]
res["D_top_kernels_us_per_step"] = {e.key[:70]: round(e.self_device_time_total / 15, 1) for e in top}
res["D_sync_calls_per_step"] = {e.key: round(e.count / 15, 1) for e in ka if any(s in e.key for s in ("cudaStreamSynchronize", "cudaMemcpyAsync", "cudaDeviceSynchronize", "cudaMemcpy"))}
save()

# ---------------------------------------------------------------- E: CUDA graph feasibility (surrogate loss, no float() inside)
def surrogate(m, batch):
    with torch.autocast("cuda", torch.bfloat16):
        out = m(batch)
        K = out["cand"].shape[-1]
        l = out["logp"][..., :K].clamp(min=-30).mean() + out["logp"][..., K].mean() + out["types"].float().mean() + out["scale"].float().mean()
    return l


def whole(m, o, batch):
    o.zero_grad(set_to_none=True)
    l = surrogate(m, batch)
    l.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 3.0)
    o.step()
    return l


model2 = JT.build_head(args).to(dev)
model2.train()
model2._step_py = 10 ** 6
opt2 = torch.optim.AdamW(model2.parameters(), lr=torch.tensor(1e-3, device=dev), weight_decay=1e-4, betas=(0.9, 0.98), eps=1e-6, capturable=True)
static = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in gb.items() if k in JK.INPUTS or k in ("y", "ylong", "ytype", "yscale")}
sync()
t = time.time()
for _ in range(30):
    whole(model2, opt2, static)
sync()
res["E_eager_surrogate_whole_step_ms"] = ms((time.time() - t) / 30)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(4):
        whole(model2, opt2, static)
torch.cuda.current_stream().wait_stream(s)
sync()
graph = torch.cuda.CUDAGraph()
try:
    opt2.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        static_l = whole(model2, opt2, static)
    sync()
    t = time.time()
    for _ in range(50):
        graph.replay()
    sync()
    res["E_graph_whole_step_replay_ms"] = ms((time.time() - t) / 50)
    res["E_graph_capture"] = "ok"
    res["E_graph_loss_finite"] = bool(torch.isfinite(static_l).item())
except Exception as ex:  # noqa: BLE001
    res["E_graph_capture"] = f"FAILED {type(ex).__name__}: {str(ex)[:300]}"
save()

# ---------------------------------------------------------------- F: GPU-resident frames: the training fields of a batch assembled on the GPU
fields = ("tok", "pos", "dirv", "e", "e_raw", "s", "uout", "uin", "flags", "valid", "gt_type")
tot = 0
for n_ in fields:
    tot += f[n_].dtype.itemsize * int(np.prod(f[n_].shape))
for n_ in ("ypair", "ypair_long", "ypair_scale"):
    tot += f[n_].dtype.itemsize * int(np.prod(f[n_].shape))
res["F_train_fields_GB_for_the_whole_TRAIN_cache"] = round(tot / 1e9, 2)
res["F_frames"] = len(ds)
P = f["ypair"].shape[1]
yp = torch.from_numpy(np.array(f["ypair"][: 8])).to(dev).long()          # (8, P, 2)
ypl = torch.from_numpy(np.array(f["ypair_long"][: 8])).to(dev)
yps = torch.from_numpy(np.array(f["ypair_scale"][: 8])).to(dev)
K = ds.K


def gpu_densify(yp, ypl, yps):
    B = yp.shape[0]
    ok = yp[..., 0] >= 0
    bi = torch.arange(B, device=yp.device)[:, None].expand_as(ok)[ok]
    i, j = yp[..., 0][ok], yp[..., 1][ok]
    y = torch.zeros(B, K, K, dtype=torch.uint8, device=yp.device)
    yl = torch.zeros(B, K, K, dtype=torch.bool, device=yp.device)
    ys = torch.full((B, K, K, 2), float("nan"), device=yp.device)
    y[bi, i, j] = 1
    yl[bi, i, j] = ypl[ok]
    ys[bi, i, j] = yps[ok]
    return y, yl, ys


gpu_densify(yp, ypl, yps)
sync()
t = time.time()
for _ in range(30):
    gpu_densify(yp, ypl, yps)
sync()
res["F_gpu_densify_ms_per_batch"] = ms((time.time() - t) / 30)
res["F_cpu_getitem_plus_collate_ms_per_batch_w0"] = ms(res["A_getitem_warm_ms_per_frame"] / 1e3 * 8 + res["A_collate_ms_per_batch"] / 1e3)
save()
print(json.dumps(res, indent=1, default=str))
