"""throw-away (t26, track A): CPU cost of ONE lift call with a projection-table cache hit and with a miss (what a per-sample geometric image jitter would turn every call into), on the real
training grid (120 x 104 cells, 8 height slices, oversample 4) with random features. Run from bev_fusion/: ~/miniconda3/envs/gcar/bin/python temp/t26/lift_cost.py"""
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "training"), os.path.join(ROOT, "tests"), ROOT, os.path.join(ROOT, "..")]
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
import lift_pd as LP  # noqa: E402
from camera_encoder import nuscenes_front_ida  # noqa: E402
from feature_cache import CachedBEVFusionModel  # noqa: E402
from model import ModelConfig  # noqa: E402
from nuscenes_dataset import DST_HW  # noqa: E402
from test_slots import batch_for  # noqa: E402

cfg = ModelConfig(camera_features="dual", pretrained="", trunk_pretrained="")           # the real grid, 8 height slices, oversample 4
m = CachedBEVFusionModel(cfg).eval()
b = batch_for(cfg, B=1, Fn=3)
K, T, ida, hw = b["K"][0, 0], b["T_cam_car"][0, 0], b["ida"], b["hw"]
print(f"grid {m.grid.nx} x {m.grid.ny}, {len(m.grid.z_levels)} height slices, oversample {cfg.oversample}, threads {torch.get_num_threads()}")


def jittered(i):
    """a per-sample geometric jitter: scale 0.94 .. 1.06 and a crop offset up to 16 px (the t26 card's code 3): another ida, so another cache key"""
    g = torch.Generator().manual_seed(i)
    s, (dx, dy) = 0.94 + 0.12 * float(torch.rand(1, generator=g)), (torch.rand(2, generator=g) * 32 - 16).tolist()
    a = ida.clone()
    a[:2, :2] *= s
    a[0, 2] += dx
    a[1, 2] += dy
    return a


def time_it(fn, n=12):
    fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    t.sort()
    return 1000 * t[len(t) // 2]


ctx, prob = b["ctx"][0], b["prob"][0]
with torch.no_grad():
    lift = m.lift
    key = LP.lift_key(K, T, hw, ida)
    hit = time_it(lambda: lift(ctx, prob, K, T, hw, ida, key=key))
    idas = [jittered(i) for i in range(64)]
    keys = [LP.lift_key(K, T, hw, a) for a in idas]
    n = [0]

    def miss_call():
        i = n[0] % 64
        n[0] += 1
        return lift(ctx, prob, K, T, hw, idas[i], key=keys[i])

    miss = time_it(miss_call)
    build = time_it(lambda: LP.build_table(m.grid, lift.z_levels, K, T, hw, lift.depth_centres, ida))
    keyt = time_it(lambda: LP.lift_key(K, T, hw, ida), n=200)
print(f"one lift call (3 frames of one sample, DualBEV source): cache hit {hit:.1f} ms, miss {miss:.1f} ms (the table build alone {build:.1f} ms), the key {keyt * 1000:.0f} us")
print(f"per optimiser step (18 key frames x 2 sources = 36 lift calls in the forward, 36 again in the backward's recomputation): a hit costs {36 * hit:.0f} ms, all misses {36 * miss:.0f} ms (+{36 * (miss - hit):.0f} ms forward, "
      f"another +{36 * (miss - hit):.0f} ms if the recomputation misses too) on this CPU")
