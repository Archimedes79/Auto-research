"""throw-away (t26, track A): relative CPU cost of the student's gradient-free forward on the OLDER frames against the forward + recomputation + backward of the NEWEST frame (EfficientViT-B1, random
weights, 256 x 704, batch of 2 key frames = 2 newest + 4 older frames per call). Run from bev_fusion/: ~/miniconda3/envs/gcar/bin/python temp/t26/student_cost.py"""
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "..")]
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
import student_encoder as S  # noqa: E402

st = S.StudentEncoder("efficientvit_b1", pretrained=False)
st.keep_bn_eval()
st.checkpoint_encode = True
st.train()
n_par = sum(p.numel() for p in st.parameters())
cur = torch.randint(0, 256, (2, 3, 256, 704), dtype=torch.uint8)
old = torch.randint(0, 256, (4, 3, 256, 704), dtype=torch.uint8)


def timed(fn, n=3):
    fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return sorted(t)[len(t) // 2]


def older():
    with torch.no_grad():
        return st(old)


def newest_fwd_bwd():
    out = st(cur)                                          # checkpoint_encode: the backbone is recomputed in the backward
    (out["logit16"].square().mean() + out["ctx8"].square().mean()).backward()


def newest_fwd_only():
    with torch.no_grad():
        return st(cur)


t_old, t_new, t_fwd = timed(older), timed(newest_fwd_bwd), timed(newest_fwd_only)
print(f"EfficientViT-B1 ({n_par / 1e6:.1f} M parameters), {torch.get_num_threads()} threads, CPU")
print(f"forward only, 2 frames:                      {1000 * t_fwd:7.0f} ms  ({1000 * t_fwd / 2:.0f} ms per frame)")
print(f"older frames, no_grad forward, 4 frames:     {1000 * t_old:7.0f} ms  ({1000 * t_old / 4:.0f} ms per frame)")
print(f"newest frames, forward + recompute + backward, 2 frames: {1000 * t_new:7.0f} ms  ({1000 * t_new / 2:.0f} ms per frame)")
print(f"ratio older-frame cost per step (36 frames) / newest-frame cost per step (18 frames) = {(t_old / 4 * 36) / (t_new / 2 * 18):.2f} "
      f"(with seq_bptt a clip of 3 runs the encoder 3 + 1 + 2 + 3 = 9 times for 3 frames: the newest side costs about {9 / 3:.0f} forwards + the backward, this one counted 1 recompute)")
