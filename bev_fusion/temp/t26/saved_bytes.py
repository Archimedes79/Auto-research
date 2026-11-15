"""throw-away (t26, track A): the bytes autograd keeps for the backward of the camera path, student_frames 'current' vs 'all' (EfficientViT-B1, B = 2 key frames, F = 3), and the peak resident set of the call (CPU)"""
import os
import resource
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "training"), os.path.join(ROOT, "tests"), ROOT, os.path.join(ROOT, "..")]
torch.set_num_threads(2)
from feature_cache import CachedBEVFusionModel  # noqa: E402
from model import ModelConfig  # noqa: E402
from test_slots import BASE, batch_for  # noqa: E402

DUAL = dict(BASE, camera_features="dual")


def saved_bytes(fn):
    seen, total = set(), [0]

    def pack(x):
        k = (x.data_ptr(), x.dtype, tuple(x.shape))
        if x.numel() and k not in seen:
            seen.add(k)
            total[0] += x.numel() * x.element_size()
        return x

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        fn()
    return total[0]


torch.manual_seed(0)
img = torch.randint(0, 256, (2, 3, 256, 704), dtype=torch.uint8)
for frames in ("current", "all"):
    m = CachedBEVFusionModel(ModelConfig(**dict(DUAL, camera_student="evit_b1", student_trainable=True, student_frames=frames))).train()
    b = batch_for(ModelConfig(**DUAL), B=2, Fn=3)
    b["lane_img"], b["lane_img_old"] = img, torch.randint(0, 256, (2, 2, 3, 256, 704), dtype=torch.uint8)
    print(frames, f"autograd keeps {saved_bytes(lambda: m.camera_bevs(b)) / 1e6:.1f} MB for the backward of the camera path (2 samples x 3 frames, real grid is the test grid 32 x 32)")
print(f"peak resident set of the process: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3:.0f} MB")
