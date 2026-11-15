"""CPU micro-benchmark (scratch): the host cost of JEnds.rank_matrix at batch 1 with realistic keypoint counts, to see how much of the 9.7 ms of the pair head it is."""
import os, sys, time
import numpy as np
import torch
ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/train", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jends as JE
torch.set_num_threads(1)
m = JE.JEnds(d=64, order=True, evidence=True)
rng = np.random.default_rng(0)
for nval in (8, 16, 24, 32):
    pos = torch.tensor(rng.uniform(-40, 40, (1, 64, 2)), dtype=torch.float32)
    ang = rng.uniform(0, 2 * np.pi, (1, 64))
    D = torch.tensor(np.stack([np.cos(ang), np.sin(ang)], -1), dtype=torch.float32)
    valid = torch.zeros(1, 2, 32, dtype=torch.bool)
    valid[:, :, :nval] = True
    for _ in range(5):
        m.rank_matrix(pos, D, valid)
    t0 = time.perf_counter()
    n = 200
    for _ in range(n):
        m.rank_matrix(pos, D, valid)
    print(f"valid keypoints per kind {nval:2d}: rank_matrix {(time.perf_counter() - t0) / n * 1e3:.3f} ms per frame (CPU, one thread)")
