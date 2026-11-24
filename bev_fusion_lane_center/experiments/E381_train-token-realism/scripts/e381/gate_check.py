"""E381 pre-run check (CPU, scratch, before any run; critic B1): how many GT links leave the head's gate when the inputs carry the calibrated noise. The E373 ref head (frozen, eval) on the first N TRAIN batches of the
seed's cache (bs 8, the loader without shuffling), the token loss's own info (`gt_outside`, `gt_links`, `gt_class`) clean and with `jtoken_train.add_noise` at (SP, SE). No head score is read: the loss info counts labels
against the gate only. usage: python gate_check.py <seed> <SP> <SE> [N]"""
import os
import sys

import numpy as np
import torch

R = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "training/train", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import jtoken as JK  # noqa: E402
import jtoken_train as JKT  # noqa: E402
import lane_ptok_data as LP  # noqa: E402

torch.set_num_threads(2)
S, SP, SE = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 60
m, args = JKT.load_model(f"e373_cb_s{S}", "cpu")
ds = LP.PTokData("train", S, "e292_a", None, 0.0)
idx = np.linspace(0, len(ds) - 1, N * 8).round().astype(int)
acc = {k: np.zeros(3) for k in ("clean", "noisy")}
with torch.no_grad():
    for i in range(N):
        b = LP.collate([ds[int(j)] for j in idx[i * 8:(i + 1) * 8]])
        for name, bb in (("clean", b), ("noisy", JKT.add_noise(b, SP, SE, 2 + S, i)[0])):
            _, info = JK.jtoken_loss(m(bb), bb, none_weight=0.25)
            acc[name] += [info["gt_outside"], info["gt_links"], info["gt_class"]]
for name, (o, g, c) in acc.items():
    print(f"seed {S} {name}: {N} batches x 8 TRAIN frames, GT links {g:.0f}, outside the gate {o:.0f} ({o / g:.4f}), ylong against the chord class {c:.0f} ({c / g:.4f})")
print(f"RESULT seed {S} SP {SP} SE {SE}: gate-lost share clean {acc['clean'][0] / acc['clean'][1]:.4f}, noisy {acc['noisy'][0] / acc['noisy'][1]:.4f}; class flips clean {acc['clean'][2] / acc['clean'][1]:.4f}, noisy {acc['noisy'][2] / acc['noisy'][1]:.4f}")
