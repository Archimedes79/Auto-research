"""E377 throw-away: `jtoken.near_none` (float32, the loss's label) against the float64 label of `e377_near.py` (R 2.0) on the card's frames (run, seed, split, every) -> per-frame agreement and the near tokens per frame.
usage: bash tools/env/_py14.sh temp/e377/e377c_label_check.py e373_cb_s0 0 train 3"""
import argparse
import os
import sys

import numpy as np
import torch

torch.set_num_threads(2)
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jtoken_eval as E  # noqa: E402
import jtoken as JK  # noqa: E402

run, seed, split, every = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
r = E.read_dump(argparse.Namespace(run=run, model="model.pt", seed=seed, split=split, every=every, drop=0.0, drop_seed=0), E.RUNS, 0)
ds, idx = r["ds"], r["idx"]
mine, ref, diff = [], [], 0
for i, t, v in zip(idx, r["toks"], r["valid"]):
    it = ds[int(i)]
    yt, v = np.asarray(it["ytype"]), np.asarray(v, bool)
    pos, dirv = np.asarray(t["pos"], float), np.asarray(t["dirv"], float)
    asg = v & (yt != 0)
    near_ref = v & (yt == 0) & ((np.linalg.norm(pos[:, None] - pos[None], axis=2) <= 2.0) & (dirv @ dirv.T >= 0.5) & asg[None, :]).any(1)
    tt = lambda k: torch.from_numpy(np.asarray(it[k]))[None]  # noqa: E731
    near = JK.near_none(tt("pos").float(), tt("dirv").float(), tt("valid").bool(), tt("ytype").long())[0].numpy()
    assert np.array_equal(np.asarray(it["valid"], bool), v), "valid of the item differs from the dump's"
    mine.append(int(near.sum()))
    ref.append(int(near_ref.sum()))
    diff += int((near != near_ref).sum())
print(f"{run} {split} every {every}: {len(mine)} frames; near_none (float32, loss) {np.mean(mine):.2f} per frame, the card's float64 label {np.mean(ref):.2f}; tokens that differ {diff}; frames that differ {sum(a != b for a, b in zip(mine, ref))}")
