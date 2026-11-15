"""E276 step 4 gate: the l4 lane graphs have EXACTLY the geometry of l3 (every key of the l3 graph equal, only the l4 labels and parameters added), on all 873 VAL frames and a random sample of the rest.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh check_geometry.py <l3 store> <l4 store> [n_random]
Exit 1 and a list of the first differences when a frame differs; prints the counts.
"""
import glob
import gzip
import os
import pickle
import sys

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "training"))
import qgraph_eval as QE  # noqa: E402

s3, s4 = sys.argv[1], sys.argv[2]
n_rand = int(sys.argv[3]) if len(sys.argv) > 3 else 500
L4_KEYS = {"tier", "vis", "occ", "tier_src"}                       # per piece
NEW_TOP = {"ego_path", "tier_info"}
bad = []


def same(a, b, path=""):
    """a (l3) must be contained in b (l4): equal values, ignoring the l4 additions"""
    if isinstance(a, dict):
        if not isinstance(b, dict):
            return [path]
        out = []
        for k, v in a.items():
            if k == "signature" or (path == "" and k == "params"):
                continue
            if k not in b:
                out.append(path + "/" + k + " (missing)")
                continue
            out += same(v, b[k], path + "/" + str(k))
        return out
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return [path + " (length)"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += same(x, y, path + f"[{i}]")
            if len(out) > 3:
                break
        return out
    if isinstance(a, np.ndarray):
        return [] if (isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b)) else [path + " (array)"]
    return [] if a == b else [path + f" ({a!r} != {b!r})"]


val = {f.token for f in QE.val_frames()}
t3 = {os.path.basename(f)[:-7] for f in glob.glob(os.path.join(s3, "*.pkl.gz"))}
t4 = {os.path.basename(f)[:-7] for f in glob.glob(os.path.join(s4, "*.pkl.gz"))}
common = sorted(t3 & t4)
rng = np.random.RandomState(7)
pick = sorted(val & t3 & t4) + [common[i] for i in rng.choice(len(common), min(n_rand, len(common)), replace=False)]
pick = list(dict.fromkeys(pick))
print(f"l3 store {len(t3)} frames, l4 store {len(t4)} frames, common {len(common)}; checking {len(pick)} (VAL {len(val & t3 & t4)} of {len(val)} + a random sample)", flush=True)
labelled = 0
for i, tok in enumerate(pick):
    a = pickle.load(gzip.open(os.path.join(s3, tok + ".pkl.gz"), "rb"))
    b = pickle.load(gzip.open(os.path.join(s4, tok + ".pkl.gz"), "rb"))
    d = same(a, b)
    if d:
        bad.append((tok, d[:3]))
    labelled += int(all(("tier" in p and "vis" in p and "occ" in p and "tier_src" in p) for p in b["pieces"]) and "ego_path" in b and "tier_info" in b)
    if (i + 1) % 300 == 0:
        print(i + 1, "checked", flush=True)
print(f"geometry equal in {len(pick) - len(bad)} of {len(pick)} frames; l4 labels present in {labelled}")
for t, d in bad[:5]:
    print("DIFFERENT:", t, d)
sys.exit(1 if bad else 0)
