"""The cached frames the lane data is built on: where the frozen foundation features and the per-part files lie, which rows a split holds, the feature
normalisation (D152 follow-up: moved here from the reference chain, which reads the same functions: training/data/p8_lib.py, p16_data.py).

    D                      the folder of the cached features, token lists, poses and graphs (WSL home; data/README.md)
    cache(name, part)      a stored per-part file, read once: gt | rt (route) | sd | rpts | poses | src (the perceived-graph cache of the reference chain)
    load_src(src)          cache("src", src)
    norm_stats()           (mean, std) of the feature channels
    Frames(split, ks)      "train" | "val_clean" | "test_clean": part, rows, ks, mean, std and gt (the GT graphs of the part, read on first use)

Self-test (real cached files, no features): wsl bash tools/env/_py14.sh net/prep/frames.py
"""
import json
import os
import pickle

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = f"{os.path.expanduser('~')}/exp_gcar_b123"
OUT = f"{ROOT}/runs"
SPLITS = ("train", "val_clean", "test_clean")
_C = {}


def cache(name, part):
    key = (name, part)
    if key not in _C:
        if name == "gt":
            _C[key] = pickle.load(open(f"{D}/graph_gt_{part}.pkl", "rb"))
        elif name == "rt":
            _C[key] = pickle.load(open(f"{D}/route_gt_{part}.pkl", "rb"))
        elif name == "sd":
            _C[key] = pickle.load(open(f"{D}/sd_{part}.pkl", "rb"))
        elif name == "rpts":
            _C[key] = pickle.load(open(f"{D}/route_pts_{part}.pkl", "rb"))
        elif name == "poses":
            z = np.load(f"{D}/poses_{part}.npz")
            _C[key] = {k: z[k] for k in z.files}
        elif name == "src":
            _C[key] = pickle.load(open(f"{D}/p8/{part}.pkl", "rb"))
    return _C[key]


def load_src(src):
    """perceived-graph cache of p8_build.py"""
    return cache("src", src)


def norm_stats():
    z = np.load(f"{OUT}/refiner_stats_dec32_logit7.npz")
    return z["mean"].astype(np.float32), z["std"].astype(np.float32)


def train_rows(gt):
    """every row of the train part that has a GT graph"""
    return np.arange(len(gt))


class Frames:
    """the rows of a split in the cached files: `part` ("train" | "val"), `rows` (the rows of the part, in the split's order), `ks` (the chosen indices into rows)"""

    def __init__(self, split="train", ks=None):
        assert split in SPLITS, (split, SPLITS)
        self.mean, self.std = norm_stats()
        if split == "train":
            self.part = "train"
            self.rows = train_rows(self.gt)
        else:
            src = load_src(split)
            self.part = src["part"]
            self.rows = np.asarray([int(r) for r in src["ids"]])
        self.ks = np.arange(len(self.rows)) if ks is None else np.asarray(ks)

    @property
    def gt(self):
        return cache("gt", self.part)


def _selftest():
    sp = json.load(open(f"{ROOT}/data/splits.json"))                       # the project's split file: the rows from outside this module
    z = np.load(f"{OUT}/refiner_stats_dec32_logit7.npz")
    for split in ("train", "val_clean"):                                    # never the TEST split in a test
        f = Frames(split, ks=[0, 1])
        want = sp["train" if split == "train" else "val"]
        assert [int(r) for r in f.rows] == [int(r) for r in want], f"{split}: the rows are not data/splits.json in its order"
        assert np.array_equal(f.mean, z["mean"].astype(np.float32)) and np.array_equal(f.std, z["std"].astype(np.float32)) and not np.array_equal(f.mean, f.std)
        assert f.part == ("train" if split == "train" else "val") and f.mean.dtype == np.float32 and f.mean.shape == f.std.shape and list(f.ks) == [0, 1]
        assert len(f.rows) == (8414 if split == "train" else 873) and f.rows.dtype == np.int64 and f.rows.max() < len(f.gt) and len(set(f.rows.tolist())) == len(f.rows)
        assert cache("gt", f.part) is f.gt
    try:
        Frames("oof0")
        raise SystemExit("an unknown split was accepted")
    except AssertionError:
        pass
    print("frames selftest ok: train and val_clean rows, normalisation, one read per file")


if __name__ == "__main__":
    _selftest()
