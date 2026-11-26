"""Reading the toy data: the manifest, the raw shards and their truth. Training, running and evaluation read through here (numpy only)."""
import hashlib
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))        # the problem folder
MANIFEST = os.path.join(ROOT, "data", "manifest.json")


def data_root(root=""):
    """the folder of the heavy data: the argument (a relative one is relative to the problem folder), else TOY_DATA, else data/store. Every site resolves it for itself"""
    r = root or os.environ.get("TOY_DATA") or os.path.join("data", "store")
    return os.path.normpath(os.path.join(ROOT, os.path.expanduser(r)))


def manifest(path=MANIFEST):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def manifest_hash(path=MANIFEST):
    """12 hex digits of the manifest: the identity of the raw data"""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split_shards(split, man=None):
    """the manifest entries of a split, in manifest order"""
    out = [s for s in (man or manifest())["shards"] if s["split"] == split]
    if not out:
        raise ValueError(f"split {split!r}: no shard of the manifest has it")
    return out


def raw_path(root, name):
    return os.path.join(data_root(root), "raw", name + ".npy")


def truth_path(root, signature, name):
    return os.path.join(data_root(root), "gt", signature, name + ".npy")


def load_split(split, root="", signature=None):
    """the frames of a split, shard after shard: {frames (N, S, S) float32, ann (N, 3), scene (N,) str, index (N,) int, masks (N, S, S) bool when a truth signature is given}"""
    parts = {"frames": [], "ann": [], "scene": [], "index": [], "masks": []}
    for s in split_shards(split):
        p = raw_path(root, s["name"])
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p}: the raw shard is missing (bash tools/prepare.sh make <config>)")
        a = np.load(p, allow_pickle=False)
        parts["frames"].append(a["frame"])
        parts["ann"].append(a["ann"])
        parts["scene"].append(a["scene"])
        parts["index"].append(a["index"])
        if signature:
            t = truth_path(root, signature, s["name"])
            if not os.path.isfile(t):
                raise FileNotFoundError(f"{t}: the truth of signature {signature} is missing (bash tools/prepare.sh make <config>)")
            parts["masks"].append(np.load(t, allow_pickle=False))
    return {k: np.concatenate(v) for k, v in parts.items() if v}
