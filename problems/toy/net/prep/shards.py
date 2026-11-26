"""Reading the toy data: the manifest, the raw shards and their ground truth. Training, running and evaluation read through here (numpy only).

Every shard is checked against its hash when it is read (the manifest for the raw data, the store's index for the ground truth), and a holdout
split of problem.json opens only in a run of `research test`."""
import hashlib
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))        # the problem folder
MANIFEST = os.path.join(ROOT, "dataset", "manifest.json")


def data_root(root=""):
    """the folder of the toy's data: the config's data_root (absolute in a run), else RESEARCH_DATA/toy"""
    r = root or (os.path.join(os.environ["RESEARCH_DATA"], "toy") if os.environ.get("RESEARCH_DATA") else "")
    if not r or "{" in r:
        raise ValueError(f"no data folder: give data_root (got {root!r}) or set RESEARCH_DATA")
    return os.path.abspath(os.path.expanduser(r))


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


def holdout_guard(split):
    from Evaluation.defaults import guard
    guard(split, ROOT)


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


def truth_index(root, signature):
    with open(os.path.join(data_root(root), "gt", signature, "index.json"), encoding="utf-8") as f:
        return json.load(f)


def load_split(split, root="", signature=None):
    """the frames of a split, shard after shard: {frames (N, S, S) float32, ann (N, 3), scene (N,) str, index (N,) int, masks (N, S, S) bool when a
    ground-truth signature is given}. A shard whose bytes are not the recorded ones is refused"""
    holdout_guard(split)
    parts = {"frames": [], "ann": [], "scene": [], "index": [], "masks": []}
    index = truth_index(root, signature)["shards"] if signature else {}
    for s in split_shards(split):
        p = raw_path(root, s["name"])
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p}: the raw shard is missing (step prepare: gt/generate.py make)")
        if file_hash(p) != s["sha256"]:
            raise ValueError(f"{p} is not the shard the manifest names (another content)")
        a = np.load(p, allow_pickle=False)
        parts["frames"].append(a["frame"])
        parts["ann"].append(a["ann"])
        parts["scene"].append(a["scene"])
        parts["index"].append(a["index"])
        if signature:
            t = truth_path(root, signature, s["name"])
            if not os.path.isfile(t) or s["name"] not in index:
                raise FileNotFoundError(f"{t}: the ground truth of signature {signature} is missing (step prepare: gt/generate.py make)")
            if file_hash(t) != index[s["name"]]["sha256"]:
                raise ValueError(f"{t} is not the ground truth its index names (another content)")
            parts["masks"].append(np.load(t, allow_pickle=False))
    return {k: np.concatenate(v) for k, v in parts.items() if v}
