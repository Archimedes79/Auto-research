"""The raw data of the toy problem: made by code from a seed, so any machine makes the same bytes and nothing is downloaded.

A scene is a disc that drifts through a 32 x 32 picture for a few frames. A raw shard (one .npy, a structured array) holds per frame the noisy picture, the
annotation of the disc (cx, cy, r), the scene name and the frame index. The manifest (dataset/manifest.json, a text record in git) names every shard with its
split, size and sha256; the shards are derived data (<data root>/toy/raw/, not in git). The step `prepare` (gt/generate.py make) calls this first.

    python dataset/make_data.py make --config F     write the shards that are missing and compare every one with the manifest
    python dataset/make_data.py have --config F     exit 0: every shard of the manifest is there with its hash, 1: not
    python dataset/make_data.py manifest            write the manifest from the recipe below (only after a change of the recipe)
Exit codes: 0 done, 1 not there, 2 wrong arguments or another content than the manifest says (nothing is overwritten).
"""
import argparse
import hashlib
import io
import json
import os
import sys

import numpy as np

from net.prep import shards as S

RECIPE = {"dataset": "toy-disc", "seed": 7, "size": 32, "frames_per_scene": 4, "noise": 0.7, "splits": {"train": {"shards": 16, "scenes": 10}, "val": {"shards": 4, "scenes": 10}}}


def scene(recipe, split_no, k):
    """(annotations (T, 3), pictures (T, S, S)) of scene k of a split: its own random stream, so scenes do not depend on each other"""
    rng = np.random.default_rng([recipe["seed"], split_no, k])
    size, frames = recipe["size"], recipe["frames_per_scene"]
    r = rng.uniform(3.0, 7.0)
    start, step = rng.uniform(8.0, size - 8.0, 2), rng.uniform(-2.0, 2.0, 2)
    centre = np.arange(size) + 0.5
    ann, pics = [], []
    for t in range(frames):
        cx, cy = start + step * t
        disc = ((centre[None, :] - cx) ** 2 + (centre[:, None] - cy) ** 2) <= r * r
        ann.append((cx, cy, r))
        pics.append(disc + rng.normal(0.0, recipe["noise"], (size, size)))
    return np.asarray(ann, np.float32), np.asarray(pics, np.float32)


def shard_bytes(recipe, split, number):
    """the .npy file of one shard as bytes (the same recipe, the same bytes)"""
    size, per, frames = recipe["size"], recipe["splits"][split]["scenes"], recipe["frames_per_scene"]
    split_no = sorted(recipe["splits"]).index(split)
    dtype = np.dtype([("scene", "U12"), ("index", "i4"), ("ann", "f4", (3,)), ("frame", "f4", (size, size))])
    arr = np.zeros(per * frames, dtype)
    for j in range(per):
        k = number * per + j
        ann, pics = scene(recipe, split_no, k)
        rows = slice(j * frames, (j + 1) * frames)
        arr["scene"][rows], arr["index"][rows] = f"{split}_{k:03d}", np.arange(frames)
        arr["ann"][rows], arr["frame"][rows] = ann, pics
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def build(recipe):
    """[(manifest entry, bytes)] of every shard of a recipe"""
    out = []
    for split, spec in recipe["splits"].items():
        for n in range(spec["shards"]):
            b = shard_bytes(recipe, split, n)
            out.append(({"name": f"{split}_{n:03d}", "split": split, "scenes": spec["scenes"], "frames": spec["scenes"] * recipe["frames_per_scene"], "bytes": len(b),
                         "sha256": hashlib.sha256(b).hexdigest()}, b))
    return out


def have(root):
    """(ok, why): every shard of the manifest is there with its hash"""
    for s in S.manifest()["shards"]:
        p = S.raw_path(root, s["name"])
        if not os.path.isfile(p):
            return False, f"{s['name']} is missing"
        if S.file_hash(p) != s["sha256"]:
            return False, f"{s['name']} has another content than the manifest says"
    return True, "every shard is there"


def make(root):
    """write the missing shards; -> exit code"""
    stored = {s["name"]: s for s in S.manifest()["shards"]}
    built = build(S.manifest()["recipe"])
    if sorted(stored) != sorted(e["name"] for e, _ in built):
        print("the recipe of the manifest does not make the shards the manifest names", file=sys.stderr)
        return 2
    new = 0
    for entry, b in built:
        if entry["sha256"] != stored[entry["name"]]["sha256"]:
            print(f"{entry['name']}: this machine makes other bytes than the manifest says (another numpy?): nothing was written for it", file=sys.stderr)
            return 2
        p = S.raw_path(root, entry["name"])
        if os.path.isfile(p):
            if S.file_hash(p) != entry["sha256"]:
                print(f"{p} exists with another content: it is not overwritten, move it away", file=sys.stderr)
                return 2
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p + f".tmp{os.getpid()}", "wb") as f:
            f.write(b)
        os.replace(p + f".tmp{os.getpid()}", p)
        new += 1
    print(f"raw shards: {len(built)} in {S.data_root(root)}/raw, {new} written")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["make", "have", "manifest"])
    ap.add_argument("--config", help="a training config: its data_root is used")
    ap.add_argument("--root", default="", help="the data folder (default: the config's data_root, else RESEARCH_DATA/toy)")
    a = ap.parse_args(argv)
    root = a.root
    if a.config and not root:
        with open(a.config, encoding="utf-8") as f:
            root = json.load(f).get("data_root", "")
    if a.what == "manifest":
        man = {"_about": "the raw data of the toy problem, named by hashes (dataset/make_data.py makes the shards from this recipe)", "recipe": RECIPE,
               "shards": [e for e, _ in build(RECIPE)]}
        with open(S.MANIFEST, "w", encoding="utf-8", newline="\n") as f:
            json.dump(man, f, indent=1)
            f.write("\n")
        print(f"manifest written: {len(man['shards'])} shards, hash {S.manifest_hash()}")
        return 0
    if a.what == "have":
        ok, why = have(root)
        print(why)
        return 0 if ok else 1
    return make(root)


if __name__ == "__main__":
    sys.exit(main())
