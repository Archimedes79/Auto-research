"""The ground truth of the toy problem, in the pattern of GtGenerator: a store named by the signature (the rasteriser's code, its options and the
hash of the raw data's manifest) with an index of shard hashes; a second run writes only what is missing, and a stored shard that differs from
a fresh rasterisation is refused, never re-recorded. The ground truth is made from the raw annotations, never edited by hand.

    python gt/generate.py make     --config F     raw shards (dataset/make_data.py) and the ground truth: what is missing is written, all is checked
    python gt/generate.py have     --config F     exit 0: raw data and the store of this signature are complete, 1: not
    python gt/generate.py key      [--config F]   the signature
    python gt/generate.py check    --config F     known answers, and every stored shard against a fresh one
    python gt/generate.py view     --config F --out picture.png      eight val frames with the ground truth outlined (needs matplotlib)
Exit codes: 0 done, 1 not there or incomplete, 2 wrong arguments or a stored file that is not what it must be.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

from dataset import make_data
from gt import raster
from net.prep import shards as S

SUPERSAMPLE = 4


class Differs(Exception):
    """a stored file that is not what the recipe makes"""


def signature(supersample=SUPERSAMPLE):
    with open(raster.__file__, "rb") as f:
        code = f.read()
    return hashlib.sha256(code + json.dumps({"supersample": int(supersample), "data": S.manifest_hash()}).encode()).hexdigest()[:12]


def _store(root, sig):
    return os.path.dirname(S.truth_path(root, sig, "x"))


def _write(path, data):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _npy(mask):
    import io
    buf = io.BytesIO()
    np.save(buf, mask, allow_pickle=False)
    return buf.getvalue()


def generate(root, supersample=SUPERSAMPLE):
    """write what is missing, refuse what differs -> {signature, written, shards}"""
    sig, man = signature(supersample), S.manifest()
    os.makedirs(_store(root, sig), exist_ok=True)
    written, entries = 0, {}
    for s in man["shards"]:
        raw_path = S.raw_path(root, s["name"])
        if S.file_hash(raw_path) != s["sha256"]:
            raise Differs(f"{raw_path} is not the raw shard the manifest names")
        raw = np.load(raw_path, allow_pickle=False)
        data = _npy(raster.rasterise(raw["ann"], man["recipe"]["size"], supersample))
        dst = S.truth_path(root, sig, s["name"])
        if os.path.isfile(dst):
            with open(dst, "rb") as f:
                if f.read() != data:
                    raise Differs(f"{dst} differs from a fresh rasterisation: it is not overwritten, move it away")
        else:
            _write(dst, data)
            written += 1
        entries[s["name"]] = {"sha256": hashlib.sha256(data).hexdigest(), "frames": s["frames"]}
    index = {"signature": sig, "supersample": int(supersample), "manifest_hash": S.manifest_hash(), "shards": entries, "complete": True}
    _write(os.path.join(_store(root, sig), "index.json"), (json.dumps(index, indent=1) + "\n").encode())
    return {"signature": sig, "written": written, "shards": len(entries)}


def have(root, supersample=SUPERSAMPLE):
    """(ok, why)"""
    ok, why = make_data.have(root)
    if not ok:
        return False, f"raw data: {why}"
    sig = signature(supersample)
    try:
        index = S.truth_index(root, sig)
    except (OSError, ValueError):
        return False, f"no ground-truth store for signature {sig}"
    if not index.get("complete") or index.get("manifest_hash") != S.manifest_hash():
        return False, f"the store {sig} is not complete for the data of this manifest"
    for s in S.manifest()["shards"]:
        p, e = S.truth_path(root, sig, s["name"]), index["shards"].get(s["name"])
        if not e or not os.path.isfile(p) or S.file_hash(p) != e["sha256"]:
            return False, f"{s['name']} of the store {sig} is missing or changed"
    return True, f"raw data and ground truth {sig} complete"


def check(root, supersample=SUPERSAMPLE):
    """known answers of the rasteriser (off-centre and asymmetric, so a transposed mask fails), then the store against a fresh rasterisation"""
    one = raster.rasterise([(16.5, 16.5, 0.5)], 32, supersample)[0]
    assert one.sum() == 1 and one[16, 16], "a disc of radius 0.5 on a pixel centre must be exactly that pixel"
    off = raster.rasterise([(5.5, 20.5, 0.5)], 32, supersample)[0]       # cx 5.5 = column 5, cy 20.5 = row 20
    assert off.sum() == 1 and off[20, 5] and not off[5, 20], "x is the column and y the row (a transposed mask fails here)"
    assert raster.rasterise([(16.0, 16.0, 100.0)], 32, supersample).all(), "a huge disc must cover everything"
    assert not raster.rasterise([(16.0, 16.0, 0.0)], 32, supersample).any(), "an empty disc must be empty"
    area = raster.rasterise([(15.3, 17.1, 6.0)], 32, supersample).sum()
    assert abs(area - np.pi * 36) < 0.05 * np.pi * 36, f"area {area} of a disc of radius 6"
    ok, why = have(root, supersample)
    assert ok, why
    sig = signature(supersample)
    for s in S.manifest()["shards"]:
        raw = np.load(S.raw_path(root, s["name"]), allow_pickle=False)
        assert np.array_equal(np.load(S.truth_path(root, sig, s["name"])), raster.rasterise(raw["ann"], 32, supersample)), f"{s['name']} differs from a fresh one"
    return f"ground truth {sig}: known answers and {len(S.manifest()['shards'])} shards agree"


def view(out, root, supersample=SUPERSAMPLE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = S.load_split("val", root, signature(supersample))
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for ax, k in zip(axes.ravel(), range(0, 32, 4)):
        ax.imshow(data["frames"][k], cmap="gray")
        ax.contour(data["masks"][k], levels=[0.5], colors="lime")
        ax.set_title(f"{data['scene'][k]}/{data['index'][k]}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["make", "have", "key", "check", "view"])
    ap.add_argument("--config", help="a run's config: its data_root and gt_supersample are used")
    ap.add_argument("--root", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    cfg = {}
    if a.config:
        with open(a.config, encoding="utf-8") as f:
            cfg = json.load(f)
    root, ss = a.root or cfg.get("data_root", ""), cfg.get("gt_supersample", SUPERSAMPLE)
    try:
        if a.what == "make":
            code = make_data.make(root)
            if code:
                return code
            r = generate(root, ss)
            print(f"ground truth {r['signature']}: {r['shards']} shards, {r['written']} written")
        elif a.what == "have":
            ok, why = have(root, ss)
            print(why)
            return 0 if ok else 1
        elif a.what == "key":
            print(signature(ss))
        elif a.what == "check":
            print(check(root, ss))
        else:
            if not a.out:
                ap.error("view needs --out")
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            print(view(a.out, root, ss))
    except Differs as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
