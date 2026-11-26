"""The ground truth of the toy problem, in the pattern of GtGenerator: a store named by the signature (hash of the rasteriser and its options) with an index;
a second run writes only what is missing; the truth is made from the raw annotations, never edited by hand.

    bash tools/py.sh gt/generate.py generate [--config F]     write the truth of every shard of the manifest that is missing
    bash tools/py.sh gt/generate.py have     [--config F]     exit 0: the store of this signature is complete, 1: not
    bash tools/py.sh gt/generate.py key      [--config F]     the signature
    bash tools/py.sh gt/generate.py check    [--config F]     known answers, and every stored shard against a fresh one
    bash tools/py.sh gt/generate.py view --out picture.png    eight val frames with the truth outlined (needs matplotlib)
Exit codes: 0 done, 1 not there or incomplete, 2 wrong arguments.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

from gt import raster
from net.prep import shards as S

SUPERSAMPLE = 4


def signature(supersample=SUPERSAMPLE):
    with open(raster.__file__, "rb") as f:
        code = f.read()
    return hashlib.sha256(code + json.dumps({"supersample": int(supersample)}).encode()).hexdigest()[:12]


def _store(root, sig):
    return os.path.dirname(S.truth_path(root, sig, "x"))


def _write(path, writer):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        writer(f)
    os.replace(tmp, path)


def generate(root="", supersample=SUPERSAMPLE):
    """write what is missing -> {signature, written, shards}"""
    sig, man = signature(supersample), S.manifest()
    os.makedirs(_store(root, sig), exist_ok=True)
    written, entries = 0, {}
    for s in man["shards"]:
        dst = S.truth_path(root, sig, s["name"])
        if not os.path.isfile(dst):
            raw = np.load(S.raw_path(root, s["name"]), allow_pickle=False)
            mask = raster.rasterise(raw["ann"], man["recipe"]["size"], supersample)
            _write(dst, lambda f: np.save(f, mask, allow_pickle=False))
            written += 1
        entries[s["name"]] = {"sha256": S.file_hash(dst), "frames": s["frames"]}
    index = {"signature": sig, "supersample": int(supersample), "manifest_hash": S.manifest_hash(), "shards": entries, "complete": True}
    _write(os.path.join(_store(root, sig), "index.json"), lambda f: f.write((json.dumps(index, indent=1) + "\n").encode()))
    return {"signature": sig, "written": written, "shards": len(entries)}


def have(root="", supersample=SUPERSAMPLE):
    """(ok, why)"""
    sig = signature(supersample)
    try:
        with open(os.path.join(_store(root, sig), "index.json"), encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, ValueError):
        return False, f"no truth store for signature {sig}"
    if not index.get("complete") or index.get("manifest_hash") != S.manifest_hash():
        return False, f"the store {sig} is not complete for the data of this manifest"
    for s in S.manifest()["shards"]:
        p, e = S.truth_path(root, sig, s["name"]), index["shards"].get(s["name"])
        if not e or not os.path.isfile(p) or S.file_hash(p) != e["sha256"]:
            return False, f"{s['name']} of the store {sig} is missing or changed"
    return True, f"truth {sig} complete"


def check(root="", supersample=SUPERSAMPLE):
    """known answers of the rasteriser, then the store against a fresh rasterisation; raises AssertionError"""
    one = raster.rasterise([(16.5, 16.5, 0.5)], 32, supersample)[0]
    assert one.sum() == 1 and one[16, 16], "a disc of radius 0.5 on a pixel centre must be exactly that pixel"
    assert raster.rasterise([(16.0, 16.0, 100.0)], 32, supersample).all(), "a huge disc must cover everything"
    assert not raster.rasterise([(16.0, 16.0, 0.0)], 32, supersample).any(), "an empty disc must be empty"
    area = raster.rasterise([(15.3, 17.1, 6.0)], 32, supersample).sum()
    assert abs(area - np.pi * 36) < 0.05 * np.pi * 36, f"area {area} of a disc of radius 6"
    ok, why = have(root, supersample)
    assert ok, why
    sig = signature(supersample)
    for s in S.manifest()["shards"]:
        raw = np.load(S.raw_path(root, s["name"]), allow_pickle=False)
        assert np.array_equal(np.load(S.truth_path(root, sig, s["name"])), raster.rasterise(raw["ann"], 32, supersample)), f"{s['name']} differs from a fresh truth"
    return f"truth {sig}: known answers and {len(S.manifest()['shards'])} shards agree"


def view(out, root="", supersample=SUPERSAMPLE):
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
    ap.add_argument("what", choices=["generate", "have", "key", "check", "view"])
    ap.add_argument("--config", help="a training config: its data_root and gt_supersample are used")
    ap.add_argument("--root", default="")
    ap.add_argument("--supersample", type=int, default=0)
    ap.add_argument("--out", default="temp/truth.png")
    a = ap.parse_args(argv)
    cfg = {}
    if a.config:
        with open(a.config, encoding="utf-8") as f:
            cfg = json.load(f)
    root, ss = a.root or cfg.get("data_root", ""), a.supersample or cfg.get("gt_supersample", SUPERSAMPLE)
    if a.what == "generate":
        r = generate(root, ss)
        print(f"truth {r['signature']}: {r['shards']} shards, {r['written']} written")
    elif a.what == "have":
        ok, why = have(root, ss)
        print(why)
        return 0 if ok else 1
    elif a.what == "key":
        print(signature(ss))
    elif a.what == "check":
        print(check(root, ss))
    else:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        print(view(a.out, root, ss))
    return 0


if __name__ == "__main__":
    sys.exit(main())
