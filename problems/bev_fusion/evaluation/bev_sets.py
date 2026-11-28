"""The sets the evaluation runs on: one definition, the one of the training (`splits/<name>.json`, read by `net/prep/splits.py`). No second list of scenes.

    python -P -m bev_fusion.evaluation.bev_sets [--list]       the sets that exist (name, scenes, GT version, what it is)
Tests: tests/test_bev_sets.py (known answers on the real files and on a throw-away splits folder)

A set is `<split>.<part>`: the name of a training split (`small`, `large`, `xl`: the file the training config's `"split"` names) and its part, `train` or `val`
(`small.val` = the 10 val scenes of splits/small.json). The bare names `val` and `train` (the default `val`) are the lists the run itself was trained and
validated on (its `config.json`, as before): for a run trained on `large`, `val` and `large.val` are the same 82 scenes. A new set is a new split file in
`splits/` (and then also usable
by the training), never a list in the evaluation. The TEST split is not a set, and `test` and `<split>.test` are refused here: no BEV TEST evaluation is built
in v2 yet, so `research test bev_fusion` ends here (a TEST run needs this set, and the owner's row in TEST_LOG.md).

Where it is used: the set of an Evaluation (`Evaluation.run(RUN, "small.val")`, `spread:1000:small.val`; `val` / `train` as always); `bev_adapter.build(run,
split=<set>)` puts the set's lists into the run's config before the data is made, so the run's checkpoint and caches are the run's own and only the scenes
change. The set name and its hash
(`set_hash` of the Source's description) are part of the evaluation id, so evaluations of different sets are kept apart.
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PARTS = ("train", "val")
RESERVED = ("val", "train", "test")                      # the bare names and TEST: no split file may carry them (val.train would read like the bare names)


def _splits():
    """net/prep/splits.py: the one reader of the split files"""
    from bev_fusion.net.prep import splits
    return splits


def name_ok(split: str) -> bool:
    """a usable split name: lower-case letters, digits and _, not one of RESERVED, not starting with warm or ctl (the prefixes of the output tags: `_warm_...`,
    `_ctl_...`)"""
    return re.fullmatch(r"(?!warm|ctl)[a-z0-9_]+", str(split)) is not None and split not in RESERVED


def parse(name: str):
    """'val' -> (None, 'val') the run's own lists; 'small.val' -> ('small', 'val'); ValueError for anything else (TEST included)"""
    if name in PARTS:
        return None, name
    split, dot, part = str(name).partition(".")
    if name == "test" or (dot and part == "test" and re.fullmatch(r"[a-z0-9_]+", split)) or (dot and part in PARTS and split == "test"):
        raise ValueError(f"set {name!r}: the TEST split is not offered (no BEV TEST evaluation is built yet)")
    if not (dot and part in PARTS and name_ok(split)):
        raise ValueError(f"set {name!r}: val | train (the run's own lists) or <split>.<part> with part train | val, e.g. small.val (`evaluation/bev_sets.py --list`)")
    return split, part


def part_of(name: str) -> str:
    """'train' or 'val': the part of a set (the nuScenes official split the scenes are checked against)"""
    return parse(name)[1]


def tag(name: str) -> str:
    """the suffix of the output files of a set: '' for val, '_train' for train (as before), '_small_val' for small.val"""
    split, part = parse(name)
    return ("_train" if part == "train" else "") if split is None else f"_{split}_{part}"


def lists(name: str, own_train, own_val, splits_dir: str = None):
    """(train scenes, val scenes) to evaluate with: the run's own for a bare name, the lists of the split's file for `<split>.<part>` (both parts, so the
    prior-map baseline and the dataset see one consistent split). ValueError for a split that has no file"""
    split, _ = parse(name)
    if split is None:
        return list(own_train), list(own_val)
    d = _splits().load(split, *([splits_dir] if splits_dir else []))
    return list(d["train"]), list(d["val"])


def names(splits_dir: str = None) -> list:
    """every set the splits folder defines: ['large.train', 'large.val', 'small.train', ...]"""
    S = _splits()
    folder = splits_dir or S.SPLITS
    out = []
    for f in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if f.endswith(".json"):
            if not name_ok(f[:-5]):
                raise ValueError(f"{os.path.join(folder, f)}: a split cannot be named {f[:-5]!r} (lower-case letters, digits, _; not val, train, test; not starting with warm or ctl)")
            S.load(f[:-5], *([splits_dir] if splits_dir else []))          # a broken file raises here, with its name
            out += [f"{f[:-5]}.{p}" for p in PARTS]
    return out


def counts(splits_dir: str = None) -> dict:
    """{set: number of scenes} of every set the splits folder defines"""
    S = _splits()
    out = {}
    for n in names(splits_dir):
        split, part = n.split(".")
        out[n] = len(S.load(split, *([splits_dir] if splits_dir else []))[part])
    return out


def set_hash(name: str, splits_dir: str = None) -> str:
    """12 hex digits of the scene names of a NAMED set (net/prep/splits.py list_hash of the file's part): the identity of the set, so a split file edited under
    the same name is seen"""
    split, part = parse(name)
    if split is None:
        raise ValueError(f"set {name!r} is the run's own list: it has no file to hash")
    S = _splits()
    return S.list_hash(S.load(split, *([splits_dir] if splits_dir else []))[part])


def table(splits_dir: str = None) -> str:
    S = _splits()
    lines = ["set              scenes  gt  what", "val / train      the run's own lists (config.json of the run)"]
    for n in names(splits_dir):
        split, part = n.split(".")
        d = S.load(split, *([splits_dir] if splits_dir else []))
        lines.append(f"{n:16s} {len(d[part]):6d}  {d.get('gt', '?')}  {d.get('about', '')[:90]}")
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv in ([], ["--list"]):
        print(table())
        return 0
    print(__doc__.split("Where it is used")[0], file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
