"""The sets the evaluation runs on: one definition, the one of the training (`splits/<name>.json`, read by `training/splits.py`). No second list of scenes.

    python evaluation/bev_sets.py [--list]       the sets that exist (name, scenes, GT version, what it is)
    python evaluation/bev_sets.py --selftest     known answers on the real files and on a throw-away splits folder

A set is `<split>.<part>`: the name of a training split (`small`, `large`, `xl`: the file the training config's `"split"` names) and its part, `train` or `val`
(`small.val` = the 10 val scenes of splits/small.json). The bare names `val` and `train` (the default `val`) are the lists the run itself was trained and validated on
(its `config.json`, as before): for a run trained on `large`, `val` and `large.val` are the same 82 scenes. A new set is a new split file in `splits/` (and then also usable
by the training), never a list in the evaluation. The TEST split is not a set: it needs a logged row under Mirko's recorded OK (`run.py --split test`), so `test` and
`<split>.test` are refused here.

Where it is used: `evaluation/evaluate.py RUN --split small.val ...` and `bev_suite.py RUN --split small.val <command>` (`--split` takes a set name; `val` / `train` as always),
`bev_adapter.build(run, split=<set>)` puts the set's lists into the run's config before the data is made, so the run's checkpoint and caches are the run's own and only the
scenes change. Outputs of a named set carry its name (`eval/card_small_val.json`, `eval/suite/quant/eval_small_val_n80`); the card says `"split": "small.val"`, its `provenance.data` holds the
set, the scenes and the frame hash, so `decide.py` (a card's split must be the one the experiment declares, the frame hashes must agree) and `bev_suite.py compare --split
small.val` (the store of that name on both runs) keep sets apart.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PARTS = ("train", "val")
RESERVED = ("val", "train", "test")                      # the bare names and TEST: no split file may carry them (val.train would read like the bare names)


def _splits():
    """training/splits.py, loaded by its path (a plain `import splits` would meet the folder bev_fusion/splits first)"""
    spec = importlib.util.spec_from_file_location("bev_training_splits", os.path.join(ROOT, "training", "splits.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def name_ok(split: str) -> bool:
    """a usable split name: lower-case letters, digits and _, not one of RESERVED, not starting with warm or ctl (the prefixes of the output tags: `_warm_...`, `_ctl_...`)"""
    return re.fullmatch(r"(?!warm|ctl)[a-z0-9_]+", str(split)) is not None and split not in RESERVED


def parse(name: str):
    """'val' -> (None, 'val') the run's own lists; 'small.val' -> ('small', 'val'); ValueError for anything else (TEST included)"""
    if name in PARTS:
        return None, name
    split, dot, part = str(name).partition(".")
    if name == "test" or (dot and part == "test" and re.fullmatch(r"[a-z0-9_]+", split)) or (dot and part in PARTS and split == "test"):
        raise ValueError(f"set {name!r}: the TEST split is not offered (a logged row under a recorded decision, then `run.py --split test`)")
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
    """(train scenes, val scenes) to evaluate with: the run's own for a bare name, the lists of the split's file for `<split>.<part>` (both parts, so the prior-map baseline
    and the dataset see one consistent split). ValueError for a split that has no file"""
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


def set_hash(name: str, splits_dir: str = None) -> str:
    """12 hex digits of the scene names of a NAMED set (training/splits.py list_hash of the file's part): the identity of the set, so a split file edited under the same name is seen"""
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


def selftest():
    S = _splits()
    # 1 names: every file of splits/ gives its two parts, nothing else
    have = names()
    files = sorted(f[:-5] for f in os.listdir(S.SPLITS) if f.endswith(".json"))
    assert have == [f"{f}.{p}" for f in files for p in PARTS] and "small.val" in have and "large.train" in have, have
    # 2 parse: the grammar, TEST and everything odd refused
    assert parse("val") == (None, "val") and parse("train") == (None, "train") and parse("small.val") == ("small", "val") and parse("xl.train") == ("xl", "train")
    bads = ("test", "small.test", "large.test", "test.val", "", "small", "small.", ".val", "Small.val", "small.VAL", "small.val.x", "a/b.val", "..val", "small val", "val.train",
            "train.val", "train.train", "warm_x.val", "ctl_x.train", None, 3)
    for bad in bads:
        try:
            parse(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} was accepted")
    for t in ("small.test", "test", "test.val"):
        try:
            parse(t)
        except ValueError as e:
            assert "TEST" in str(e), (t, e)
    # 3 tag: the old names unchanged, a named set carries its name, and two sets never share a tag
    assert tag("val") == "" and tag("train") == "_train" and tag("small.val") == "_small_val" and tag("small.train") == "_small_train"
    assert len({tag(n) for n in ["val", "train"] + have}) == 2 + len(have)
    assert part_of("small.val") == "val" and part_of("train") == "train" and part_of("xl.train") == "train"
    # 4 lists: bare = the run's own (a copy), named = the file's, the two parts disjoint, small inside large
    assert lists("val", ["a"], ["b"]) == (["a"], ["b"]) and lists("train", [], []) == ([], [])
    sm_t, sm_v = lists("small.val", ["own"], ["own"])
    lg_t, lg_v = lists("large.val", ["own"], ["own"])
    assert (len(sm_t), len(sm_v), len(lg_t), len(lg_v)) == (60, 10, 428, 82) and not set(sm_t) & set(sm_v) and set(sm_v) <= set(lg_v) and set(sm_t) <= set(lg_t), "the sizes of splits/README.md"
    assert lists("small.train", [], [])[1] == sm_v and lists("small.val", [], [])[0] == sm_t, "both parts of the split come with either name"
    try:
        lists("nosuch.val", [], [])
        raise AssertionError("an unknown split was accepted")
    except ValueError as e:
        assert "no split 'nosuch'" in str(e) and "small" in str(e), e
    # 5 a throw-away folder: a new set is a new file, and names / lists / table follow it (no list lives in the evaluation)
    with tempfile.TemporaryDirectory() as tmp:
        import json
        with open(os.path.join(tmp, "tiny.json"), "w") as f:
            json.dump({"about": "toy", "gt": "v8", "train": ["s1", "s2"], "val": ["s3"]}, f)
        assert names(tmp) == ["tiny.train", "tiny.val"] and lists("tiny.val", [], [], tmp) == (["s1", "s2"], ["s3"])
        assert "tiny.val" in table(tmp) and "toy" in table(tmp)
        assert set_hash("tiny.val", tmp) == S.list_hash(["s3"]) != set_hash("tiny.train", tmp) and set_hash("small.val") == S.list_hash(sm_v)
        try:
            set_hash("val")
            raise AssertionError("the run's own list has no file")
        except ValueError:
            pass
        for reserved in ("val", "test", "warm_x"):                           # a split file with a reserved name breaks `names` loudly
            with open(os.path.join(tmp, reserved + ".json"), "w") as f:
                json.dump({"about": "x", "gt": "v8", "train": ["a"], "val": ["b"]}, f)
            try:
                names(tmp)
                raise AssertionError(f"a split file named {reserved} was accepted")
            except ValueError as e:
                assert "cannot be named" in str(e), e
            os.remove(os.path.join(tmp, reserved + ".json"))
        with open(os.path.join(tmp, "bad.json"), "w") as f:
            json.dump({"train": ["x"], "val": ["x"]}, f)                      # a scene in both parts
        try:
            names(tmp)
            raise AssertionError("a broken split file was accepted")
        except ValueError as e:
            assert "bad.json" in str(e), e
    print(f"bev_sets selftest ok: {len(have)} sets from {len(files)} split files, the grammar (TEST and {len(bads) - 1} malformed names refused), tags, lists (small 60 / 10 inside large 428 / 82), a toy folder")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--selftest"]:
        selftest()
        return 0
    if argv in ([], ["--list"]):
        print(table())
        return 0
    print(__doc__.split("Where it is used")[0], file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
