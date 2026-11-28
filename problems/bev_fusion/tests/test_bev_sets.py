"""Tests of evaluation/bev_sets.py (its former self-test).

    python tests/test_bev_sets.py      (from the problem folder, with its module path)
"""
from __future__ import annotations
import os
import tempfile

from evaluation.bev_sets import PARTS, _splits, counts, lists, names, parse, part_of, set_hash, table, tag


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
        assert "tiny.val" in table(tmp) and "toy" in table(tmp) and counts(tmp) == {"tiny.train": 2, "tiny.val": 1} and counts()["small.val"] == 10
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


if __name__ == "__main__":
    selftest()
