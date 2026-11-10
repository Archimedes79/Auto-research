def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


# ---- bev_sets: reserved split names, set_hash, a strict command line
patch("evaluation/bev_sets.py", [
    ('''PARTS = ("train", "val")
''', '''PARTS = ("train", "val")
RESERVED = ("val", "train", "test")                      # the bare names and TEST: no split file may carry them (val.train would read like the bare names)
'''),
    ('''def parse(name: str):
    """'val' -> (None, 'val') the run's own lists; 'small.val' -> ('small', 'val'); ValueError for anything else (TEST included)"""
    if name in PARTS:
        return None, name
    split, dot, part = str(name).partition(".")
    if re.fullmatch(r"[a-z0-9_]+", split) and dot and part == "test" or name == "test":
        raise ValueError(f"set {name!r}: the TEST split is not offered (a logged row under a recorded decision, then `run.py --split test`)")
    if not (re.fullmatch(r"[a-z0-9_]+", split) and dot and part in PARTS):
        raise ValueError(f"set {name!r}: val | train (the run's own lists) or <split>.<part> with part train | val, e.g. small.val (`evaluation/bev_sets.py --list`)")
    return split, part
''', '''def name_ok(split: str) -> bool:
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
'''),
    ('''        if f.endswith(".json"):
            S.load(f[:-5], *([splits_dir] if splits_dir else []))          # a broken file raises here, with its name''',
     '''        if f.endswith(".json"):
            if not name_ok(f[:-5]):
                raise ValueError(f"{os.path.join(folder, f)}: a split cannot be named {f[:-5]!r} (lower-case letters, digits, _; not val, train, test; not starting with warm or ctl)")
            S.load(f[:-5], *([splits_dir] if splits_dir else []))          # a broken file raises here, with its name'''),
    ('''def table(splits_dir: str = None) -> str:''', '''def set_hash(name: str, splits_dir: str = None) -> str:
    """12 hex digits of the scene names of a NAMED set (training/splits.py list_hash of the file's part): the identity of the set, so a split file edited under the same name is seen"""
    split, part = parse(name)
    if split is None:
        raise ValueError(f"set {name!r} is the run's own list: it has no file to hash")
    S = _splits()
    return S.list_hash(S.load(split, *([splits_dir] if splits_dir else []))[part])


def table(splits_dir: str = None) -> str:'''),
    ('''    bads = ("test", "small.test", "large.test", "", "small", "small.", ".val", "Small.val", "small.VAL", "small.val.x", "a/b.val", "..val", "small val", None, 3)''',
     '''    bads = ("test", "small.test", "large.test", "test.val", "", "small", "small.", ".val", "Small.val", "small.VAL", "small.val.x", "a/b.val", "..val", "small val", "val.train",
            "train.val", "train.train", "warm_x.val", "ctl_x.train", None, 3)'''),
    ('''    try:
        parse("small.test")
    except ValueError as e:
        assert "TEST" in str(e), e''', '''    for t in ("small.test", "test", "test.val"):
        try:
            parse(t)
        except ValueError as e:
            assert "TEST" in str(e), (t, e)'''),
    ('''        assert "tiny.val" in table(tmp) and "toy" in table(tmp)''', '''        assert "tiny.val" in table(tmp) and "toy" in table(tmp)
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
            os.remove(os.path.join(tmp, reserved + ".json"))'''),
    ('''    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        selftest()
        return 0
    print(table())
    return 0''', '''    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--selftest"]:
        selftest()
        return 0
    if argv in ([], ["--list"]):
        print(table())
        return 0
    print(__doc__.split("Where it is used")[0], file=sys.stderr)
    return 2'''),
])

# ---- bev_adapter: the set's hash in the source identity (named sets only), no silent drop of a set's scenes
patch("evaluation/bev_adapter.py", [
    ('''        def __init__(self, cfg, ds, ns, workers: int = 2, control: str = "", split: str = "val", evaluator: Optional[str] = None):
            self.cfg, self.ds, self.ns, self.workers, self.control, self.split = cfg, ds, ns, workers, control, split''',
     '''        def __init__(self, cfg, ds, ns, workers: int = 2, control: str = "", split: str = "val", evaluator: Optional[str] = None, set_hash: Optional[str] = None):
            self.cfg, self.ds, self.ns, self.workers, self.control, self.split = cfg, ds, ns, workers, control, split
            self.set_hash = set_hash                                      # a named set (small.val): the hash of its scene list, part of the store identity (val / train: None, nothing new)'''),
    ('''"gt_signature": sig, "gt_version": gt_version(sig), "evaluator_hash": self.evaluator}''',
     '''"gt_signature": sig, "gt_version": gt_version(sig), "evaluator_hash": self.evaluator,
                    **({"set_hash": self.set_hash} if self.set_hash else {})}'''),
    ('''    source = _make_source_class()(cfg, ds, ns, workers=workers, control=control, split=split,
                                      evaluator=evaluator or evaluator_hash())''',
     '''    named = bev_sets.parse(split)[0] is not None
    source = _make_source_class()(cfg, ds, ns, workers=workers, control=control, split=split,
                                      evaluator=evaluator or evaluator_hash(), set_hash=bev_sets.set_hash(split) if named else None)
    if named:                                                      # make_datasets drops scenes that are not in the official part or have no front-camera files, silently: not for a set
        want = bev_sets.lists(split, [], [])[0 if bev_sets.part_of(split) == "train" else 1]
        gone = sorted(set(want) - set(source.scenes()))
        if gone:
            raise SystemExit(f"set {split}: {len(gone)} of its {len(want)} scenes cannot be read by this run's data ({', '.join(gone[:5])}{', ...' if len(gone) > 5 else ''}): "
                             f"not in the official {bev_sets.part_of(split)} split of nuScenes, or no front-camera files; its features / GT caches may still be missing too")'''),
])

# ---- bev_suite: one guard for the run's own commands, outputs of a named set in folders of its own, the prior store, no per-command guards
patch("evaluation/bev_suite.py", [
    ('''    if bev_sets.parse(a.split)[0] is not None:
        print(f"report: the page of a run is made from its own evaluation (--split val | train), not from the set {a.split}")
        return 2
    p = argparse.ArgumentParser(prog="report", allow_abbrev=False)''', '''    p = argparse.ArgumentParser(prog="report", allow_abbrev=False)'''),
    ('''    if bev_sets.parse(a.split)[0] is not None:
        print(f"generalisation: the run's own val and train lists (--split val), not the set {a.split}")
        return 2
    p = argparse.ArgumentParser(prog="generalisation", allow_abbrev=False)''', '''    p = argparse.ArgumentParser(prog="generalisation", allow_abbrev=False)'''),
    ('''    if bev_sets.parse(a.split)[0] is not None:
        print(f"memory: writes eval/memory.json of the run's own val frames (--split val), not of the set {a.split}")
        return 2
    p = argparse.ArgumentParser(prog="memory")''', '''    p = argparse.ArgumentParser(prog="memory")'''),
    ('''    try:
        bev_sets.parse(a.split)
    except ValueError as ex:
        p.error(str(ex))
    a.command = command''', '''    try:
        named = bev_sets.parse(a.split)[0] is not None
    except ValueError as ex:
        p.error(str(ex))
    a.command = command
    if named and a.command and a.command[0] in OWN_COMMANDS:
        p.error(f"`{a.command[0]}` is about the run's own evaluation (--split val | train), not about the set {a.split}")
    if named and a.out is None and a.command and a.command[0] != "eval":     # pictures, videos, stores, the module probe: a folder of the set's own, never the run's plain ones
        a.out = os.path.join(a.run, "eval", ("modules" if a.command[0] == "modules" else "suite") + bev_sets.tag(a.split))'''),
    ('''def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)''', '''OWN_COMMANDS = ("memory", "report", "generalisation", "time", "timing")       # write the run's plain files (memory.json, report/, latency.json, ...): not on a named set


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)'''),
    ('''    stores = [os.path.join(quant_dir, name)] + ([os.path.join(quant_dir, f"prior{e.prior}_n{e.n or 'all'}")] if e.prior else [])''',
     '''    ptag = bev_sets.tag(a.split) if bev_sets.parse(a.split)[0] is not None else ""     # the prior baseline of another set is another store (its train frames are the set's)
    stores = [os.path.join(quant_dir, name)] + ([os.path.join(quant_dir, f"prior{ptag}{e.prior}_n{e.n or 'all'}")] if e.prior else [])'''),
    ('''                                  name=f"prior{e.prior}_n{e.n or 'all'}", progress=50)''', '''                                  name=f"prior{ptag}{e.prior}_n{e.n or 'all'}", progress=50)'''),
])

# ---- evaluate: --time is the run's, the in-process frames go to the set's folder
patch("evaluation/evaluate.py", [
    ('''    if a.all and a.split != "val":''', '''    if a.time and a.split != "val":
        raise SystemExit(f"--time measures the run, not a set of scenes: leave out --split {a.split}")
    if a.all and a.split != "val":'''),
    ('''--quant (with --curves / --fusion / --velocity), --qual, --videos and
--time. The TEST split is not offered.''', '''--quant (with --curves / --fusion / --velocity), --qual and --videos (pictures and videos into eval/suite_<split>_<part>/, the numbers' stores into eval/suite/quant/
under the set's name); --time is the run's. The TEST split is not offered.'''),
    ('''        suite = bev_suite.make_suite(argparse.Namespace(run=a.run, device=a.device, workers=a.workers, split=a.split, ckpt_dir=a.ckpt_dir, control="", out=a.out, state="carry"))''',
     '''        out = a.out or (os.path.join(a.run, "eval", "suite" + bev_sets.tag(a.split)) if bev_sets.parse(a.split)[0] is not None else None)      # a named set: a folder of its own
        suite = bev_suite.make_suite(argparse.Namespace(run=a.run, device=a.device, workers=a.workers, split=a.split, ckpt_dir=a.ckpt_dir, control="", out=out, state="carry"))'''),
])

# ---- tests
patch("tests/test_evaluate_door.py", [
    ('''and refused("--all", "--split", "small.val")
''', '''and refused("--all", "--split", "small.val")
    assert refused("--time", "--split", "small.val") and refused("--time", "--split", "train") and not refused("--time"), "--time is the run's"
    assert all(refused("--quant", "--split", x) for x in ("test.val", "val.train", "warm_x.val", "ctl_x.train")), "reserved split names"
'''),
    ('''    if not glob.glob(os.path.join(run, "ckpt_*.pt")) or not os.path.isfile(os.path.join(run, "config.json")):
        print(f"SKIP sets check: {run} has no checkpoint here")
        return False
    sys.path.insert(0, os.path.join(ROOT, "training"))''', '''    assert glob.glob(os.path.join(run, "ckpt_*.pt")) and os.path.isfile(os.path.join(run, "config.json")), f"{run} has no checkpoint here: the check was asked for and cannot run"
    sys.path.insert(0, os.path.join(ROOT, "training"))'''),
    ('''    if split is None:
        print(f"SKIP sets check: {run} was not trained on a split of splits/ (its lists are inline)")
        return False''', '''    assert split is not None, f"{run} was not trained on a split of splits/ (its lists are inline and equal none of {sp}): pick another run"'''),
    ('''            assert used and used <= want and card["split"] == other, (sorted(used), card["split"])''',
     '''            assert used and used <= want and card["split"] == other, (sorted(used), card["split"])
            assert card["provenance"]["data"]["use"]["set"] == other and json.load(open(os.path.join(ROOT, folder, "suite", "quant", "eval_small_val_n10", "manifest.json")))["source"]["set_hash"] == bev_sets.set_hash(other), "the set's hash is in the store identity"
            assert os.path.isdir(os.path.join(ROOT, jobs["named_val"][0], "suite", "quant")) and not os.path.exists(os.path.join(ROOT, jobs["val"][0], "eval", "card_small_val.json"))'''),
])
patch("evaluation/bev_selftest.py", [
    ('''    assert bev_suite.tag_of(True, "train", "zero_radar") == "_warm_train_ctl_zero_radar" and bev_suite.tag_of(False, "val", "") == "" and bev_suite.tag_of(False, "small.val", "") == "_small_val"
''', '''    assert bev_suite.tag_of(True, "train", "zero_radar") == "_warm_train_ctl_zero_radar" and bev_suite.tag_of(False, "val", "") == "" and bev_suite.tag_of(False, "small.val", "") == "_small_val"
    tags = {bev_suite.tag_of(w, s, c) for w in (False, True) for s in ("val", "train", "small.val", "small.train", "large.train", "xl.val") for c in ("", "zero_radar")}
    assert len(tags) == 2 * 6 * 2, "warm / set / control never collide"
    import contextlib
    import io
    from evaluation import bev_sets
    for cmd in bev_suite.OWN_COMMANDS:                                  # the commands that write the run's plain files refuse a named set, before anything is built
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                bev_suite.main(["runs/none", "--split", "small.val", cmd])
            raise AssertionError(cmd)
        except SystemExit as ex:
            assert ex.code == 2, (cmd, ex.code)
    for bad in ("small.test", "test", "val.train"):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                bev_suite.main(["runs/none", "--split", bad, "list"])
            raise AssertionError(bad)
        except SystemExit as ex:
            assert ex.code == 2, (bad, ex.code)
    try:                                                                # compare --split names the store of the set (no runs here: the message says which store it looked for)
        bev_suite.cmd_compare(["runs/none_a", "runs/none_b", "--split", "small.val", "--n", "80"])
        raise AssertionError("compare found a store that does not exist")
    except SystemExit as ex:
        assert "eval_small_val_n80" in str(ex), ex
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            bev_suite.cmd_compare(["runs/none_a", "runs/none_b", "--split", "small.test"])
        raise AssertionError("compare accepted TEST")
    except SystemExit as ex:
        assert ex.code == 2, ex.code
'''),
])
print("ok")
