def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:90])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evaluation/bev_suite.py", [
    ('''from evaluation import bev_adapter as A  # noqa: E402  (numpy only at import: the chain's code is imported later, after setup_paths)''',
     '''from evaluation import bev_adapter as A  # noqa: E402  (numpy only at import: the chain's code is imported later, after setup_paths)
from evaluation import bev_sets  # noqa: E402'''),
    ('''        official = set(create_splits_scenes().get(split, []))
    except Exception:                                    # no devkit: the counts stay unknown, the names stay''',
     '''        official = set(create_splits_scenes().get(bev_sets.part_of(split), []))      # a set small.val is checked against the official val scenes
    except Exception:                                    # no devkit: the counts stay unknown, the names stay'''),
    ('''    return {"split_rule": f"nuScenes official split '{split}' (nuscenes.utils.splits.create_splits_scenes), restricted to the scenes with cached features; "''',
     '''    return {"set": split, "split_rule": f"nuScenes official split '{bev_sets.part_of(split)}' (nuscenes.utils.splits.create_splits_scenes), restricted to the scenes with cached features; "'''),
    ('''    p.add_argument("--tag", default="")
    p.add_argument("--n", default="all", help="the n of the evaluations (as in the store name evalTAG_nN)")''',
     '''    p.add_argument("--tag", default="")
    p.add_argument("--split", default="val", help="the set both runs were evaluated on (val, train or <split>.<part>, evaluation/bev_sets.py --list); it adds its suffix to --tag")
    p.add_argument("--n", default="all", help="the n of the evaluations (as in the store name evalTAG_nN)")'''),
    ('''    c = p.parse_args(argv)
    from evalsuite.quantitative import Quantitative
    name = f"eval{c.tag}''',
     '''    c = p.parse_args(argv)
    try:
        c.tag += bev_sets.tag(c.split)
    except ValueError as ex:
        p.error(str(ex))
    from evalsuite.quantitative import Quantitative
    name = f"eval{c.tag}'''),
    ('''b - a; a = {c.run_a}, b = {c.run_b}; val; {res['n_common_frames']} common frames")''',
     '''b - a; a = {c.run_a}, b = {c.run_b}; {c.split}; {res['n_common_frames']} common frames")'''),
    ('''    p = argparse.ArgumentParser(prog="report", allow_abbrev=False)''',
     '''    if bev_sets.parse(a.split)[0] is not None:
        print(f"report: the page of a run is made from its own evaluation (--split val | train), not from the set {a.split}")
        return 2
    p = argparse.ArgumentParser(prog="report", allow_abbrev=False)'''),
    ('''    p = argparse.ArgumentParser(prog="generalisation", allow_abbrev=False)''',
     '''    if bev_sets.parse(a.split)[0] is not None:
        print(f"generalisation: the run's own val and train lists (--split val), not the set {a.split}")
        return 2
    p = argparse.ArgumentParser(prog="generalisation", allow_abbrev=False)'''),
    ('''    p = argparse.ArgumentParser(prog="memory")''',
     '''    if bev_sets.parse(a.split)[0] is not None:
        print(f"memory: writes eval/memory.json of the run's own val frames (--split val), not of the set {a.split}")
        return 2
    p = argparse.ArgumentParser(prog="memory")'''),
    ('''    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--curves", action="store_true", help="also PR / ROC''',
     '''    p.add_argument("--split", default="val", help="the set to run on: val | train (the run's own lists) or <split>.<part> of splits/ (small.val, large.train, ...; evaluation/bev_sets.py --list)")
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--curves", action="store_true", help="also PR / ROC'''),
    ('''    a, command = p.parse_known_args(argv)                      # the global options anywhere; the rest is the command
    a.command = command''',
     '''    a, command = p.parse_known_args(argv)                      # the global options anywhere; the rest is the command
    try:
        bev_sets.parse(a.split)
    except ValueError as ex:
        p.error(str(ex))
    a.command = command'''),
    ('''--split val|train  --ckpt-dir DIR (moved checkpoints)''', '''--split val|train|<split>.<part>  --ckpt-dir DIR (moved checkpoints)'''),
])
print("ok")
