"""The split lists (`splits/*.json`) and `net/prep/splits.py` (resolve, manifest, split.json). Known answers, no data, no GPU.

  1. the files: `large` = exactly the lists the stored config of E27.t21_s0 carries inline (428 / 82 scenes; base.json names the split), `xl` 700 / 82 with large inside it, `small` a
     subset of `large` (60 / 10); train and val never share a scene; every file names a released version of GtGenerator; train scenes are
     official nuScenes train scenes, val scenes official val scenes
  2. resolve: `"split": "large"` (base.json) gives the lists of the stored inline config (same order), the key `split` is gone and `_split` names it; the
     trainer's `read_config` resolves a split itself and takes a resolved config; an inline config comes back unchanged; only `split: null` drops the key (keeping inline lists); `''`, 0, false,
     [] and a number are refused; split + inline lists (either list alone too), an unknown name and a path in the name are refused
  3. manifest and split.json of a run folder (built here with known content; the real run t20_base_s0 in the site's run root when it is there: split large
     428 / 82)
Run from the problem folder: ~/miniconda3/envs/gcar/bin/python -P -m bev_fusion.tests.test_splits
"""
import json
import os
import subprocess
import sys
import tempfile

import GtGenerator
from bev_fusion.net import netpath
from bev_fusion.net.prep import splits as S

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def raises(fn, text):
    try:
        fn()
    except ValueError as e:
        return text in str(e)
    return False


def config(name):
    with open(os.path.join("training", "configs", name + ".json"), encoding="utf-8") as f:
        return json.load(f)


base = config("base")
with open(os.path.join("experiments", "E027_t21_measured", "runs", "E27.t21_s0", "config.json"), encoding="utf-8") as f_:
    stored = json.load(f_)                                       # a config that carries its lists inline
large, small, xl = S.load("large"), S.load("small"), S.load("xl")

# 1 the files
check(base.get("split") == "large" and all(base.get(k) == [] for k in S.LISTS), "base.json names its split (large); its lists are empty (a card may set them, with split null)")
check(large["train"] == stored["train_scenes"] and large["val"] == stored["val_scenes"] and (len(large["train"]), len(large["val"])) == (428, 82), "large = the lists the stored config of E27.t21_s0 carries (428 / 82, same order)")
check((len(xl["train"]), len(xl["val"])) == (700, 82), "xl: 700 / 82 scenes")
check(set(large["train"]) < set(xl["train"]) and xl["val"] == large["val"], "xl = large + 272 train scenes, the same val scenes")
check(set(small["train"]) <= set(large["train"]) and set(small["val"]) <= set(large["val"]) and (len(small["train"]), len(small["val"])) == (60, 10), "small: 60 / 10 scenes, a subset of large")
check(all(not set(d["train"]) & set(d["val"]) for d in (small, large, xl)), "no scene is in train and val of one split")
versions = set(GtGenerator.version_names())
check(all(d["gt"] in versions for d in (small, large, xl)), f"every split names a released version of GtGenerator ({large['gt']})")
from nuscenes.utils.splits import create_splits_scenes
official = create_splits_scenes()
check(all(set(d["train"]) <= set(official["train"]) and set(d["val"]) <= set(official["val"]) for d in (small, large, xl)), "train scenes are official train scenes, val scenes official val scenes (the devkit's lists)")

# 2 resolve
inline = {k: v for k, v in base.items() if k != "split"}
res = S.resolve(base)
check(res["train_scenes"] == stored["train_scenes"] and res["val_scenes"] == stored["val_scenes"], "base.json (split large) resolves to the lists of the stored inline config")
check("split" not in res and res["_split"] == "large" and {k: v for k, v in res.items() if k not in S.LISTS + ("_split",)} == {k: v for k, v in inline.items() if k not in S.LISTS},
      "the key split is replaced, nothing else changes")
check(S.resolve(stored) is stored and S.resolve(dict(stored, name="x")) == dict(stored, name="x"), "an inline config comes back unchanged")
check(S.resolve(dict(stored, split=None)) == stored, "split null drops the key and keeps the inline lists")
check(raises(lambda: S.resolve(dict(inline, split=None)), "whole official split") and raises(lambda: S.resolve(dict(inline, split=None, train_scenes=[], val_scenes=[])), "whole official split"),
      "split null without a list is refused: it would train on the whole official split")
card = S.resolve(dict(inline, split=None, train_scenes=small["train"], val_scenes=small["val"]))
check(card["train_scenes"] == small["train"] and card["val_scenes"] == small["val"] and "split" not in card, "a card's explicit lists with split null are taken as they are")
check(all(raises(lambda v=v: S.resolve(dict(inline, split=v)), "split") for v in ("", 0, False, [], 1, {})),
      "only null drops the key: '', 0, false, [] and {} are refused with a message (1 too: a number is no name)")
check(raises(lambda: S.resolve(dict(inline, split="large", val_scenes=["scene-0003"])), "keep one") and raises(lambda: S.resolve(dict(inline, split="large", train_scenes=["scene-0001"])), "keep one"),
      "split + only an inline val_scenes (or only train_scenes) is refused too")
check(S.resolve(dict(inline, split="large", train_scenes=[], val_scenes=[]))["train_scenes"] == large["train"], "empty inline lists are no lists: the split fills them (the documented way to move a stored config to a split)")
check(raises(lambda: S.resolve(dict(stored, split="large")), "keep one") and raises(lambda: S.resolve(dict(inline, split="larg")), "no split 'larg'")
      and raises(lambda: S.resolve(dict(inline, split="../splits/large")), "letters, digits"), "split + inline lists, an unknown name and a path in the name are refused")
with tempfile.TemporaryDirectory() as tmp:                       # a split file that is wrong is refused when it is read
    for label, d in (("a scene in train and val", {"train": ["a", "b"], "val": ["b"]}), ("a scene twice", {"train": ["a", "a"], "val": ["b"]}), ("an empty list", {"train": ["a"], "val": []})):
        json.dump(d, open(os.path.join(tmp, "bad.json"), "w"))
        check(raises(lambda: S.load("bad", tmp), "bad.json"), f"split file with {label}: refused")
try:
    from bev_fusion.training.train import read_config  # noqa: E402 - needs torch (the gcar environment)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "c.json")
        for label, cfg_ in (("a config that names a split: resolved by the reader", dict(inline, split="small")), ("a resolved config (the comment key _split)", S.resolve(dict(inline, split="small")))):
            json.dump(cfg_, open(p, "w"))
            c = read_config(p)
            check(list(c.train_scenes) == small["train"] and list(c.val_scenes) == small["val"], f"the trainer's read_config holds the lists of {label}")
except ImportError as e:
    check(False, f"train.py not importable here ({e}): run the test in the gcar environment")
with tempfile.TemporaryDirectory() as tmp:
    cli = [sys.executable, "-P", "-m", S.__name__]                # the module's command line, started as research starts a module (the roots it was given)
    p = subprocess.run(cli + ["resolve"], input=json.dumps(dict(inline, split="xl")), capture_output=True, text=True)
    check(p.returncode == 0 and json.loads(p.stdout) == S.resolve(dict(inline, split="xl")), "the command line (stdin -> stdout) gives the same as resolve()")
    p = subprocess.run(cli + ["names", "small"], capture_output=True, text=True)
    check(p.returncode == 0 and p.stdout.split("\n")[:-1] == small["train"] + small["val"], "names: the scenes of a split, train then val, one per line (the format of GtGenerator --scenes-file)")
    p = subprocess.run(cli + ["resolve"], input=json.dumps(dict(inline, split="nope")), capture_output=True, text=True)
    check(p.returncode == 1 and "no split 'nope'" in p.stderr and "large" in p.stderr, "an unknown split on the command line: exit 1, the message names the splits that exist")

# 3 manifest and split.json
with tempfile.TemporaryDirectory() as tmp:
    run = os.path.join(tmp, "runs", "t99_toy_s0")
    os.makedirs(run)
    json.dump(S.resolve(dict(inline, split="small")), open(os.path.join(run, "config.json"), "w"))
    json.dump({"gt_signature": "0.5_4_6a7aa7864a", "gt_version": "v8"}, open(os.path.join(run, "gt.json"), "w"))
    rec = os.path.join(tmp, "run.json")
    json.dump({"hypothesis": "h", "prediction": "p", "rule": "r", "commit": "x"}, open(rec, "w"))
    m = S.manifest(run, rec)
    check(m == {"run": "t99_toy_s0", "split": {"name": "small", "train": {"n": 60, "hash": S.list_hash(small["train"])}, "val": {"n": 10, "hash": S.list_hash(small["val"])}},
                "gt": {"signature": "0.5_4_6a7aa7864a", "version": "v8"}, "card": {"hypothesis": "h", "prediction": "p", "rule": "r"}},
          "manifest: split name found from the lists, counts and hashes, GT, the card")
    check(S.list_hash(["b", "a"]) == S.list_hash(["a", "b"]) != S.list_hash(["a", "c"]), "the hash is of the set of scenes (order does not matter)")
    check(S.manifest(run)["card"] is None, "without a record the card is null")
    cfg = json.load(open(os.path.join(run, "config.json")))
    json.dump(dict(cfg, train_scenes=cfg["train_scenes"][:-1]), open(os.path.join(run, "config.json"), "w"))
    check(S.manifest(run)["split"]["name"] == "inline" and S.manifest(run)["split"]["train"]["n"] == 59, "one scene fewer is no named split any more: inline")
    json.dump(dict(cfg, val_scenes=large["val"]), open(os.path.join(run, "config.json"), "w"))
    check(S.manifest(run)["split"]["name"] == "inline", "small's train with large's val is no named split either (the val list counts)")
    json.dump(dict(cfg, train_scenes=[], val_scenes=[]), open(os.path.join(run, "config.json"), "w"))
    check(S.manifest(run)["split"]["name"] == "official", "no lists = the whole official split")
    json.dump(cfg, open(os.path.join(run, "config.json"), "w"))
    path = S.write_split_json(run, os.path.join(tmp, "results", "t99.toy_s0"), frames=["f1", "f2"])
    d = json.load(open(path))
    check(path.endswith(os.path.join("t99.toy_s0", "split.json")) and d["train"] == small["train"] and d["val"] == small["val"] and d["split"] == "small"
          and d["gt"] == {"signature": "0.5_4_6a7aa7864a", "version": "v8"} and d["evaluation_frames"] == ["f1", "f2"] and d["official_split"] == {"train": "train", "val": "val"},
          "split.json holds the lists of the run, the split name, the GT and the evaluation frames it was given")
    check(json.load(open(S.write_split_json(run, os.path.join(tmp, "results", "other"))))["evaluation_frames"] is None, "frames unknown = null")
    json.dump(dict(inline, split="small"), open(os.path.join(run, "config.json"), "w"))       # a run whose config names its split (an import-only run)
    dn = json.load(open(S.write_split_json(run, os.path.join(tmp, "results", "named"))))
    check((dn["split"], dn["train"], dn["val"]) == ("small", small["train"], small["val"]) and S.manifest(run)["split"]["name"] == "small",
          "split.json and the manifest of a run whose config names the split hold its lists, not 'official'")
    json.dump(cfg, open(os.path.join(run, "config.json"), "w"))
    check(raises(lambda: S.manifest(os.path.join(tmp, "none")), "no readable config.json"), "a folder without config.json is refused")
    check(d["note"] is None, "a run with a config has no note")
    # an older run: a folder, no config.json (9 such folders in runs/ on 2026-11-03)
    old = os.path.join(tmp, "runs", "t99_old_s0")
    os.makedirs(old)
    d0 = json.load(open(S.write_split_json(old, os.path.join(tmp, "results", "old"))))
    check((d0["train"], d0["val"], d0["split"]) == (None, None, None) and d0["note"] == "no config.json: scene lists unknown" and d0["gt"] == {"signature": None, "version": None},
          "split.json of a run without config.json: nulls and a note, not a refusal")
    check(raises(lambda: S.write_split_json(os.path.join(tmp, "none"), os.path.join(tmp, "results", "none")), "no run folder"), "split.json of a run folder that does not exist is refused")
real = os.path.join(netpath.SITE, "runs", "t20_base_s0")
if os.path.isdir(real):
    m = S.manifest(real)
    check(m["run"] == "t20_base_s0" and m["split"]["name"] == "large" and (m["split"]["train"]["n"], m["split"]["val"]["n"]) == (428, 82),
          f"the real run t20_base_s0: split large 428 / 82 ({m['split']['name']}, {m['split']['train']['n']} / {m['split']['val']['n']})")
else:
    print(f"SKIP the real run t20_base_s0: not in {os.path.join(netpath.SITE, 'runs')}")

print(f"{len(fails)} failed" if fails else "test_splits passed")
sys.exit(1 if fails else 0)
