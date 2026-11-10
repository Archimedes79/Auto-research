"""The split lists (`splits/*.json`), `training/splits.py` (resolve, manifest, split.json) and the Training hook that calls it. Known answers, no data, no GPU.

  1. the files: `large` = exactly the lists of the stored config t16_full.json (428 / 82 scenes), `xl` = those of t17_data.json (700 / 82, large inside it), `small` a
     subset of `large` (60 / 10); train and val never share a scene; every file names a GtGenerator version of GtGenerator/versions.csv; train scenes are official nuScenes train scenes, val scenes official val scenes
  2. resolve: `"split": "large"` gives the lists of the inline t16_full.json (same order), the key `split` is gone (the trainer refuses it) and `_split` names it; the
     trainer's `read_config` accepts the result; an inline config comes back unchanged; only `split: null` drops the key (keeping inline lists); `''`, 0, false, [] and a number
     are refused; split + inline lists (either list alone too), an unknown name and a path in the name are refused
  3. through the code of `Training check` / `run` (`plan_start` with the real block of project.json, whose resolve command is checked): the lists arrive in the config, a note says so;
     the jobs of splits/README.md (move a stored config to a split) work, the plain `"set": {"split": NAME}` on a config with lists is refused
  4. manifest and split.json of a run folder (built here with known content; the real run t16_full_s0 when it is on disk: split large, commit 40e7ff5b3c)
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_splits.py
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.getcwd()
sys.path.insert(0, "training")
sys.path.insert(0, os.path.join(ROOT, "..", "autoresearch"))
import splits as S  # noqa: E402

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


t16, t17 = config("t16_full"), config("t17_data")
large, small, xl = S.load("large"), S.load("small"), S.load("xl")

# 1 the files
check(large["train"] == t16["train_scenes"] and large["val"] == t16["val_scenes"] and (len(large["train"]), len(large["val"])) == (428, 82), "large = the lists of t16_full.json (428 / 82, same order)")
check(xl["train"] == t17["train_scenes"] and xl["val"] == t17["val_scenes"] and (len(xl["train"]), len(xl["val"])) == (700, 82), "xl = the lists of t17_data.json (700 / 82)")
check(set(large["train"]) < set(xl["train"]) and xl["val"] == large["val"], "xl = large + 272 train scenes, the same val scenes")
check(set(small["train"]) <= set(large["train"]) and set(small["val"]) <= set(large["val"]) and (len(small["train"]), len(small["val"])) == (60, 10), "small: 60 / 10 scenes, a subset of large")
check(all(not set(d["train"]) & set(d["val"]) for d in (small, large, xl)), "no scene is in train and val of one split")
with open(os.path.join("..", "GtGenerator", "versions.csv"), newline="", encoding="utf-8") as f:
    versions = {r["version"] for r in csv.DictReader(f)}
check(all(d["gt"] in versions for d in (small, large, xl)), f"every split names a GtGenerator version ({large['gt']}) of GtGenerator/versions.csv")
from nuscenes.utils.splits import create_splits_scenes  # noqa: E402
official = create_splits_scenes()
check(all(set(d["train"]) <= set(official["train"]) and set(d["val"]) <= set(official["val"]) for d in (small, large, xl)), "train scenes are official train scenes, val scenes official val scenes (the devkit's lists)")

# 2 resolve
inline = {k: v for k, v in t16.items() if k not in ("train_scenes", "val_scenes")}
res = S.resolve(dict(inline, split="large"))
check(res["train_scenes"] == t16["train_scenes"] and res["val_scenes"] == t16["val_scenes"], "split large resolves to the lists of the inline t16_full.json")
check("split" not in res and res["_split"] == "large" and {k: v for k, v in res.items() if k not in ("train_scenes", "val_scenes", "_split")} == inline, "the key split is replaced, nothing else changes")
check(S.resolve(t16) is t16 and S.resolve(dict(t16, name="x")) == dict(t16, name="x"), "an inline config comes back unchanged")
check(S.resolve(dict(t16, split=None)) == t16, "split null drops the key and keeps the inline lists")
check(all(raises(lambda v=v: S.resolve(dict(inline, split=v)), "split") for v in ("", 0, False, [], 1, {})),
      "only null drops the key: '', 0, false, [] and {} are refused with a message (1 too: a number is no name)")
check(raises(lambda: S.resolve(dict(inline, split="large", val_scenes=["scene-0003"])), "keep one") and raises(lambda: S.resolve(dict(inline, split="large", train_scenes=["scene-0001"])), "keep one"),
      "split + only an inline val_scenes (or only train_scenes) is refused too")
check(S.resolve(dict(inline, split="large", train_scenes=[], val_scenes=[]))["train_scenes"] == large["train"], "empty inline lists are no lists: the split fills them (the documented way to move a stored config to a split)")
check(raises(lambda: S.resolve(dict(t16, split="large")), "keep one") and raises(lambda: S.resolve(dict(inline, split="larg")), "no split 'larg'")
      and raises(lambda: S.resolve(dict(inline, split="../splits/large")), "letters, digits"), "split + inline lists, an unknown name and a path in the name are refused")
with tempfile.TemporaryDirectory() as tmp:                       # a split file that is wrong is refused when it is read
    for label, d in (("a scene in train and val", {"train": ["a", "b"], "val": ["b"]}), ("a scene twice", {"train": ["a", "a"], "val": ["b"]}), ("an empty list", {"train": ["a"], "val": []})):
        json.dump(d, open(os.path.join(tmp, "bad.json"), "w"))
        check(raises(lambda: S.load("bad", tmp), "bad.json"), f"split file with {label}: refused")
try:
    from train import read_config  # noqa: E402 - needs torch (the gcar environment)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "c.json")
        json.dump(S.resolve(dict(inline, split="small")), open(p, "w"))
        c = read_config(p)
        check(list(c.train_scenes) == small["train"] and list(c.val_scenes) == small["val"], "the trainer's read_config takes the resolved config and holds the lists")
        json.dump(dict(inline, split="small"), open(p, "w"))
        try:
            read_config(p)
            check(False, "the trainer refuses an unresolved split key")
        except TypeError:
            check(True, "the trainer refuses an unresolved split key (loud, not 700 scenes silently)")
except ImportError as e:
    check(False, f"train.py not importable here ({e}): run the test in the gcar environment")
with tempfile.TemporaryDirectory() as tmp:
    p = subprocess.run([sys.executable, "training/splits.py", "resolve"], input=json.dumps(dict(inline, split="xl")), capture_output=True, text=True)
    check(p.returncode == 0 and json.loads(p.stdout) == S.resolve(dict(inline, split="xl")), "the command line (stdin -> stdout) gives the same as resolve()")
    p = subprocess.run([sys.executable, "training/splits.py", "names", "small"], capture_output=True, text=True)
    check(p.returncode == 0 and p.stdout.split("\n")[:-1] == small["train"] + small["val"], "names: the scenes of a split, train then val, one per line (the format of gt_scenes.py names)")
    p = subprocess.run([sys.executable, "training/splits.py", "resolve"], input=json.dumps(dict(inline, split="nope")), capture_output=True, text=True)
    check(p.returncode == 1 and "no split 'nope'" in p.stderr and "large" in p.stderr, "an unknown split on the command line: exit 1, the message names the splits that exist")

# 3 the Training hook
from Training import compose as C  # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    block = json.load(open("project.json"))["training"]
    check(block.get("resolve") == "python3 training/splits.py resolve", "project.json: training.resolve is the splits.py command (the hook is live)")
    check(json.load(open("project.json")).get("package", {}).get("split_record") == "python3 training/splits.py split-json {run_dir} {out_dir}", "project.json: package.split_record is the splits.py command (package.py writes results/<id>/split.json with it)")
    base = os.path.join(tmp, "base.json")
    json.dump(dict(inline, split="small"), open(base, "w"))
    plan = C.plan_start(ROOT, block, base, "split.test", [], [], no_card=True)
    check(plan["config"]["train_scenes"] == small["train"] and plan["config"]["val_scenes"] == small["val"] and "split" not in plan["config"], "plan_start: the composed config carries the lists of the split")
    check("resolve: train_scenes = 60 entries" in plan["notes"] and "resolve: split removed" in plan["notes"], f"plan_start: a note names what was resolved ({plan['notes']})")
    json.dump(dict(inline, split="nope"), open(base, "w"))
    try:
        C.plan_start(ROOT, block, base, "split.test", [], [], no_card=True)
        check(False, "plan_start refuses an unknown split")
    except C.ComposeError as e:
        check("no split 'nope'" in str(e), "plan_start refuses an unknown split with the message of the resolver")
    json.dump(dict(t16, name="t16_full"), open(base, "w"))
    check(C.plan_start(ROOT, block, base, "split.test", [], [], no_card=True)["notes"] == [], "plan_start: an inline config passes the hook untouched, no note")
    # the commands of splits/README.md: a stored config with lists moves to a split with --set train_scenes=[] --set val_scenes=[] --set split=NAME; plain --set split=NAME is refused
    cfg_ = C.plan_start(ROOT, block, base, "split.test", ["small"], ["train_scenes=[]", "val_scenes=[]", "split=small"], no_card=True)["config"]
    check(cfg_["train_scenes"] == small["train"] and cfg_["val_scenes"] == small["val"] and cfg_["steps"] == 400 and "split" not in cfg_, "plan_start: --preset small --set train_scenes=[] --set val_scenes=[] --set split=small on t16_full.json")
    try:
        C.plan_start(ROOT, block, base, "split.test", [], ["split=small"], no_card=True)
        check(False, "plan_start refuses --set split=small on a config that has lists")
    except C.ComposeError as e:
        check("--set train_scenes=[]" in str(e), "plan_start refuses --set split=small on a config that has lists and the message names the command that works")

# 4 manifest and split.json
with tempfile.TemporaryDirectory() as tmp:
    run = os.path.join(tmp, "runs", "t99_toy_s0")
    os.makedirs(os.path.join(run, "code"))
    json.dump(S.resolve(dict(inline, split="small")), open(os.path.join(run, "config.json"), "w"))
    json.dump({"gt_signature": "0.5_4_6a7aa7864a", "gt_version": "v8"}, open(os.path.join(run, "gt.json"), "w"))
    open(os.path.join(run, "code", "SNAPSHOT.txt"), "w").write("commit 0123456789 (git archive of bev_fusion/training; run t99.toy_s0, tag bf/t99.toy_s0)\n")
    rec = os.path.join(tmp, "run.json")
    json.dump({"hypothesis": "h", "prediction": "p", "rule": "r", "commit": "x"}, open(rec, "w"))
    m = S.manifest(run, rec)
    check(m == {"run": "t99_toy_s0", "split": {"name": "small", "train": {"n": 60, "hash": S.list_hash(small["train"])}, "val": {"n": 10, "hash": S.list_hash(small["val"])}},
                "gt": {"signature": "0.5_4_6a7aa7864a", "version": "v8"}, "code": {"commit": "0123456789", "tag": "bf/t99.toy_s0"}, "card": {"hypothesis": "h", "prediction": "p", "rule": "r"}},
          "manifest: split name found from the lists, counts and hashes, GT, commit and tag of the snapshot, the card")
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
    check(raises(lambda: S.manifest(os.path.join(tmp, "none")), "no readable config.json"), "a folder without config.json is refused")
    check(d["note"] is None, "a run with a config has no note")
    old = os.path.join(tmp, "runs", "t99_old_s0")                                  # an older run: a folder, no config.json (9 such folders in runs/ on 2026-11-03)
    os.makedirs(old)
    d0 = json.load(open(S.write_split_json(old, os.path.join(tmp, "results", "old"))))
    check((d0["train"], d0["val"], d0["split"]) == (None, None, None) and d0["note"] == "no config.json: scene lists unknown" and d0["gt"] == {"signature": None, "version": None},
          "split.json of a run without config.json: nulls and a note, not a refusal")
    check(raises(lambda: S.write_split_json(os.path.join(tmp, "none"), os.path.join(tmp, "results", "none")), "no run folder"), "split.json of a run folder that does not exist is refused")
real = os.path.join("runs", "t16_full_s0")
if os.path.isdir(real):
    m = S.manifest(real)
    check(m["split"]["name"] == "large" and (m["split"]["train"]["n"], m["split"]["val"]["n"]) == (428, 82) and m["code"]["commit"] == "40e7ff5b3c" and m["gt"]["version"] == "v8",
          f"the real run t16_full_s0: split large 428 / 82, commit 40e7ff5b3c, GT v8 ({m['split']['name']}, {m['code']['commit']}, {m['gt']['version']})")
else:
    print("  skip the real run t16_full_s0 (not on this disk)")

print(f"{len(fails)} failed" if fails else "test_splits passed")
sys.exit(1 if fails else 0)
