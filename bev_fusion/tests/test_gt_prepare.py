"""training/gt_scenes.py: the scene names of a config and the GtGenerator check / build commands built from them (no data, no GPU, no GtGenerator run: the commands are recorded).

  1. mini and trainval splits: the official scene lists of the config's train and val split (8 + 2 for mini, 700 + 150 for trainval), restricted by train_scenes /
     val_scenes, without repeats, train part first
  2. the names file lies next to the config, one name per line, and GtGenerator reads it (`GtGenerator.cli` accepts it: one name per line, # comments)
  3. check = `python -m GtGenerator have --scenes-file F --version V --dataroot D`, build = `bash GtGenerator/run_nice.sh -m GtGenerator generate D --version V --scenes-file F --workers N`,
     both run from the repository top; the exit code of the check is passed on; a wrong command is a usage error (exit 2)
  4. the whole chain with a real config of the project (t10_dual_s0): names for the whole trainval, a project.json "prepare" entry that `python -m Training start` can use
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_gt_prepare.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, "training")
import gt_scenes as G  # noqa: E402
from train import TrainConfig  # noqa: E402

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


# 1 the scenes of a config
mini = G.scene_names(TrainConfig(version="v1.0-mini", train_split="train", val_split="val"))
check(len(mini) == 10 and len(set(mini)) == 10 and mini[0] == "scene-0061", f"mini: 8 train + 2 val scenes, no repeats ({len(mini)})")
one = G.scene_names(TrainConfig(version="v1.0-mini", train_scenes=("scene-0061",), val_scenes=("scene-0103",)))
check(one == ["scene-0061", "scene-0103"], f"train_scenes / val_scenes restrict their part: {one}")
full = G.scene_names(TrainConfig(version="v1.0-trainval", train_split="train", val_split="val"))
check(len(full) == 850 and len(set(full)) == 850, f"trainval: 700 train + 150 val scenes ({len(full)})")
same = G.scene_names(TrainConfig(version="v1.0-trainval", train_split="val", val_split="val"))
check(len(same) == 150, "a scene in both parts is listed once")
check(G.scene_names(TrainConfig(version="v1.0-mini", train_split="mini_train", val_split="mini_val")) == mini, "mini_train / mini_val spelled out give the same list")

# 2 the names file
d = tempfile.mkdtemp()
cfg_path = os.path.join(d, "x.json")
json.dump({"version": "v1.0-mini", "train_scenes": ["scene-0061"], "val_scenes": ["scene-0103"]}, open(cfg_path, "w"))
check(G.names_path(cfg_path) == os.path.join(d, "x.scenes.txt"), "the names file lies next to the config")
names = G.write_names(__import__("train").read_config(cfg_path), G.names_path(cfg_path))
check(open(G.names_path(cfg_path)).read() == "scene-0061\nscene-0103\n" and names == ["scene-0061", "scene-0103"], "one name per line")

# 3 the commands
calls = []


def fake(code):
    def run(cmd, cwd=None):
        calls.append((cmd, cwd))
        return code
    return run


rc = G.main([cfg_path, "check"], runner=fake(1))
cmd, cwd = calls[-1]
check(rc == 1 and cmd[1:4] == ["-m", "GtGenerator", "have"] and cmd[cmd.index("--scenes-file") + 1] == G.names_path(cfg_path) and cmd[cmd.index("--version") + 1] == "v1.0-mini"
      and cmd[cmd.index("--dataroot") + 1] == os.path.expanduser("~/data/nuscenes_full") and cwd == G.REPO, f"check: {cmd[1:]}, exit code passed on, run from {cwd}")
check(os.path.isdir(os.path.join(G.REPO, "GtGenerator")) and os.path.isfile(os.path.join(G.REPO, "GtGenerator", "run_nice.sh")), "the repository top holds GtGenerator/run_nice.sh")
rc = G.main([cfg_path, "build", "--workers", "3"], runner=fake(0))
cmd, cwd = calls[-1]
check(rc == 0 and cmd[:5] == ["bash", "GtGenerator/run_nice.sh", "-m", "GtGenerator", "generate"] and cmd[cmd.index("--workers") + 1] == "3" and "--scenes-file" in cmd and "--no-lane" in cmd and cwd == G.REPO, f"build: {cmd[1:]} (the front GT only: no lane stores)")
rc = G.main([cfg_path, "build"], runner=fake(0))
check(calls[-1][0][calls[-1][0].index("--workers") + 1] == "2", "two workers by default")
n_before = len(calls)
with contextlib.redirect_stdout(io.StringIO()):                                  # the usage text
    bad = (G.main([cfg_path, "nonsense"], runner=fake(0)), G.main([], runner=fake(0)))
check(bad == (2, 2) and len(calls) == n_before, "a wrong command or none is a usage error and runs nothing")

# 4 a real config of the project and the project.json entry
real = os.path.join("training", "configs", "t10_dual_s0.json")
rn = G.scene_names(__import__("train").read_config(real))
cf = json.load(open(real))
check(len(rn) == len(set(rn)) and 10 <= len(rn) <= 850 and (cf.get("train_scenes") or cf.get("val_scenes") or len(rn) == 850), f"t10_dual_s0: {len(rn)} sequences, a subset of the official split where the config lists scenes")
pj = json.load(open("project.json"))
prep = pj.get("training", {}).get("prepare", [])
check(len(prep) == 1 and prep[0]["name"] == "gt" and "{config}" in prep[0]["check"] and "{config}" in prep[0]["build"] and "gt_scenes.py" in prep[0]["check"] and "gt_scenes.py" in prep[0]["build"]
      and prep[0]["check"].rstrip('"').endswith("check") and prep[0]["build"].rstrip('"').endswith("build"), "project.json: the prepare entry `gt` calls this wrapper for check and build")

print("GT PREPARE GATE: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
sys.exit(1 if fails else 0)
