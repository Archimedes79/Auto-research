"""scratch (diagnostic, no verdict): the config of the overfit test and a check that its scenes load.

8 TRAIN scenes (seeded random among the train scenes with stored GT of the signature t20 trained on), the network warm-started from the finished baseline t20_base_s0, trained on
these scenes only, no augmentation. If the lane layer can be fitted on 8 scenes, the loss and the resolution are able to carry the labels, and the weak lane markings are a matter of
generalisation (data, supervision efficiency) or of label noise that averages out; if even 8 scenes cannot be fitted, the labels, the loss or the representation are the limit.
python temp/lane_tests/overfit_prepare.py [--steps 600] [--seed 0]  -> temp/compose/overfit8_s0.json + temp/lane_tests/overfit8_scenes.txt"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]
SIG = "6a7aa7864a"


def train_scenes_with_gt():
    root = os.path.expanduser("~/data/nuscenes_full")
    from nuscenes.utils.splits import create_splits_scenes
    scenes = {s["token"]: s["name"] for s in json.load(open(os.path.join(root, "v1.0-trainval", "scene.json")))}
    samples = {s["token"]: s["scene_token"] for s in json.load(open(os.path.join(root, "v1.0-trainval", "sample.json")))}
    train = set(create_splits_scenes()["train"])
    n = {}
    for f in glob.glob(os.path.join(ROOT, "data", "cache", "gt", f"*_{SIG}.npz")):
        sc = scenes.get(samples.get(os.path.basename(f).split("_")[0], ""), "")
        if sc in train:
            n[sc] = n.get(sc, 0) + 1
    return {k: v for k, v in n.items() if v >= 30}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-scenes", type=int, default=8)
    a = ap.parse_args()
    base = json.load(open(os.path.join(ROOT, "temp", "compose", "t21_ctl_s0.json")))          # the composed config of the t21 control: 'split large' already expanded into scene lists
    names = sorted(base["train_scenes"])                                 # the scenes the baseline trained on: their features and GT are stored
    print(len(names), "train scenes of the baseline's split", flush=True)
    rng = np.random.RandomState(a.seed)
    order = list(rng.permutation(len(names)))
    base["_about"] = ("OVERFIT TEST (diagnostic, no verdict; Mirko's order 2026-11-05): the composed config of t21_ctl_s0 (the t20 network, window path, warm-started from the finished baseline runs/t20_base_s0) trained "
                      f"{a.steps} steps on 8 train scenes only, no augmentation, lr 1e-4. Question: can the lane layer be fitted on a handful of scenes? Frames per scene ~40.")
    base["name"] = "overfit8_s0"
    base["steps"] = a.steps
    base["warmup"] = 50
    base["ema"] = 0.95
    base["flip_aug"] = 0.0
    base["eval_every"] = a.steps
    base["eval_batches"] = 20
    base["save_every"] = 300
    base["grad_diag_every"] = 300
    got = [names[i] for i in order[:a.n_scenes]]
    from train import read_config, make_datasets
    base["train_scenes"] = got
    p = os.path.join(ROOT, "temp", "lane_tests", "_probe_cfg.json")
    json.dump(base, open(p, "w"))
    cfg = read_config(p)
    tr, _ = make_datasets(cfg)                                           # one build of the devkit tables for all 8 scenes
    n = len(tr)
    print(f"{n} samples in {len(got)} scenes; loading the first, a middle and the last", flush=True)
    for i in (0, n // 2, n - 1):
        it = tr[i]
        print("  item", i, "ok", flush=True)
    out = os.path.join(ROOT, "temp", "compose", "overfit8_s0.json")
    json.dump(base, open(out, "w"), indent=1)
    open(os.path.join(ROOT, "temp", "lane_tests", "overfit8_scenes.txt"), "w").write(" ".join(got) + "\n")
    print("scenes:", " ".join(got))
    print("wrote", out)


if __name__ == "__main__":
    main()
