"""Build or check the image cache a config names (net/prep/image_cache.py holds the cache itself):
    python training/build_image_cache.py build-cache | check-cache | build-cache-older | check-cache-older <config> [--workers 1] [--n 20]"""
import argparse
import json
import sys

import numpy as np
import torch

from net.prep import image_cache as IC
from net.prep import splits


def load_config(path):
    """the TrainConfig of a config file as the trainer reads it: a "split" is replaced by its scene lists (net/prep/splits.py resolve), so the builders
    cover the scenes of the run, not the whole official split"""
    from training import train as T
    return T.TrainConfig(**{k: v for k, v in splits.resolve(json.load(open(path))).items() if not k.startswith("_")})


def _cli(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-cache", "check-cache", "build-cache-older", "check-cache-older"])
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n", type=int, default=20, help="check-cache: samples compared with the dataset's own image")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    # the builders need the plain datasets: no cache to read (the one they build, the teachers' features), no student to feed
    cfg.lane_images = cfg.feature_cache = cfg.feature_cache2 = ""
    from training import train as T
    dss = T.make_datasets(cfg)
    root, older = cfg_root(a.config), a.cmd.endswith("older")
    if a.cmd.startswith("build"):
        print("written / checked" + (" the older frames of" if older else ""), IC.build_cache(dss, root, a.workers, older=older), "samples")
        return
    ds, errs = dss[1], []
    for i in range(0, len(ds.samples), max(1, len(ds.samples) // a.n))[:a.n]:
        for sd in (IC.older_records(ds, ds.samples[i]) if older else [ds.gt.cam(ds.samples[i])]):
            ref = torch.tensor(ds._image_uint8(sd)).permute(2, 0, 1).float()
            got = IC.load_image(IC.older_root(root) if older else root, sd["token"]).float()
            errs.append((got - ref).abs().mean().item())
    print(f"{'older-frame ' if older else ''}cache vs dataset image, mean abs error per pixel (0..255) over {len(errs)} val frames: mean {np.mean(errs):.2f}, max {np.max(errs):.2f}")
    assert np.max(errs) < 3.0, "the cache does not match the dataset's image"
    scenes = set(cfg.train_scenes) | set(cfg.val_scenes)
    missing = IC.older_missing_scenes(root, cfg.n_cam, cfg.cam_stride, scenes) if older else IC.key_missing_scenes(root, scenes)
    assert not missing, f"BUILT.json has no finished build of {IC.older_key(cfg.n_cam, cfg.cam_stride) if older else IC.KEY} for {len(missing)} scene(s) of the config, e.g. {missing[:3]}"
    print(f"BUILT.json covers all {len(scenes)} scenes of the config")


def cfg_root(config):
    root = json.load(open(config)).get("lane_images", "")
    if not root:
        raise SystemExit(f"{config} has no lane_images folder")
    return root


if __name__ == "__main__":
    _cli(sys.argv[1:])
