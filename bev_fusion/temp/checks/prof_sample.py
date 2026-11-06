"""One-off: where do the seconds per sample go (cold GT cache vs warm)?"""
import os, sys, time, shutil
sys.path.insert(0, "training")
import torch
torch.set_num_threads(2)
from nuscenes.nuscenes import NuScenes
from feature_cache import CachedFrontRadarDataset, FeatureStore
from nuscenes_dataset import available_scenes
root = os.path.expanduser("~/data/nuscenes_full")
nusc = NuScenes("v1.0-trainval", dataroot=root, verbose=False)
sc = [t for t in available_scenes(nusc, "val") if nusc.get("scene", t)["name"] in ("scene-0003",)]
cold = "/tmp/gt_prof"
shutil.rmtree(cold, ignore_errors=True)
ds = CachedFrontRadarDataset(root, "v1.0-trainval", sc, store=FeatureStore("data/cache/cam_feats"), n_cam=3, cam_stride=2, with_gt=True, nusc=nusc, cache_dir=cold)
nogt = CachedFrontRadarDataset(root, "v1.0-trainval", sc, store=FeatureStore("data/cache/cam_feats"), n_cam=3, cam_stride=2, with_gt=False, nusc=nusc)
for i in range(3):
    t = time.time(); nogt[i]; a = time.time() - t
    t = time.time(); ds[i]; b = time.time() - t
    t = time.time(); ds[i]; c = time.time() - t
    print(f"sample {i}: no GT {a:.2f} s, cold GT {b:.2f} s, warm GT {c:.2f} s")
import cProfile, pstats
shutil.rmtree(cold, ignore_errors=True)
pr = cProfile.Profile(); pr.enable(); ds[5]; pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(14)
