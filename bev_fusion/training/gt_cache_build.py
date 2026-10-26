"""Prefill the GT cache (data/cache/gt) with several worker processes.

The dataset writes the GT of a sample (keyframes k-1, k, k+1) into the cache on first use, which costs seconds per sample
(map rasterisation) and stalls training; this tool does the same with `workers` processes before training, using the dataset's
own cache writer (`FrontRadarDataset._gt`), so the files are exactly the ones the trainer reads. The cache key contains the
signature of gt_bev.py: after an edit of the GT generator the cache must be built again. Resumable (existing files are
skipped by the writer). The workers are forks of one process that has loaded the devkit once (memory is shared).

Usage:  python training/gt_cache_build.py <dataroot> <version> <cache_dir> --split train|val|mini_train|mini_val
            [--scenes scene-0003 ...] [--workers 4] [--limit N]
Self-test (nuScenes mini, 2 workers, 4 samples): python training/gt_cache_build.py
"""
import argparse
import os
import sys
import tempfile
import time

import torch

try:
    from .nuscenes_dataset import FrontRadarDataset, available_scenes
except ImportError:                                   # run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nuscenes_dataset import FrontRadarDataset, available_scenes


class _GTOnly(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds.samples)

    def __getitem__(self, i):
        t = time.time()
        self.ds._gt(self.ds.samples[i])
        return time.time() - t


def prefill(ds, workers=4, log_every=200):
    """-> (samples, seconds spent in the workers, wall seconds)."""
    loader = torch.utils.data.DataLoader(_GTOnly(ds), batch_size=None, shuffle=False, num_workers=workers)
    t0, busy = time.time(), 0.0
    for n, dt in enumerate(loader, 1):
        busy += float(dt)
        if n % log_every == 0 or n == len(ds.samples):
            print(f"{n}/{len(ds.samples)} samples, {time.time() - t0:.0f} s, {busy / n:.2f} s per sample in the workers", flush=True)
    return len(ds.samples), busy, time.time() - t0


def _selftest():
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))                 # the bev_fusion folder
    from nuscenes.nuscenes import NuScenes
    root = os.path.expanduser("~/data/nuscenes")
    nusc = NuScenes("v1.0-mini", dataroot=root, verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        ds = FrontRadarDataset(root, "v1.0-mini", available_scenes(nusc, "mini_train")[:1], with_gt=True, cache_dir=tmp, nusc=nusc)
        ds.samples = ds.samples[:4]
        n, busy, wall = prefill(ds, workers=2, log_every=10)
        files = [f for f in os.listdir(tmp) if f.endswith(".npz")]
        assert n == 4 and len(files) == 4, (n, files)
        def no_generation(*a, **k):
            raise AssertionError("second pass: cache miss, the GT was generated again")
        ds.gt.training_sample = no_generation                                                 # inherited by the forked workers
        n2, busy2, wall2 = prefill(ds, workers=2, log_every=10)                              # second pass: all from the cache
        assert n2 == 4 and len(os.listdir(tmp)) == 4
        print(f"gt_cache_build selftest ok: 4 samples cached by 2 workers ({busy / 4:.1f} s per sample in the workers, "
              f"{wall:.0f} s wall), second pass reads them all without generating (no new files)")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _selftest()
    else:
        p = argparse.ArgumentParser(description="prefill the GT cache")
        p.add_argument("dataroot")
        p.add_argument("version")
        p.add_argument("cache_dir")
        p.add_argument("--split", required=True)
        p.add_argument("--scenes", nargs="*", default=None, help="scene names (default: the whole split)")
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--limit", type=int, default=None)
        a = p.parse_args()
        from nuscenes.nuscenes import NuScenes
        root = os.path.expanduser(a.dataroot)
        nusc = NuScenes(a.version, dataroot=root, verbose=False)
        toks = available_scenes(nusc, a.split)
        if a.scenes:
            toks = [t for t in toks if nusc.get("scene", t)["name"] in set(a.scenes)]
        ds = FrontRadarDataset(root, a.version, toks, with_gt=True, cache_dir=a.cache_dir, nusc=nusc)
        if a.limit:
            ds.samples = ds.samples[:a.limit]
        n, busy, wall = prefill(ds, a.workers)
        print(f"done: {n} samples, {wall:.0f} s wall, {busy / max(n, 1):.2f} s per sample in the workers")
