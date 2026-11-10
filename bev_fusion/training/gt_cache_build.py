"""Prefill the GT cache (data/cache/gt) with several worker processes.

The dataset writes the GT of a sample (keyframes k-1, k, k+1) into the cache on first use, which costs seconds per sample
(map rasterisation) and stalls training; this tool does the same with `workers` processes before training, using the dataset's
own cache writer (`FrontRadarDataset._gt`), so the files are exactly the ones the trainer reads. The cache key contains the
signature of the generator (`FrontRadarDataset.gt_signature()`: GtGenerator source + flags): after an edit of GtGenerator the cache must be built
again. Resumable (existing files are
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


def memoize_keyframes(gt, keep=4):
    """Neighbouring samples share the keyframes k-1, k, k+1: remember the last `keep` keyframe GTs of this process (a worker
    gets consecutive samples). training_sample writes dt / steps into the meta of the dicts it receives, so every hit is
    returned as a copy of the dict and of its meta."""
    orig, memo = gt.keyframe_gt, {}

    def keyframe_gt(token, *args, **kw):
        key = (token, repr(args), repr(sorted(kw.items())))
        if key not in memo:
            if len(memo) >= keep:
                memo.pop(next(iter(memo)))
            memo[key] = orig(token, *args, **kw)
        r = memo[key]
        return {**r, "meta": dict(r["meta"])}
    gt.keyframe_gt = keyframe_gt


def prefill(ds, workers=4, log_every=200, chunk=16):
    """-> (samples, seconds spent in the workers, wall seconds). The workers get chunks of consecutive samples."""
    memoize_keyframes(ds.gt)                                                                # inherited by the forked workers
    n = len(ds.samples)
    chunks = [list(range(i, min(i + chunk, n))) for i in range(0, n, chunk)]
    loader = torch.utils.data.DataLoader(_GTOnly(ds), batch_sampler=chunks, num_workers=workers)
    t0, busy, done = time.time(), 0.0, 0
    for times in loader:
        busy += float(times.sum())
        done += len(times)
        print(f"{done}/{n} samples, {time.time() - t0:.0f} s, {busy / done:.2f} s per sample in the workers", flush=True)
    return n, busy, time.time() - t0


def _selftest():
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))                 # the bev_fusion folder
    from nuscenes.nuscenes import NuScenes
    root = os.path.expanduser("~/data/nuscenes")
    nusc = NuScenes("v1.0-mini", dataroot=root, verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        ds = FrontRadarDataset(root, "v1.0-mini", available_scenes(nusc, "mini_train")[:1], with_gt=True, cache_dir=tmp, nusc=nusc)
        ds.samples = ds.samples[:4]
        fresh = FrontRadarDataset(root, "v1.0-mini", available_scenes(nusc, "mini_train")[:1], with_gt=True, cache_dir=None, nusc=nusc)
        fresh.samples = ds.samples
        n, busy, wall = prefill(ds, workers=2, chunk=2)
        files = [f for f in os.listdir(tmp) if f.endswith(".npz")]
        assert n == 4 and len(files) == 4, (n, files)
        # the memoised keyframes give the same GT as a plain generation (dt / steps in the meta are not mixed up)
        import numpy as np
        for tok in ds.samples[1:3]:
            a, b = ds._gt(tok), fresh.gt.training_sample(tok, fresh.grid, fresh.gt_ss)
            assert set(a) == set(b) == {"cur", "prev", "next"}
            for k in a:
                assert set(a[k]["meta"]) == set(b[k]["meta"]), (k, set(a[k]["meta"]) ^ set(b[k]["meta"]))
                for n_, v in b[k].items():
                    if isinstance(v, np.ndarray):
                        assert np.allclose(a[k][n_].astype(np.float32), v, rtol=2e-3, atol=2e-3), (k, n_)   # cached as float16
                if k != "cur":
                    assert a[k]["meta"]["dt"] == b[k]["meta"]["dt"] and a[k]["meta"]["steps"] == b[k]["meta"]["steps"]

        def no_generation(*a, **k):
            raise AssertionError("second pass: cache miss, the GT was generated again")
        ds.gt.training_sample = no_generation                                                 # inherited by the forked workers
        n2, busy2, wall2 = prefill(ds, workers=2, chunk=2)                                   # second pass: all from the cache
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
