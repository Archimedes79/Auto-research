"""Tests of net/prep/gt_cache_build.py (its former self-test).

    python -P -m bev_fusion.tests.test_gt_cache_build      (from the problem folder, with its module path)
"""
import os
import tempfile

from bev_fusion.net import netpath
from bev_fusion.net.prep.gt_cache_build import prefill
from bev_fusion.net.prep.nuscenes_dataset import FrontRadarDataset, available_scenes


def _selftest():
    os.chdir(netpath.SITE)                                       # the data folder: a relative cache folder is below it
    from nuscenes.nuscenes import NuScenes
    root = netpath.nuscenes_root(mini=True)
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
    _selftest()
