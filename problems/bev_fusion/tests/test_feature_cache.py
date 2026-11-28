"""Tests of net/prep/feature_cache.py (its former self-test).

    python tests/test_feature_cache.py      (from the problem folder, with its module path)
"""
import os
import time

import numpy as np
import torch
import torch.utils.checkpoint

from net.prep.feature_cache import CachedFrontRadarDataset, FEAT_HW, FeatureStore, N_CTX, N_PROB, build_radar_cache, check_feature_hw, collate_cached, shard_complete, write_shard
from net.prep.nuscenes_dataset import FrontRadarDataset


def _selftest_radar():
    """the radar cache returns exactly what FrontRadarDataset computes (dtypes, shapes, values); a second read comes from
    the file; the parallel fill covers every sample."""
    import tempfile
    root_nusc = os.path.expanduser("~/data/nuscenes")
    ds = FrontRadarDataset(root_nusc, "v1.0-mini", n_cam=3, cam_stride=2, with_gt=False)
    with tempfile.TemporaryDirectory() as tmp:
        cds = CachedFrontRadarDataset(root_nusc, "v1.0-mini", store=None, radar_cache=tmp, n_cam=3, cam_stride=2,
                                      with_gt=False, nusc=ds.nusc)
        cds.samples = cds.samples[:12]
        for t in cds.samples[:3]:
            s, cur = ds.nusc.get("sample", t), ds.gt.cam(t)
            ref = ds._radar(s, cur)
            first = cds._radar(s, cur)                                  # computed and written
            assert os.path.exists(cds._radar_path(s, cur))
            t0 = time.time()
            second = cds._radar(s, cur)                                 # read back
            t_read = time.time() - t0
            for k in ref:
                for got in (first, second):
                    assert got[k].dtype == ref[k].dtype and torch.equal(got[k], ref[k]), k
        n = build_radar_cache(cds, workers=2)
        assert n == 9 and all(os.path.exists(cds._radar_path(ds.nusc.get("sample", t), ds.gt.cam(t))) for t in cds.samples)
        assert not [f for f in os.listdir(tmp) if ".tmp" in f], "no partial files left"
        # a whole item through the cached dataset: cached features attached, radar equal to the plain dataset's; a batch
        # collates (this path broke once when the class lost its __getitem__)
        fdir, item_checked = "data/cache/cam_feats_mini", False
        if os.path.isdir(fdir):
            store = FeatureStore(fdir)
            cds.store, cds.stores = store, [store]                      # the item reads the per-source list made at construction
            ok = [i for i, t in enumerate(cds.samples) if all(store.has(f["token"]) for f in cds._cam_frames(t)[0])]
            if ok:
                it = cds[ok[0]]
                ref = ds[ds.samples.index(cds.samples[ok[0]])]
                assert all("prob" in f and "ctx" in f for f in it["frames"])
                assert all(torch.equal(it["radar"][k], ref["radar"][k]) for k in ref["radar"])
                b = collate_cached([it, it])
                assert b["prob"].shape[:2] == (2, 3) and len(b["radar"]["b"]) == 2 * len(ref["radar"]["x"])
                item_checked = True
    print(f"radar cache selftest ok: cached radar equals the computed one exactly (every column, dtype), read back in "
          f"{1000 * t_read:.1f} ms, parallel fill of 9 samples, no partial files; whole cached item + collate "
          f"{'checked' if item_checked else 'NOT checked (no mini feature cache)'}")


def _selftest_shards():
    """write_shard / FeatureStore without nuScenes: round trip, the feature-size guard, a cut build (row count != index)."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        a = np.random.rand(3, N_PROB + N_CTX, *FEAT_HW).astype(np.float16)
        write_shard(d, "scene-x", a, ["t0", "t1", "t2"])
        assert sorted(os.listdir(d)) == ["scene-x.index.npz", "scene-x.npy"], os.listdir(d)
        s = FeatureStore(d)
        p, c = s.get("t2")
        assert torch.equal(torch.cat([p, c]), torch.from_numpy(a[2])) and s.feature_hw() == FEAT_HW
        check_feature_hw(s, "dualbev")
        try:
            check_feature_hw(s, "bevfusion")
            raise AssertionError("a DualBEV-size cache passed as BEVFusion features")
        except ValueError:
            pass
        np.savez(os.path.join(d, "scene-x.index.npz"), tokens=np.array(["t0", "t1"], dtype="U40"))   # a build cut between the files
        try:
            FeatureStore(d).get("t0")
            raise AssertionError("a shard whose row count differs from its index was read")
        except RuntimeError:
            pass
        assert not shard_complete(d, "scene-x", {"t0", "t1"}), "a cut shard is rebuilt by the next build, not skipped"
        write_shard(d, "scene-x", a, ["t0", "t1", "t2"])
        assert shard_complete(d, "scene-x", {"t0", "t1"}) and not shard_complete(d, "scene-x", {"t9"})
        failed = False
        try:
            write_shard(d, "scene-y", np.array([lambda: 0], dtype=object), ["t0"])      # np.save cannot pickle it
        except Exception:                                                               # noqa: BLE001
            failed = True
        assert failed and not any(".tmp" in n for n in os.listdir(d)), os.listdir(d)
    print("feature store selftest ok: atomic shard round trip (no temporary file left after a failure), feature-size guard, cut build "
          "refused and rebuilt")


if __name__ == "__main__":
    _selftest_shards()
    _selftest_radar()
