"""Gate of three data contracts of the training path (repository review 2026-10-31, F1 - F3); CPU, seconds, no nuScenes needed.

F1  the GT cache of `FrontRadarDataset._gt` is written atomically (a temporary name, then os.replace): a writer killed half-way
    leaves no file under the final name; a truncated or empty cache file (left by an older writer) is rebuilt, not a crash.
F2  `losses.sample_targets` keys the targets by SIDE (prev -0.5, cur 0, next +0.5): every list holds one entry per sample in
    batch order, also when a keyframe gap is 0.75 s (scene-0994; before, that sample's target landed under -1.0 s and the
    -0.5 s list had B - 1 entries for B predictions); the model decodes each sample with its true gap (`train.batch_ego`).
F3  `train.record_gt` writes runs/<name>/gt.json (the GT identity; `evaluation/bev_suite.py compare` checks it; its own checks are in
    evaluation/bev_selftest.py).

Usage (from the bev_fusion folder): python tests/test_gt_cache_targets.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from lift_pd import Grid  # noqa: E402
from losses import sample_targets  # noqa: E402
from nuscenes_dataset import FrontRadarDataset  # noqa: E402


class FakeGT:
    """stands in for autogt's NuScenesGT: a fixed signature and a tiny training sample; counts the generator calls"""

    def __init__(self):
        self.calls = 0

    def signature(self):
        return "testsig000"

    def training_sample(self, token, grid, ss):
        self.calls += 1
        layer = np.arange(12, dtype=np.float32).reshape(3, 4)
        return {k: {"drivable_area": layer + i, "meta": {"dt": dt}} for i, (k, dt) in enumerate((("cur", 0.0), ("prev", -0.5), ("next", 0.5)))}


def dataset(cache_dir):
    ds = FrontRadarDataset.__new__(FrontRadarDataset)        # no nuScenes: only what _gt reads
    ds.cache_dir, ds.grid, ds.gt_ss, ds.gt = cache_dir, Grid(), 4, FakeGT()
    return ds


def check_f1():
    with tempfile.TemporaryDirectory() as d:
        ds = dataset(d)
        g = ds._gt("tok")
        files = os.listdir(d)
        assert files == ["tok_0.5_4_testsig000.npz"], files                 # the final name only, no temporary left
        path = os.path.join(d, files[0])
        assert ds.gt.calls == 1 and np.array_equal(ds._gt("tok")["prev"]["drivable_area"], g["prev"]["drivable_area"])
        assert ds.gt.calls == 1, "the second read comes from the cache"
        # a truncated file and an empty file (a killed writer of the old code): rebuilt, not a crash
        for size in (os.path.getsize(path) // 2, 0):
            with open(path, "r+b") as f:
                f.truncate(size)
            n = ds.gt.calls
            g2 = ds._gt("tok")
            assert ds.gt.calls == n + 1 and np.array_equal(g2["next"]["drivable_area"], g["next"]["drivable_area"]), size
            assert os.listdir(d) == [files[0]] and os.path.getsize(path) > 0
        # a writer that dies half-way leaves nothing under the final name (and no temporary file)
        os.remove(path)
        real = np.savez_compressed

        def dies(file, *a, **kw):
            if hasattr(file, "write"):
                file.write(b"PK\x03\x04 partial")
            else:
                with open(file if str(file).endswith(".npz") else str(file) + ".npz", "wb") as f:
                    f.write(b"PK\x03\x04 partial")
            raise KeyboardInterrupt("killed")
        np.savez_compressed = dies
        try:
            ds._gt("tok")
            raise AssertionError("the writer must have died")
        except KeyboardInterrupt:
            pass
        finally:
            np.savez_compressed = real
        assert os.listdir(d) == [], os.listdir(d)
    print("F1 ok: GT cache written under a temporary name + os.replace; truncated and empty files rebuilt; a killed writer leaves nothing")


def gt_sample(prev_dt, next_dt, tag):
    """a training sample with the ego steps of its true keyframe gaps (12 Hz steps, 10 m/s)"""
    def side(dt):
        n = int(round(abs(dt) * 12))
        return {"meta": {"dt": dt, "steps": [(np.sign(dt) / 12, 10.0, 0.0)] * n}, "tag": tag}
    return {"cur": {"meta": {"dt": 0.0, "steps": []}, "tag": tag}, "prev": side(prev_dt), "next": side(next_dt)}


def check_f2():
    batch = [gt_sample(-0.5, 0.5, "a"), gt_sample(-0.75, 0.5, "b"), gt_sample(-0.5, 0.42, "c")]   # b: the 0.75 s gap of scene-0994
    gts = sample_targets(batch)
    assert sorted(gts) == [-0.5, 0.0, 0.5], sorted(gts)
    for dt, side in ((-0.5, "prev"), (0.0, "cur"), (0.5, "next")):
        assert [g["tag"] for g in gts[dt]] == ["a", "b", "c"], (dt, [g["tag"] for g in gts[dt]])
        assert all(g is s[side] for g, s in zip(gts[dt], batch)), dt
    # the model's output at -0.5 is decoded with each sample's TRUE gap (train.batch_ego keys by side, too)
    import train as T
    ego = T.batch_ego(batch, tuple(sorted(gts)), "cpu")
    assert np.allclose(ego[-0.5][0].numpy(), [-0.5, -0.75, -0.5], atol=1e-6), ego[-0.5][0]
    # a batch where only some samples carry a side cannot be aligned with the predictions: refused
    try:
        sample_targets([gt_sample(-0.5, 0.5, "a"), {"cur": batch[0]["cur"], "next": batch[0]["next"]}])
        raise AssertionError("a side missing in one sample must be refused")
    except ValueError:
        pass
    print("F2 ok: targets keyed by side, one per sample in batch order (0.75 s gap included); ego decode uses the true gap")


def check_f3():
    """train.record_gt: runs/<name>/gt.json holds the GT identity of the run; a resume on other GT keeps the earlier one"""
    import json
    import train as T

    class D:
        def __init__(self, sig):
            self.sig = sig

        def gt_signature(self):
            return self.sig
    with tempfile.TemporaryDirectory() as d:
        assert dataset(d).gt_signature() == "0.5_4_testsig000"                  # the suffix of the cache files
        assert T.record_gt(d, (D("0.5_4_aaa"), D("0.5_4_aaa"))) == "0.5_4_aaa"
        assert json.load(open(os.path.join(d, "gt.json"))) == {"gt_signature": "0.5_4_aaa"}
        T.record_gt(d, (D("0.5_4_aaa"), D("0.5_4_aaa")))                          # the same GT on resume: unchanged
        assert json.load(open(os.path.join(d, "gt.json"))) == {"gt_signature": "0.5_4_aaa"}
        T.record_gt(d, (D("0.5_4_bbb"), D("0.5_4_bbb")))                          # other GT on resume: recorded, the earlier kept
        assert json.load(open(os.path.join(d, "gt.json"))) == {"gt_signature": "0.5_4_bbb", "earlier": ["0.5_4_aaa"]}
        assert T.record_gt(d, (object(),)) is None                                 # datasets without a GT identity: nothing written
    print("F3 ok: gt.json records the GT identity of a run (grid, supersampling, autogt signature); a GT change on resume is kept")


if __name__ == "__main__":
    fails = 0
    for fn in (check_f1, check_f2, check_f3):
        try:
            fn()
        except Exception as e:                                       # noqa: BLE001
            import traceback
            fails += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
    print("gt cache + targets: " + ("all checks pass" if not fails else f"{fails} FAILED"))
    sys.exit(1 if fails else 0)
