"""Gate of three data contracts of the training path (repository review 2026-10-31, F1 - F3); CPU, seconds, no nuScenes needed.

F1  the GT cache of `FrontRadarDataset._gt` is written atomically (a temporary name, then os.replace): a writer killed half-way
    leaves no file under the final name; a truncated or empty cache file (left by an older writer) is rebuilt, not a crash.
F2  `losses.sample_targets` keys the targets by SIDE (prev -0.5, cur 0, next +0.5): every list holds one entry per sample in
    batch order, also when a keyframe gap is 0.75 s (scene-0994; before, that sample's target landed under -1.0 s and the
    -0.5 s list had B - 1 entries for B predictions); the model decodes each sample with its true gap (`train.batch_ego`).
F3  `train.record_gt` writes runs/<name>/gt.json (the GT identity; `evaluation/bev_suite.py compare` checks it; its own checks are in
    evaluation/bev_selftest.py).
F4  the GT cache key (`FrontRadarDataset.gt_signature`) holds the grid extents when they are not the default front grid (second
    repository review 2026-10-31, B5); the default grid keeps the names of all existing cache files.
F5  `autogt build` writes a provisional index.json before the first keyframe and refuses to continue a folder of another resolution /
    supersampling, also a folder of an interrupted build without index (B3).
F6  `autogt.video.run` leaves no temporary pictures after a rendering, write or encoding error (B8).
F7  `train.trainable_state` / `load_checkpoint`: with freeze_camera false the camera weights are in the checkpoint and restored (B4).

Usage (from the bev_fusion folder): python tests/test_gt_cache_targets.py
"""
import os
import sys
import tempfile
import time

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
    ego = T.batch_ego([{k: s[k]["meta"]["steps"] for k in ("prev", "next")} for s in batch], tuple(sorted(gts)), "cpu")
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
    from autogt.versions import table
    v5 = next(r for r in table() if r["version"] == "v5")["signature"]
    with tempfile.TemporaryDirectory() as d2:                                      # D29: a released signature also gets its version key
        T.record_gt(d2, (D("0.5_4_" + v5), D("0.5_4_" + v5)))
        assert json.load(open(os.path.join(d2, "gt.json"))) == {"gt_signature": "0.5_4_" + v5, "gt_version": "v5"}
    print("F3 ok: gt.json records the GT identity of a run (grid, supersampling, autogt signature); a GT change on resume is kept")


def check_f4():
    """gt_signature (= the suffix of the cache files) carries the grid extents when they are not the default front grid"""
    with tempfile.TemporaryDirectory() as d:
        ds = dataset(d)
        assert ds.gt_signature() == "0.5_4_testsig000"                             # the default grid keeps the names of every existing cache file
        other = Grid(x_max=40.0, y_min=-20.0, y_max=20.0)                          # same resolution, other extents: other GT
        ds.grid = other
        assert ds.gt_signature() == "0.5_x0_40_y-20_20_4_testsig000", ds.gt_signature()
        ds.grid = Grid()
        ds._gt("tok")
        n = ds.gt.calls
        ds.grid = other
        ds._gt("tok")
        assert ds.gt.calls == n + 1, "a cache written for other extents must not be returned"
        assert sorted(os.listdir(d)) == ["tok_0.5_4_testsig000.npz", "tok_0.5_x0_40_y-20_20_4_testsig000.npz"], os.listdir(d)
        # the model's grid (cfg.model["grid_kwargs"]) and the datasets' grid must agree: otherwise the labels are cut for another area
        import train as TR

        class Model:
            grid = Grid()
        ds.grid = Grid()
        TR.check_grids(Model(), ds)
        Model.grid = Grid(x_min=10.0, x_max=70.0)
        try:
            TR.check_grids(Model(), ds)
            raise AssertionError("a model grid other than the GT grid must be refused")
        except ValueError as e:
            assert "not the grid of the GT" in str(e)
        # the evaluation's copy of the identity is the dataset's own (one formula)
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from evaluation import bev_adapter
        ds.grid = other
        assert bev_adapter.gt_signature(ds) == ds.gt_signature() == "0.5_x0_40_y-20_20_4_testsig000"
    print("F4 ok: the GT cache key holds the grid extents when they differ from the default front grid; default names unchanged")


def refused(fn, what):
    try:
        fn()
    except SystemExit as e:
        assert what in str(e), str(e)
        return
    raise AssertionError(f"expected a refusal ({what})")


def check_f5():
    """autogt build: a folder of another resolution / supersampling is never continued, also after an interrupted build"""
    import json
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from autogt import cli
    from autogt.grid import GridSpec

    class Gen:
        grid, ss = GridSpec(res=0.5), 4

        def signature(self):
            return "testsig000"
    with tempfile.TemporaryDirectory() as d:
        sig_dir = os.path.join(d, "testsig000")
        os.makedirs(sig_dir)
        # an interrupted build of the time before the provisional index: keyframes, no index.json -> compared with the cell grid
        np.savez_compressed(os.path.join(sig_dir, "a.npz"), drivable_area=np.zeros((120, 104), np.float16))
        cli.check_resume(sig_dir, GridSpec(res=0.5), 4)
        refused(lambda: cli.check_resume(sig_dir, GridSpec(res=0.25), 4), "(240, 208)")
        # the provisional index of a running build says what the folder holds
        cli.write_index(sig_dir, Gen(), "v1.0-mini", ["scene-0061"], 1, 40, {}, complete=False)
        assert {k: v for k, v in json.load(open(os.path.join(sig_dir, "index.json"))).items() if k != "data_identity"} == {     # data_identity: the raw files the GT key names (autogt 2026-11-10)
            "signature": "testsig000", "version": "v1.0-mini", "res": 0.5, "ss": 4, "options": {}, "scenes": ["scene-0061"],
            "stride": 1, "keyframes": 40, "complete": False}
        cli.check_resume(sig_dir, GridSpec(res=0.5), 4)
        refused(lambda: cli.check_resume(sig_dir, GridSpec(res=0.25), 4), "holds res 0.5")
        refused(lambda: cli.check_resume(sig_dir, GridSpec(res=0.5), 2), "ss 4")
        from autogt import clean                                                                       # `clean --stale-caches` leaves a running build alone
        assert clean._running(sig_dir), "a provisional index written just now: a build may be running"
        os.utime(sig_dir, (time.time() - 7200, time.time() - 7200))
        assert not clean._running(sig_dir), "a provisional folder untouched for two hours is an interrupted build"
        cli.write_index(sig_dir, Gen(), "v1.0-mini", ["scene-0061"], 1, 40, {})                       # the end of the build
        assert json.load(open(os.path.join(sig_dir, "index.json")))["complete"] is True and not clean._running(sig_dir)
        assert sorted(os.listdir(sig_dir)) == ["a.npz", "index.json"], os.listdir(sig_dir)             # no temporary file left
        cli.check_resume(os.path.join(d, "nothing_yet"), GridSpec(res=0.25), 4)                       # a new folder is fine

    # the wiring: `build` itself (a fake generator in place of AutoGT) and `clean.collect`
    import autogt.suite as suite_mod

    class FakeGT:
        seen, sig_dir = 0, None

        def __init__(self, dataroot, version, grid=None, ss=4, **options):
            self.grid, self.ss = grid, ss

        def signature(self):
            return "wiresig000"

        def scenes(self):
            return ["scene-1"]

        def keyframes(self, scene):
            return ["kfA", "kfB"]

        def keyframe(self, token):
            with open(os.path.join(FakeGT.sig_dir, "index.json")) as f:
                assert json.load(f)["complete"] is False, "the provisional index must exist before the first keyframe"
            FakeGT.seen += 1
            return {"drivable_area": np.zeros((self.grid.nx, self.grid.ny), np.float32), "meta": {"sample": token}}
    real = suite_mod.AutoGT
    suite_mod.AutoGT = FakeGT
    try:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out")
            FakeGT.sig_dir = os.path.join(out, "wiresig000")
            index = os.path.join(FakeGT.sig_dir, "index.json")
            assert cli.build("root", "v1.0-mini", out, res=0.5, verbose=False) == 2 and FakeGT.seen == 2
            with open(index) as f:
                assert json.load(f)["complete"] is True
            refused(lambda: cli.build("root", "v1.0-mini", out, res=0.25, verbose=False), "holds res 0.5")
            assert FakeGT.seen == 2, "a refused build must not generate anything"
            assert cli.build("root", "v1.0-mini", out, res=0.5, verbose=False) == 0               # a resume: all there already
            os.remove(index)                                                                       # an interrupted build of the time before the provisional index
            refused(lambda: cli.build("root", "v1.0-mini", out, res=0.25, verbose=False), "no index.json")
    finally:
        suite_mod.AutoGT = real
    from autogt import clean as CL
    with tempfile.TemporaryDirectory() as d:
        caches = os.path.join(d, "caches")
        for name, complete in (("old_done", True), ("old_running", False), ("cur", True)):
            os.makedirs(os.path.join(caches, name))
            with open(os.path.join(caches, name, "index.json"), "w") as f:
                json.dump({"complete": complete}, f)
        stale = {os.path.basename(p) for p, _, _ in CL.collect(stale_caches=caches, current_signature={"cur"}, base=d)}
        every = {os.path.basename(p) for p, _, _ in CL.collect(all_caches=caches, base=d)}
        assert stale == {"old_done"} and every == {"old_done", "cur"}, (stale, every)         # a build that may be running is not offered, by either option
    print("F5 ok: autogt build refuses to continue a folder of another resolution / supersampling (provisional index, or the stored grid)")


def check_f6():
    """autogt video: a rendering, encoding or write error leaves no temporary pictures; the output folder is created"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from autogt import video

    class Nusc:
        def get(self, kind, tok):
            return {"timestamp": int(float(tok) * 1e6)}

    class Layers:
        nusc = Nusc()

        def __init__(self, *a, **k):
            pass

        def keyframes(self, scene):
            return ["0", "1"]

        def camera_track(self, tok):
            return [{"timestamp": int(i * 1e5), "is_key_frame": i % 5 == 0} for i in range(11)]

        def ego_pose(self, sd):
            return np.eye(4)

    class Scene:
        def __init__(self, nusc, root, name):
            self.scene = {"description": "test"}
    keep = {k: getattr(video, k) for k in ("AutoGT", "NuScenesScene", "render", "encode")}
    imwrite = video.cv2.imwrite
    video.AutoGT, video.NuScenesScene, video.render = Layers, Scene, lambda *a: np.zeros((8, 8, 3), np.uint8)
    try:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "not_there_yet", "v.mp4")

            def encode_fails(paths, path, fps):
                assert len(paths) == 11 and all(os.path.exists(p) for p in paths)
                raise RuntimeError("encoder died")
            video.encode = encode_fails
            try:
                video.run("scene-x", out)
                raise AssertionError("the encoder error must come through")
            except RuntimeError as e:
                assert "encoder died" in str(e)
            assert os.listdir(os.path.dirname(out)) == [], os.listdir(os.path.dirname(out))             # the temporary folder is gone
            video.encode = lambda paths, path, fps: {"path": path, "frames": len(paths), "codec": "h264", "method": "test"}
            video.cv2.imwrite = lambda *a: False                                                         # a failed write is reported by its return value
            try:
                video.run("scene-x", out)
                raise AssertionError("a failed image write must raise")
            except OSError as e:
                assert "could not write" in str(e)
            assert os.listdir(os.path.dirname(out)) == []
            video.cv2.imwrite = imwrite
            assert video.run("scene-x", out)["frames"] == 11 and os.listdir(os.path.dirname(out)) == []
    finally:
        for k, v in keep.items():
            setattr(video, k, v)
        video.cv2.imwrite = imwrite
    print("F6 ok: autogt video leaves no temporary pictures after an encoder or write error; the output folder is created")


def check_f7():
    """train.trainable_state / load_checkpoint: camera weights are part of the checkpoint when the camera is trained (freeze_camera false)"""
    import dataclasses
    import torch
    import train as T

    class Net(torch.nn.Module):
        def __init__(self, frozen):
            super().__init__()
            self.cfg = type("C", (), {"freeze_camera": frozen})()
            self.camera = torch.nn.Linear(2, 2)
            self.head = torch.nn.Linear(2, 2)

    @dataclasses.dataclass
    class Cfg:
        x: int = 1

    def saved(frozen, path):
        torch.manual_seed(1)
        net, loss = Net(frozen), torch.nn.Linear(1, 1)
        with torch.no_grad():
            net.camera.weight.fill_(7.0)
        T.save_checkpoint(path, net, loss, torch.optim.SGD(net.parameters(), lr=0.1), T.EMA(net, 0.99), 3, Cfg())
        return torch.load(path, map_location="cpu", weights_only=False)

    def loaded(frozen, path):
        torch.manual_seed(2)
        net, loss = Net(frozen), torch.nn.Linear(1, 1)
        T.load_checkpoint(path, net, loss, torch.optim.SGD(net.parameters(), lr=0.1), T.EMA(net, 0.99))
        return net
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ck.pt")
        assert not any(k.startswith("model.camera.") for k in saved(True, p)["state"])               # frozen: the pre-trained file holds them
        assert not torch.allclose(loaded(True, p).camera.weight, torch.full((2, 2), 7.0))
        assert any(k.startswith("model.camera.") for k in saved(False, p)["state"])                  # trained: part of the state
        assert torch.allclose(loaded(False, p).camera.weight, torch.full((2, 2), 7.0)), "a resume must restore the trained camera weights"
        saved(True, p)                                                                               # a checkpoint without them, a model that trains them
        try:
            loaded(False, p)
            raise AssertionError("resuming a trained camera from a checkpoint without camera weights must be refused")
        except RuntimeError as e:
            assert "no camera weights" in str(e)
    print("F7 ok: a trained camera is part of the checkpoint and restored on resume; a checkpoint without it is refused for such a model")


if __name__ == "__main__":
    fails = 0
    for fn in (check_f1, check_f2, check_f3, check_f4, check_f5, check_f6, check_f7):
        try:
            fn()
        except Exception as e:                                       # noqa: BLE001
            import traceback
            fails += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
    print("gt cache + targets: " + ("all checks pass" if not fails else f"{fails} FAILED"))
    sys.exit(1 if fails else 0)
