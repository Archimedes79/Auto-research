"""Cache of the FROZEN camera heads' outputs (derived data), so training skips the camera encoder.

The pre-trained DualBEV image heads (ResNet-50 + neck + DepthNet, frozen) cost 71 GFLOPs per frame; the model needs per
frame only their outputs: the depth distribution (118, 16, 44) and the context features (80, 16, 44) at 256 x 704. They
do not depend on anything trainable, so they are computed ONCE per camera frame and stored (float16, 0.28 MB per frame;
all 98 k front frames of the 510 scenes 27 GB, only the frames the dataset uses about a third).

Storage: one shard per scene in `root` - `<scene>.npy` (float16, (n, 198, 16, 44): 118 probability channels, then 80
context channels) + `<scene>.index.npz` (the sample_data tokens, row order). Files are .npy / .npz: ignored by git, derived
data of this project (rebuilt by this script from the raw nuScenes data and the pre-trained checkpoint; data/README.md,
data/rebuild.json). The build is resumable (a scene whose shard covers its tokens is skipped).

Use:
    store = FeatureStore(root)
    ds = CachedFrontRadarDataset(dataroot, version, scenes, store=store, ...)   # no images are loaded
    loader = DataLoader(ds, batch_size=..., collate_fn=collate_cached)
    model = CachedBEVFusionModel(cfg)                                           # camera_bevs reads batch["prob"], ["ctx"]
Equivalence: the cached model's camera BEVs equal the standard model's within float16 precision (self-test).
Build:  python training/feature_cache.py build <dataroot> <version> <root> [--split train|val] [--device cuda] [--limit N]
Self-test (nuScenes mini, CPU): python training/feature_cache.py
"""
import glob
import os
import sys
import time

import numpy as np
import torch

try:
    from .camera_encoder import FrontCameraEncoder, camera_vector
    from .gt_bev import pose_matrix
    from .nuscenes_dataset import DST_HW, FrontRadarDataset, available_scenes, collate
except ImportError:                                   # run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from camera_encoder import FrontCameraEncoder, camera_vector
    from gt_bev import pose_matrix
    from nuscenes_dataset import DST_HW, FrontRadarDataset, available_scenes, collate

N_PROB, N_CTX, FEAT_HW = 118, 80, (16, 44)
CHECKPOINT = "models/pretrained/dualbev-r50-cbgs.pth"


class FeatureStore:
    """Read access to the shards of `root` (token -> (probability (118, 16, 44), context (80, 16, 44)), float16)."""

    def __init__(self, root):
        self.root, self.index, self.shards, self._mm = root, {}, [], {}
        for f in sorted(glob.glob(os.path.join(root, "*.index.npz"))):
            toks = np.load(f)["tokens"]
            sid = len(self.shards)
            self.shards.append(f[:-len(".index.npz")] + ".npy")
            for r, t in enumerate(toks):
                self.index[str(t)] = (sid, r)

    def __len__(self):
        return len(self.index)

    def has(self, token):
        return token in self.index

    def get(self, token):
        sid, r = self.index[token]
        if sid not in self._mm:
            self._mm[sid] = np.load(self.shards[sid], mmap_mode="r")
        a = torch.from_numpy(np.array(self._mm[sid][r]))
        return a[:N_PROB], a[N_PROB:]


def used_frames(ds, sample_tokens=None):
    """scene name -> set of CAM_FRONT sample_data tokens the dataset reads for these samples (same rule as `_cam_frames`)."""
    out = {}
    for t in (sample_tokens if sample_tokens is not None else ds.samples):
        track = ds.gt.camera_track(t)
        cur = ds.gt.cam(t)
        i0 = [x["token"] for x in track].index(cur["token"])
        scene = ds.nusc.get("scene", ds.nusc.get("sample", t)["scene_token"])["name"]
        for k in range(ds.n_cam):
            out.setdefault(scene, set()).add(track[max(0, i0 - k * ds.cam_stride)]["token"])
    return out


def frame_inputs(ds, sd_token):
    """The network input of one camera frame, exactly as the dataset builds it: normalised image and 27-value camera vector."""
    sd = ds.nusc.get("sample_data", sd_token)
    cs = ds.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K = torch.tensor(cs["camera_intrinsic"], dtype=torch.float32)
    T_c2e = torch.tensor(pose_matrix(cs), dtype=torch.float32)
    cv = camera_vector(K[None], ds.ida[None, :2, :2], ds.ida[None, :2, 2], T_c2e[None])[0]
    return ds._image(sd), cv


def build(ds, root, sample_tokens=None, device="cpu", batch=16, checkpoint=CHECKPOINT, verbose=True):
    """Compute and store the frozen heads' outputs for every camera frame the dataset uses (resumable per scene)."""
    os.makedirs(root, exist_ok=True)
    camera = FrontCameraEncoder()
    rep = camera.load_pretrained(checkpoint)
    if rep["missing"] or rep["shape_mismatch"]:
        raise RuntimeError(f"camera weights do not fit: {rep}")
    camera.freeze_camera_heads()
    camera = camera.to(device).eval()
    todo = used_frames(ds, sample_tokens)
    n_done, t0 = 0, time.time()
    for scene, tokens in sorted(todo.items()):
        idx_path = os.path.join(root, scene + ".index.npz")
        if os.path.exists(idx_path) and set(np.load(idx_path)["tokens"].tolist()) >= tokens:
            continue
        toks = sorted(tokens)
        arr = np.zeros((len(toks), N_PROB + N_CTX) + FEAT_HW, np.float16)
        for i in range(0, len(toks), batch):
            chunk = toks[i:i + batch]
            ins = [frame_inputs(ds, t) for t in chunk]
            img, cv = torch.stack([a for a, _ in ins]).to(device), torch.stack([b for _, b in ins]).to(device)
            with torch.no_grad():
                prob, ctx = camera.image_features(img, cv)
            arr[i:i + len(chunk)] = torch.cat([prob, ctx], 1).cpu().numpy().astype(np.float16)
        np.save(os.path.join(root, scene + ".npy"), arr)
        np.savez(idx_path, tokens=np.array(toks, dtype="U40"))
        n_done += len(toks)
        if verbose:
            print(f"{scene}: {len(toks)} frames ({n_done} done, {time.time() - t0:.0f} s)", flush=True)
    return FeatureStore(root)


def radar_signature():
    """md5 prefix of FrontRadarDataset._radar's source: a radar cache entry is only reused by the same radar code."""
    import hashlib
    import inspect
    return hashlib.md5(inspect.getsource(FrontRadarDataset._radar).encode()).hexdigest()[:10]


class CachedFrontRadarDataset(FrontRadarDataset):
    """FrontRadarDataset whose camera frames carry cached head outputs instead of images (no image is read or resized).
    radar_cache (folder, optional): the radar detections of a sample (all sweeps of the window, moved to the camera time)
    are stored once per sample as .npz and read back exactly (float32 / int64 as computed): reading the ~35 sweep files of a
    sample costs ~0.3-0.6 s, the cached entry ~5 ms (t1, 2026-10-26: the loader was the bottleneck of training)."""

    def __init__(self, *args, store, radar_cache=None, camera_features="dualbev", **kw):
        super().__init__(*args, **kw)
        self.store, self.radar_cache = store, radar_cache
        if camera_features == "bevfusion":                 # the cache was made with BEVFusion's test pipeline (scale 0.48, crop 32 / 176)
            try:
                from .bevfusion_camera import bevfusion_ida
            except ImportError:
                from bevfusion_camera import bevfusion_ida
            self.ida = bevfusion_ida()
        elif camera_features != "dualbev":
            raise ValueError(camera_features)
        self._rsig = radar_signature() if radar_cache else None

    def _image(self, sd):
        return torch.zeros(1)                              # placeholder: the cached features replace the image

    def _radar_path(self, sample, cur_sd):
        return os.path.join(self.radar_cache, f"{sample['token']}_{cur_sd['token'][:8]}_{self.radar_window}_"
                                              f"{int(self.doppler_shift)}_{self._rsig}.npz")

    def _radar(self, sample, cur_sd):
        if not self.radar_cache:
            return super()._radar(sample, cur_sd)
        path = self._radar_path(sample, cur_sd)
        if os.path.exists(path):
            z = np.load(path)
            return {k: torch.from_numpy(z[k]) for k in z.files}
        out = super()._radar(sample, cur_sd)
        os.makedirs(self.radar_cache, exist_ok=True)
        tmp = path[:-4] + f".tmp{os.getpid()}.npz"
        np.savez(tmp, **{k: v.numpy() for k, v in out.items()})
        os.replace(tmp, path)                              # atomic: a reader never sees a partial file
        return out

    def __getitem__(self, i):
        item = super().__getitem__(i)
        for f in item["frames"]:
            f["prob"], f["ctx"] = self.store.get(f["token"])
        return item


_FILL_DS = None                                           # the dataset of build_radar_cache, inherited by the forked workers


def _radar_fill(tokens):
    for t in tokens:
        _FILL_DS._radar(_FILL_DS.nusc.get("sample", t), _FILL_DS.gt.cam(t))
    return len(tokens)


def build_radar_cache(ds, workers=4):
    """fill ds.radar_cache for every sample of ds (resumable: existing entries are kept), `workers` processes."""
    import multiprocessing as mp
    global _FILL_DS
    _FILL_DS = ds
    todo = [t for t in ds.samples if not os.path.exists(ds._radar_path(ds.nusc.get("sample", t), ds.gt.cam(t)))]
    chunks = [todo[i::workers * 8] for i in range(workers * 8)]
    t0, n = time.time(), 0
    with mp.get_context("fork").Pool(workers) as pool:
        for k in pool.imap_unordered(_radar_fill, chunks):
            n += k
            print(f"radar cache: {n}/{len(todo)} samples, {time.time() - t0:.0f} s", flush=True)
    return len(todo)


def collate_cached(items):
    batch = collate(items)
    batch["prob"] = torch.stack([torch.stack([f["prob"] for f in it["frames"]]) for it in items])      # (B, F, 118, 16, 44)
    batch["ctx"] = torch.stack([torch.stack([f["ctx"] for f in it["frames"]]) for it in items])        # (B, F, 80, 16, 44)
    batch["hw"] = DST_HW
    return batch


def CachedBEVFusionModel(cfg=None):
    """BEVFusionModel whose camera branch reads batch["prob"], batch["ctx"] (the frozen heads are not instantiated)."""
    try:
        from .model import BEVFusionModel, ModelConfig
    except ImportError:
        from model import BEVFusionModel, ModelConfig
    cfg = cfg or ModelConfig()

    class _Cached(BEVFusionModel):
        def camera_bevs(self, batch):
            prob, ctx = batch["prob"].float(), batch["ctx"].float()
            B, Fn = prob.shape[:2]
            if self.cfg.camera_off:                                     # radar-only system (was ignored here until 2026-10-30:
                return torch.zeros(B, Fn, self.cfg.c_cam, self.grid.nx, self.grid.ny, device=prob.device)   # t2_cam_off ran WITH camera)
            if self.cam_adapter is not None:
                ctx = self.cam_adapter(ctx.flatten(0, 1)).view_as(ctx)
            bevs = []
            for b in range(B):                                          # one calibration per sample (scenes differ)
                vol = self.lift(ctx[b], prob[b], batch["K"][b, 0], batch["T_cam_car"][b, 0], batch["hw"], batch["ida"])
                bevs.append(self.cam_reduce(vol))
            return torch.stack(bevs)

    import dataclasses
    m = _Cached(dataclasses.replace(cfg, pretrained=""))                  # no checkpoint: the heads are dropped below
    del m.camera
    return m


def _selftest_camera_off():
    """the flags of ModelConfig must reach the cached model: camera_off gives a zero camera BEV and never reads the features."""
    m = CachedBEVFusionModel(__import__("dataclasses").replace(_model_cfg(), camera_off=True)).eval()
    batch = {"prob": torch.rand(2, 3, N_PROB, *FEAT_HW), "ctx": torch.rand(2, 3, N_CTX, *FEAT_HW)}
    out = m.camera_bevs(batch)
    assert out.shape == (2, 3, m.cfg.c_cam, m.grid.nx, m.grid.ny) and float(out.abs().sum()) == 0.0, "camera_off ignored by the cached model"
    print("camera_off selftest ok: zero camera BEV from the cached model")


def _model_cfg():
    try:
        from .model import ModelConfig
    except ImportError:
        from model import ModelConfig
    return ModelConfig()


def _selftest():
    import tempfile
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))                 # the bev_fusion folder
    try:
        from .model import BEVFusionModel, ModelConfig
    except ImportError:
        from model import BEVFusionModel, ModelConfig
    torch.manual_seed(0)
    root_nusc = os.path.expanduser("~/data/nuscenes")
    ds = FrontRadarDataset(root_nusc, "v1.0-mini", n_cam=3, cam_stride=2, with_gt=False)
    tokens = [ds.samples[10], ds.samples[200]]
    used = used_frames(ds, tokens)
    n_frames = sum(len(v) for v in used.values())
    assert 3 <= n_frames <= 6, n_frames
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        store = build(ds, tmp, tokens, device="cpu", batch=4, verbose=False)
        assert len(store) == n_frames and all(store.has(t) for v in used.values() for t in v)
        # storage: float16, shapes, probabilities sum to one
        t = next(iter(used.values())).copy().pop()
        prob, ctx = store.get(t)
        assert prob.dtype == torch.float16 and prob.shape == (N_PROB,) + FEAT_HW and ctx.shape == (N_CTX,) + FEAT_HW
        assert torch.allclose(prob.float().sum(0), torch.ones(FEAT_HW), atol=2e-2), "probabilities sum to one"
        build(ds, tmp, tokens, device="cpu", verbose=False)                                  # resumable: nothing recomputed
        # equivalence: the cached model's camera BEVs = the standard model's (same weights downstream)
        cds = CachedFrontRadarDataset(root_nusc, "v1.0-mini", store=store, n_cam=3, cam_stride=2, with_gt=False)
        ia, ib = ds.samples.index(tokens[0]), ds.samples.index(tokens[1])
        std = BEVFusionModel().eval()
        cached = CachedBEVFusionModel().eval()
        missing = cached.load_state_dict({k: v for k, v in std.state_dict().items() if not k.startswith("camera.")},
                                         strict=False)
        assert not missing.unexpected_keys and not missing.missing_keys, "the cached model has every non-camera weight"
        b_std = collate([ds[ia], ds[ib]])
        b_cac = collate_cached([cds[ia], cds[ib]])
        assert b_cac["img"].numel() < b_std["img"].numel() / 1e4, "no images are loaded"
        with torch.no_grad():
            bev_std, bev_cac = std.camera_bevs(b_std), cached.camera_bevs(b_cac)
        err = (bev_std - bev_cac).abs().max().item()
        scale = bev_std.abs().max().item()
        assert bev_std.shape == bev_cac.shape and err < 0.02 * scale + 1e-4, f"cached vs standard: {err:.4g} of {scale:.4g}"
        # the whole chain runs from the cached batch and gives the same outputs as the standard model
        with torch.no_grad():
            o_std = std(b_std)["out"][0.0]["sem"]
            o_cac = cached(b_cac)["out"][0.0]["sem"]
        err_o = (o_std - o_cac).abs().max().item()
        assert err_o < 0.05 * max(1.0, o_std.abs().max().item()), f"outputs: {err_o:.4g}"
        n_cached = sum(p.numel() for p in cached.parameters())
        n_std = sum(p.numel() for p in std.parameters())
        print(f"feature_cache selftest ok ({time.time() - t0:.0f} s): {n_frames} frames cached as float16 (0.28 MB each), "
              f"resumable, probabilities sum to 1, no images read, camera BEVs equal the standard model's within "
              f"{err / scale:.1e} of their range, chain outputs within {err_o:.1e}; the cached model has "
              f"{(n_std - n_cached) / 1e6:.1f} M fewer parameters (frozen heads not instantiated)")


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
            cds.store = store
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


def _cli(argv):
    import argparse
    if argv[:1] == ["camera-off-selftest"]:
        return _selftest_camera_off()
    if argv[:1] == ["radar-selftest"]:
        return _selftest_radar()
    p = argparse.ArgumentParser(description="build the frozen-head feature cache (build) or the radar cache (radar: root = "
                                            "the radar cache folder)")
    p.add_argument("cmd", choices=["build", "radar"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("dataroot")
    p.add_argument("version")
    p.add_argument("root")
    p.add_argument("--split", default=None, choices=[None, "train", "val"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=None, help="only the first N samples (for tests)")
    p.add_argument("--n-cam", type=int, default=3)
    p.add_argument("--cam-stride", type=int, default=2)
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args(argv)
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(a.version, dataroot=a.dataroot, verbose=False)
    scenes = available_scenes(nusc, a.split) if a.split else None
    if a.cmd == "radar":
        ds = CachedFrontRadarDataset(a.dataroot, a.version, scenes=scenes, store=None, radar_cache=a.root, n_cam=a.n_cam,
                                     cam_stride=a.cam_stride, with_gt=False, nusc=nusc)
        if a.limit:
            ds.samples = ds.samples[:a.limit]
        n = build_radar_cache(ds, a.workers)
        print(f"radar cache: {n} samples written to {a.root}; all done")
        return
    ds = FrontRadarDataset(a.dataroot, a.version, scenes=scenes, n_cam=a.n_cam, cam_stride=a.cam_stride, with_gt=False, nusc=nusc)
    tokens = ds.samples[:a.limit] if a.limit else None
    store = build(ds, a.root, tokens, device=a.device, batch=a.batch)
    print(f"cache: {len(store)} frames in {a.root}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _cli(sys.argv[1:])
    else:
        _selftest()
