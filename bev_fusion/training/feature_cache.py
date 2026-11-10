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
import torch.utils.checkpoint

try:
    from .lift_pd import lift_key
    from .camera_encoder import FrontCameraEncoder, camera_vector
    from .nuscenes_dataset import DST_HW, FrontRadarDataset, available_scenes, collate, pose_matrix
except ImportError:                                   # run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lift_pd import lift_key
    from camera_encoder import FrontCameraEncoder, camera_vector
    from nuscenes_dataset import DST_HW, FrontRadarDataset, available_scenes, collate, pose_matrix

N_PROB, N_CTX, FEAT_HW = 118, 80, (16, 44)
CHECKPOINT = "models/pretrained/dualbev-r50-cbgs.pth"


FEAT_HW_OF = {"dualbev": FEAT_HW, "bevfusion": (32, 88)}    # feature size of each camera_features cache (bevfusion:
                                                            # cache_bevfusion_features.py, stride 8 of the 256 x 704 crop)


class FeatureStore:
    """Read access to the shards of `root` (token -> (probability (118, h, w), context (80, h, w)), float16)."""

    def __init__(self, root):
        self.root, self.index, self.shards, self.counts, self._mm = root, {}, [], [], {}
        for f in sorted(glob.glob(os.path.join(root, "*.index.npz"))):
            toks = np.load(f)["tokens"]
            sid = len(self.shards)
            self.shards.append(f[:-len(".index.npz")] + ".npy")
            self.counts.append(len(toks))
            for r, t in enumerate(toks):
                self.index[str(t)] = (sid, r)

    def __len__(self):
        return len(self.index)

    def has(self, token):
        return token in self.index

    def feature_hw(self):
        """(h, w) of the stored features (header of the first shard), None for an empty store"""
        return tuple(np.load(self.shards[0], mmap_mode="r").shape[2:]) if self.shards else None

    def get(self, token):
        sid, r = self.index[token]
        if sid not in self._mm:
            mm = np.load(self.shards[sid], mmap_mode="r")
            if mm.shape[0] != self.counts[sid]:
                raise RuntimeError(f"{self.shards[sid]} has {mm.shape[0]} rows, its index names {self.counts[sid]} tokens "
                                   f"(a build was cut between the two files): rebuild that scene")
            self._mm[sid] = mm
        a = torch.from_numpy(np.array(self._mm[sid][r]))
        return a[:N_PROB], a[N_PROB:]


def check_feature_hw(store, camera_features, name="feature_cache"):
    """refuses a cache whose feature size is not the one of camera_features (e.g. the DualBEV and BEVFusion caches swapped):
    the lift would otherwise use the wrong camera geometry on the features without any error"""
    hw, want = store.feature_hw(), FEAT_HW_OF[camera_features]
    if hw is not None and hw != want:
        raise ValueError(f"{name} {store.root} holds {hw} features, camera_features {camera_features!r} needs {want}")


def write_shard(root, scene, arr, toks):
    """<scene>.npy and <scene>.index.npz through temporary names and os.replace, the index last: a reader never sees a half-written
    file, and a build cut between the two replaces leaves a row count that FeatureStore.get refuses (a scene rebuilt with the same
    number of other tokens would still pair silently, so a rebuild should not be cut)"""
    npy, idx = os.path.join(root, scene + ".npy"), os.path.join(root, scene + ".index.npz")
    tmp = os.path.join(root, f".{scene}.tmp{os.getpid()}")          # no *.index.npz name: FeatureStore never lists a temporary file
    try:
        with open(tmp + "_npy", "wb") as f:
            np.save(f, arr)
        with open(tmp + "_idx", "wb") as f:
            np.savez(f, tokens=np.array(toks, dtype="U40"))
        os.replace(tmp + "_npy", npy)
        os.replace(tmp + "_idx", idx)
    finally:
        for t in (tmp + "_npy", tmp + "_idx"):
            if os.path.exists(t):
                os.remove(t)


def shard_complete(root, scene, tokens):
    """True if the scene's shard holds `tokens` (a set) and its row count equals its index length: a build skips only such a scene,
    so a shard that FeatureStore.get refuses (a cut build) is rebuilt instead of skipped"""
    idx = os.path.join(root, scene + ".index.npz")
    try:
        toks = np.load(idx)["tokens"]
        return set(toks.tolist()) >= tokens and np.load(os.path.join(root, scene + ".npy"), mmap_mode="r").shape[0] == len(toks)
    except (OSError, ValueError, KeyError):
        return False


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
        if shard_complete(root, scene, tokens):
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
        write_shard(root, scene, arr, toks)
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

    def __init__(self, *args, store, radar_cache=None, camera_features="dualbev", lane_images=None, store2=None, **kw):
        """lane_images (folder, optional): the key-frame image cache of the lane branch (lane_branch.py, experiment t7); the
        item then carries the current frame as uint8 "lane_img" (3, H, W).
        camera_features: a key of model.CAMERAS; its feature sources take the stores in order: `store` for the first, `store2` for the second
        ("dual", t10: the DualBEV cache, then the BEVFusion cache). Every frame carries prob<suffix> / ctx<suffix> of each source and the item
        ida<suffix> of the sources after the first (the first source's transform is `ida`: the DualBEV one, or BEVFusion's when it is the
        BEVFusion family alone)."""
        super().__init__(*args, **kw)
        try:
            from .model import CAMERAS
        except ImportError:
            from model import CAMERAS
        self.store, self.radar_cache, self.lane_images, self.store2 = store, radar_cache, lane_images, store2
        if camera_features not in CAMERAS:
            raise ValueError(camera_features)
        self.sources = CAMERAS[camera_features]
        if (store2 is not None) != (len(self.sources) > 1):
            raise ValueError(f"camera_features {camera_features!r} has {len(self.sources)} feature source(s): store2 is needed for exactly "
                             "the kinds with two ('dual')")
        self.stores = [store, store2][:len(self.sources)]
        for src, st in zip(self.sources, self.stores):
            if st is not None:
                check_feature_hw(st, src.features, src.cache)
            if src.features == "bevfusion":                  # the cache was made with BEVFusion's test pipeline (scale 0.48, crop 32 / 176)
                try:
                    from .bevfusion_camera import bevfusion_ida
                except ImportError:
                    from bevfusion_camera import bevfusion_ida
                setattr(self, "ida" + src.suffix, bevfusion_ida())      # ida (alone) or ida2 (second source); the DualBEV family keeps self.ida
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
            try:
                with np.load(path) as z:
                    return {k: torch.from_numpy(z[k]) for k in z.files}
            except Exception as e:                         # noqa: BLE001 - an empty or cut file (a replace on /mnt/c is not atomic
                print(f"radar cache {os.path.basename(path)} unreadable ({type(e).__name__}), rebuilt", flush=True)  # for a reader)
        out = super()._radar(sample, cur_sd)
        os.makedirs(self.radar_cache, exist_ok=True)
        tmp = path[:-4] + f".tmp{os.getpid()}.npz"
        try:
            np.savez(tmp, **{k: v.numpy() for k, v in out.items()})
            os.replace(tmp, path)                          # atomic for the writer: no half-written file under the final name
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return out

    def __getitem__(self, i):
        item = super().__getitem__(i)
        for f in item["frames"]:
            for src, st in zip(self.sources, self.stores):
                f["prob" + src.suffix], f["ctx" + src.suffix] = st.get(f["token"])
        for src in self.sources[1:]:
            item["ida" + src.suffix] = getattr(self, "ida" + src.suffix)
        if self.lane_images:
            try:
                from .lane_branch import load_image
            except ImportError:
                from lane_branch import load_image
            item["lane_img"] = load_image(self.lane_images, item["frames"][0]["token"])
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
    for k in items[0]["frames"][0]:                    # one prob / ctx pair per feature source of the camera slot: "", then "2" (model.CAMERAS)
        if k == "prob" or (k.startswith("prob") and k[len("prob"):].isdigit()):
            s = k[len("prob"):]
            batch["prob" + s] = torch.stack([torch.stack([f["prob" + s] for f in it["frames"]]) for it in items])   # (B, F, 118, h, w)
            batch["ctx" + s] = torch.stack([torch.stack([f["ctx" + s] for f in it["frames"]]) for it in items])     # (B, F, 80, h, w)
            if s:
                batch["ida" + s] = items[0]["ida" + s]
    batch["hw"] = DST_HW
    if "lane_img" in items[0]:
        batch["lane_img"] = torch.stack([it["lane_img"] for it in items])                               # (B, 3, H, W) uint8
    return batch


def CachedBEVFusionModel(cfg=None):
    """BEVFusionModel whose camera branch reads batch["prob"], batch["ctx"] (the frozen heads are not instantiated)."""
    try:
        from .model import CAMERAS, BEVFusionModel, ModelConfig
    except ImportError:
        from model import CAMERAS, BEVFusionModel, ModelConfig
    cfg = cfg or ModelConfig()

    class _Cached(BEVFusionModel):
        CACHED = True

        def _run_student(self, batch):
            """camera_student (t22): the student on the current key-frame image batch["lane_img"] (uint8, B x 3 x 256 x 704) -> its output dict; no graph unless it trains"""
            if "lane_img" not in batch:
                raise KeyError("camera_student reads the current key-frame image batch['lane_img']: set TrainConfig.lane_images (the folder of the key-frame JPEGs, data/cache/img_front)")
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.cfg.student_trainable):
                return self.student(batch["lane_img"])

        def camera_bevs(self, batch):
            prob = batch["prob"].float()
            B, Fn = prob.shape[:2]
            if self.cfg.camera_off:                                     # radar-only system (was ignored here until 2026-10-30:
                return torch.zeros(B, Fn, self.cfg.c_cam, self.grid.nx, self.grid.ny, device=prob.device)   # t2_cam_off ran WITH camera)
            stud, side = (self._run_student(batch) if self.student is not None else None), {}
            recompute_lift = stud is not None and self.cfg.student_trainable and self.training and torch.is_grad_enabled()
            bevs = []
            for i, src in enumerate(CAMERAS[self.cfg.camera_features]):  # one lift + collapse per feature source, each with its own ida
                s = src.suffix
                p, ctx = (prob if i == 0 else batch["prob" + s].float()), batch["ctx" + s].float()
                if stud is not None:                                    # t22: frame 0 from the student, the older frames keep their cached teacher outputs; `side` for the anchor loss
                    h = self.student.HEAD_OF[src.features]
                    side.update({"p" + h: stud["prob" + h], "c" + h: stud["ctx" + h], "l" + h: stud["logit" + h], "t_p" + h: p[:, 0].detach(), "t_c" + h: ctx[:, 0].detach()})
                    p = torch.cat([stud["prob" + h].float()[:, None], p[:, 1:]], 1)
                    ctx = torch.cat([stud["ctx" + h].float()[:, None], ctx[:, 1:]], 1)
                if i == 0 and self.cam_adapter is not None:
                    ctx = self.cam_adapter(ctx.flatten(0, 1)).view_as(ctx)
                lift, collapse = getattr(self, "lift" + s), getattr(self, "cam_reduce" + s)
                keys = self._lift_keys(batch, batch["hw"], "ida" + s)    # D54: the projection-table cache keys from the CPU copies in batch["host"] (None: the lift makes them)
                vols = []
                for b in range(B):                                      # one calibration per sample (scenes differ)
                    args = (ctx[b], p[b], batch["K"][b, 0], batch["T_cam_car"][b, 0], batch["hw"], batch["ida" + s])
                    key = None if keys is None else keys[b]
                    if recompute_lift:      # t22: the student's depth + context carry a graph into the lift; without this the 118 x C frustum features of the 9 frames of a micro-batch stay in memory
                        if key is None:     # the key the lift would make from the GPU tensors (three waits for the GPU): made once here, so that no recomputation inside a backward has to wait
                            key = lift_key(*args[2:])
                        vols.append(collapse(torch.utils.checkpoint.checkpoint(lift, *args, key=key, use_reentrant=False)))
                    else:
                        vols.append(collapse(lift(*args, key=key)))
                bevs.append(torch.stack(vols))
            self._student_side = side or None
            if len(bevs) == 1:
                return bevs[0]
            return self.cam_merge(torch.cat(bevs, 2).flatten(0, 1)).view_as(bevs[0])

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
    _selftest_shards()
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


def _cli(argv):
    import argparse
    if argv[:1] == ["shards-selftest"]:
        return _selftest_shards()
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
