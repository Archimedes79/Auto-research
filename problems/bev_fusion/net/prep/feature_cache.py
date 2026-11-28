"""Cache of the FROZEN camera heads' outputs (derived data), so training skips the camera encoder.

The pre-trained DualBEV image heads (ResNet-50 + neck + DepthNet, frozen) cost 71 GFLOPs per frame; the model needs per
frame only their outputs: the depth distribution (118, 16, 44) and the context features (80, 16, 44) at 256 x 704. They
do not depend on anything trainable, so they are computed ONCE per camera frame and stored (float16, 0.28 MB per frame;
all 98 k front frames of the 510 scenes 27 GB, only the frames the dataset uses about a third).

Storage: one shard per scene in `root` - `<scene>.npy` (float16, (n, 198, 16, 44): 118 probability channels, then 80
context channels) + `<scene>.index.npz` (the sample_data tokens, row order). Files are .npy / .npz: ignored by git, derived
data of this project (rebuilt by this script from the raw nuScenes data and the pre-trained checkpoint; data/README.md). The build is resumable (a scene whose
shard covers its tokens is skipped).

Use:
    store = FeatureStore(root)
    ds = CachedFrontRadarDataset(dataroot, version, scenes, store=store, ...)   # no images are loaded
    loader = DataLoader(ds, batch_size=..., collate_fn=collate_cached)
    model = CachedBEVFusionModel(cfg)                                           # camera_bevs reads batch["prob"], ["ctx"]
Equivalence: the cached model's camera BEVs equal the standard model's within float16 precision (self-test).
Build:  python -P -m bev_fusion.net.prep.build_feature_cache build <dataroot> <version> <root> [--split train|val] [--device cuda] [--limit N]
Tests: tests/test_feature_cache.py
"""
import glob
import os
import time

import numpy as np
import torch
import torch.utils.checkpoint

from bev_fusion.net.prep.nuscenes_dataset import DST_HW, FrontRadarDataset, collate

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

    def __init__(self, *args, store, radar_cache=None, camera_features="dualbev", lane_images=None, store2=None, lane_images_older=False, **kw):
        """lane_images (folder, optional): the key-frame image cache of the lane branch (lane_branch.py, experiment t7); the
        item then carries the current frame as uint8 "lane_img" (3, H, W). lane_images_older (t26, model.student_frames 'all'): the item also carries the
        images of the older frames as uint8 "lane_img_old" (F - 1, 3, H, W), frames 1 .. F - 1 in the order of item["frames"], read from <lane_images>/sweeps
        (training/build_image_cache.py build-cache-older).
        camera_features: a key of model.CAMERAS; its feature sources take the stores in order: `store` for the first, `store2` for the second
        ("dual", t10: the DualBEV cache, then the BEVFusion cache). Every frame carries prob<suffix> / ctx<suffix> of each source and the item
        ida<suffix> of the sources after the first (the first source's transform is `ida`: the DualBEV one, or BEVFusion's when it is the
        BEVFusion family alone)."""
        if lane_images_older and not lane_images:
            raise ValueError("lane_images_older reads <lane_images>/sweeps: it needs lane_images")
        super().__init__(*args, **kw)
        if lane_images_older and self.n_cam < 2:
            raise ValueError("lane_images_older: the window has no older frame (n_cam 1)")
        from bev_fusion.net.prep.cameras import CAMERAS
        self.store, self.radar_cache, self.lane_images, self.store2 = store, radar_cache, lane_images, store2
        self.lane_images_older = bool(lane_images_older)
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
                from bev_fusion.net.prep.images import bevfusion_ida
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
            from bev_fusion.net.prep.image_cache import load_image, older_root
            item["lane_img"] = load_image(self.lane_images, item["frames"][0]["token"])
            if self.lane_images_older:                    # t26 student_frames 'all': the older frames of the window (sweeps), the same crop
                item["lane_img_old"] = torch.stack([load_image(older_root(self.lane_images), f["token"]) for f in item["frames"][1:]])
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
    if "lane_img_old" in items[0]:
        batch["lane_img_old"] = torch.stack([it["lane_img_old"] for it in items])                       # (B, F - 1, 3, H, W) uint8
    return batch
