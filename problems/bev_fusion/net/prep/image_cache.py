"""The key-frame (and older-frame) image cache of the lane branch and the camera student: where the JPEGs are, how a finished build is marked and how it is
built from a dataset. The command line that builds the cache a config names is training/build_image_cache.py."""
import json
import os
import time



JPEG_QUALITY = 95


def image_path(root, sd_token):
    return os.path.join(root, sd_token[:2], sd_token + ".jpg")


def load_image(root, sd_token):
    """uint8 tensor (3, H, W) of the cache (torchvision's JPEG decoder)."""
    from torchvision.io import decode_jpeg, read_file
    p = image_path(root, sd_token)
    if not os.path.exists(p):
        builder = "build-cache-older" if os.path.basename(os.path.normpath(root)) == OLDER else "build-cache"
        raise FileNotFoundError(f"{p}: lane image cache incomplete (python training/build_image_cache.py {builder} <config>)")
    return decode_jpeg(read_file(p))


# sub-folder of the image cache with the OLDER frames of a window (t26, model.student_frames 'all'): same layout, one file per CAM_FRONT sample_data token
OLDER = "sweeps"


# the finished-build marker of a cache folder, written when the builder has been through EVERY sample of its datasets (a crash half-way leaves none):
BUILT = "BUILT.json"


KEY = "key_frames"


def older_root(root):
    return os.path.join(root, OLDER)


def older_key(n_cam, cam_stride):
    """the frames a build covers depend on the window: the older frames of n_cam = 3, cam_stride = 2 (0, -2, -4 sweeps) are other sweeps than those of n_cam =
    2, cam_stride = 4"""
    return f"n_cam={n_cam},cam_stride={cam_stride}"


def _built(folder):
    """{key: {"scenes": [names], "when": ..., "settings": ...}} of the finished builds in <folder>/BUILT.json ({} when there is none)"""
    import json
    try:
        return json.load(open(os.path.join(folder, BUILT)))["built"]
    except (OSError, ValueError, KeyError):
        return {}


def older_built(root):
    return _built(older_root(root))


def key_built(root):
    return _built(root)


def older_missing_scenes(root, n_cam, cam_stride, scenes):
    """the scenes of `scenes` (names) that no finished build of this window covers (all of them when there is none)"""
    return sorted(set(scenes) - set(older_built(root).get(older_key(n_cam, cam_stride), {}).get("scenes", [])))


def key_missing_scenes(root, scenes):
    """the scenes of `scenes` (names) that no finished key-frame build covers (all of them when there is none)"""
    return sorted(set(scenes) - set(key_built(root).get(KEY, {}).get("scenes", [])))


def build_settings():
    """what the builder makes the JPEGs with, recorded in the marker: the quality and the resize + crop of the dataset's own image
    (nuscenes_dataset._image_uint8); a cache made with other ones is
    not the cache of this code"""
    from net.prep.images import nuscenes_front_ida
    from net.prep.nuscenes_dataset import DST_HW
    return {"jpeg_quality": JPEG_QUALITY, "hw": list(DST_HW), "ida": [round(float(x), 6) for x in nuscenes_front_ida(dst_hw=DST_HW).flatten().tolist()]}


def older_records(ds, sample_token):
    """the CAM_FRONT sample_data records of the older frames (k = 1 .. n_cam - 1) of one sample: the track of the scene, cam_stride records apart, clamped at
    its start
    (the rule of nuscenes_dataset._cam_frames and feature_cache.used_frames, so these are the frames the teacher caches hold)"""
    track = ds.gt.camera_track(sample_token)
    i0 = [x["token"] for x in track].index(ds.gt.cam(sample_token)["token"])
    return [track[max(0, i0 - k * ds.cam_stride)] for k in range(1, ds.n_cam)]


def scene_names(ds):
    """the names of the scenes the samples of a dataset belong to"""
    return {ds.nusc.get("scene", ds.nusc.get("sample", t)["scene_token"])["name"] for t in ds.samples}


_DS = None                                                # dataset of build_cache, inherited by the forked workers


def _write(root, records):
    """the JPEG of every CAM_FRONT sample_data record that the cache folder `root` does not hold yet, in the dataset's own crop"""
    from PIL import Image
    for sd in records:
        p = image_path(root, sd["token"])
        if os.path.exists(p):
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        arr = _DS._image_uint8(sd)
        tmp = p[:-4] + f".tmp{os.getpid()}.jpg"
        Image.fromarray(arr).save(tmp, quality=JPEG_QUALITY)
        os.replace(tmp, p)                                 # atomic: a reader never sees a partial file


def _fill(args):
    """the current (key) frame of the samples into <root>"""
    root, tokens = args
    _write(root, [_DS.gt.cam(t) for t in tokens])
    return len(tokens)


def _fill_older(args):
    """the older frames of the samples (older_records) into <root>/sweeps/: every record, also one that is a key frame (a clamped frame at the start of a
    scene)"""
    root, tokens = args
    _write(older_root(root), [sd for t in tokens for sd in older_records(_DS, t)])
    return len(tokens)


def _mark_built(folder, key, datasets):
    """<folder>/BUILT.json: the scenes of these datasets are now complete in the cache folder `folder` under `key` (merged with the scenes of earlier builds).
    The read-modify-write is under a lock file
    (flock, released when the process dies), so two builds of one folder do not lose each other's update"""
    import datetime
    import fcntl
    import json
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, BUILT + ".lock"), "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
        except OSError:                                    # a file system without locks (not seen on WSL): one writer at a time is then the caller's business
            pass
        built = _built(folder)
        scenes = set(built.get(key, {}).get("scenes", [])).union(*(scene_names(ds) for ds in datasets))
        built[key] = {"scenes": sorted(scenes), "when": datetime.datetime.now().isoformat(timespec="seconds"), "settings": build_settings()}
        path = os.path.join(folder, BUILT)
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump({"built": built}, fh, indent=1)
        os.replace(tmp, path)


def build_cache(datasets, root, workers=1, older=False):
    """Write the current CAM_FRONT frame of every sample of the datasets (resumable); older=True: the older frames of the window into <root>/sweeps/ instead.
    The finished-build marker
    (BUILT.json of that folder) is written once every sample is through."""
    import multiprocessing as mp
    import time
    global _DS
    fill = _fill_older if older else _fill
    t0, n = time.time(), 0
    for ds in datasets:
        _DS = ds
        chunks = [ds.samples[i::max(workers, 1) * 16] for i in range(max(workers, 1) * 16)]
        if workers <= 1:                                   # one process: no forked copies of the devkit (shared box, 2026-10-30:
            done = map(fill, [(root, c) for c in chunks])   # two forked workers + the devkit ran the box out of memory)
            for k in done:
                n += k
                print(f"lane image cache{' (older frames)' if older else ''}: {n} samples, {time.time() - t0:.0f} s", flush=True)
            continue
        with mp.get_context("fork").Pool(workers) as pool:
            for k in pool.imap_unordered(fill, [(root, c) for c in chunks]):
                n += k
                print(f"lane image cache{' (older frames)' if older else ''}: {n} samples, {time.time() - t0:.0f} s", flush=True)
    # the finished-build mark (training/build_image_cache.py checks it; student_frames 'all'): a half-built cache is not started on
    if datasets:
        _mark_built(older_root(root), older_key(datasets[0].n_cam, datasets[0].cam_stride), datasets) if older else _mark_built(root, KEY, datasets)
    return n
