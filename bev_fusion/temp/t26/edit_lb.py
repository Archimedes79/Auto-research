p = "training/lane_branch.py"
s = open(p).read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''BUILT = "BUILT.json"                                      # written into <root>/sweeps/ when build-cache-older has been through EVERY sample of its datasets: {"built": {"n_cam=3,cam_stride=2": {"scenes": [...], "when": ...}}}
''', '''BUILT = "BUILT.json"                                      # the finished-build marker of a cache folder, written when the builder has been through EVERY sample of its datasets (a crash half-way leaves none):
                                                          # <root>/sweeps/BUILT.json for the older frames, key = the window "n_cam=3,cam_stride=2"; <root>/BUILT.json for the key frames, key KEY. Content:
                                                          # {"built": {key: {"scenes": [names], "when": ..., "settings": build_settings()}}}, the scenes merged over builds
KEY = "key_frames"
''')
rep('''def older_built(root):
    """{older_key: {"scenes": [names], "when": ...}} of the finished builds in <root>/sweeps/BUILT.json ({} when there is none)"""
    import json
    try:
        return json.load(open(os.path.join(older_root(root), BUILT)))["built"]
    except (OSError, ValueError, KeyError):
        return {}


def older_missing_scenes(root, n_cam, cam_stride, scenes):
    """the scenes of `scenes` (names) that no finished build of this window covers (all of them when there is none)"""
    return sorted(set(scenes) - set(older_built(root).get(older_key(n_cam, cam_stride), {}).get("scenes", [])))
''', '''def _built(folder):
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
    """what the builder makes the JPEGs with, recorded in the marker: the quality and the resize + crop of the dataset's own image (nuscenes_dataset._image_uint8); a cache made with other ones is
    not the cache of this code"""
    from camera_encoder import nuscenes_front_ida
    from nuscenes_dataset import DST_HW
    return {"jpeg_quality": JPEG_QUALITY, "hw": list(DST_HW), "ida": [round(float(x), 6) for x in nuscenes_front_ida(dst_hw=DST_HW).flatten().tolist()]}
''')
rep('''def _mark_built(root, datasets):
    """BUILT.json: the scenes of these datasets now have every older frame of their window in <root>/sweeps/ (merged with the scenes of earlier builds of the same window)"""
    import datetime
    import json
    built, key = older_built(root), older_key(datasets[0].n_cam, datasets[0].cam_stride)
    scenes = set(built.get(key, {}).get("scenes", [])).union(*(scene_names(ds) for ds in datasets))
    built[key] = {"scenes": sorted(scenes), "when": datetime.datetime.now().isoformat(timespec="seconds")}
    os.makedirs(older_root(root), exist_ok=True)
    path = os.path.join(older_root(root), BUILT)
    tmp = path + f".tmp{os.getpid()}"
    json.dump({"built": built}, open(tmp, "w"), indent=1)
    os.replace(tmp, path)
''', '''def _mark_built(folder, key, datasets):
    """<folder>/BUILT.json: the scenes of these datasets are now complete in the cache folder `folder` under `key` (merged with the scenes of earlier builds). The read-modify-write is under a lock file
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
''')
rep('''    """Write the current CAM_FRONT frame of every sample of the datasets (resumable); older=True: the older frames of the window into <root>/sweeps/ instead, and the
    finished-build marker once every sample is through."""''', '''    """Write the current CAM_FRONT frame of every sample of the datasets (resumable); older=True: the older frames of the window into <root>/sweeps/ instead. The finished-build marker
    (BUILT.json of that folder) is written once every sample is through."""''')
rep('''    if older and datasets:
        _mark_built(root, datasets)                        # startable.py asks for this: a half-built cache is not started on
    return n''', '''    if datasets:                                           # startable.py asks for this (student_frames 'all'): a half-built cache is not started on
        _mark_built(older_root(root), older_key(datasets[0].n_cam, datasets[0].cam_stride), datasets) if older else _mark_built(root, KEY, datasets)
    return n''')
rep('''    cfg = load_config(a.config)
    cfg.lane_images = ""                                   # the datasets must not try to read the cache they build
''', '''    cfg = load_config(a.config)
    cfg.lane_images = cfg.feature_cache = cfg.feature_cache2 = ""     # the builders need the plain datasets: no cache to read (the one they build, the teachers' features), no student to feed
''')
rep('''    if older:
        missing = older_missing_scenes(root, cfg.n_cam, cfg.cam_stride, set(cfg.train_scenes) | set(cfg.val_scenes))
        assert not missing, f"BUILT.json has no finished build of {older_key(cfg.n_cam, cfg.cam_stride)} for {len(missing)} scene(s) of the config, e.g. {missing[:3]}"''', '''    scenes = set(cfg.train_scenes) | set(cfg.val_scenes)
    missing = older_missing_scenes(root, cfg.n_cam, cfg.cam_stride, scenes) if older else key_missing_scenes(root, scenes)
    assert not missing, f"BUILT.json has no finished build of {older_key(cfg.n_cam, cfg.cam_stride) if older else KEY} for {len(missing)} scene(s) of the config, e.g. {missing[:3]}"''')
open(p, "w").write(s)
