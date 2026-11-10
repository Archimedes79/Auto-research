"""A memory-light nuScenes devkit object for the GT generator: the full devkit of v1.0-trainval needs 8+ GB of RAM (3.5 million
sample_data records, 1.3 million ego poses, 1.2 million annotations) and pushes a shared machine into swapping. GtGenerator only reads
the CAM_FRONT and the five RADAR channels, and a run works through the sequences group by group, so:

    nusc = light_nuscenes("~/data/nuscenes_full", "v1.0-trainval", scenes=["scene-0001", ...])   # ~0.7 GB for 85 sequences
    names = scene_names("~/data/nuscenes_full", "v1.0-trainval")                              # all 850 names, instantly

The big tables (sample_data, ego_pose, sample_annotation) are scanned in blocks and only the records of the requested
sequences are kept (for sample_data only the CAM_FRONT / RADAR channels); `scenes=None` keeps all sequences (about 6 GB for
trainval). Everything else is the unchanged devkit, so `get`, the reverse indices, `sample["data"]` (CAM_FRONT and the radars only)
and the map API work as usual; the `prev` / `next` chains of the kept channels stay intact because a chain never leaves its sensor.

CACHE. The scan of the three big tables takes 31 - 35 s for a group of 60 sequences of trainval (sample_data 17 s, ego_pose 7 s, sample_annotation 6 s) and is repeated for
every group. `ensure_cache(dataroot, version)` reads them ONCE and writes the kept records per sequence (data/autogt/cache/light_<version>/<scene token>.pkl.gz, about
the size of the records of CAM_FRONT and the radars); `light_nuscenes` takes a group from there (a second or two) when the cache is valid, i.e. built from tables of the same size
and modification time, and scans as before otherwise (other cameras for pictures, `scenes=None`, no cache). The cache is derived data and rebuilt by `ensure_cache`.
"""
import gzip
import json
import os
import os.path as osp
import pickle
import re
import shutil
import time

_KEEP = ("/CAM_FRONT/", "/RADAR_")
_RECORD = re.compile(r"\{[^{}]*\}")                     # a record of these tables is a flat dict: no braces inside
BIG_TABLES = ("sample_data", "ego_pose", "sample_annotation")
CACHE_FORMAT = 1
CACHE_BASE = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "data", "autogt", "cache")


def _records(path, block=1 << 26):
    """the records of a pretty-printed nuScenes table, one at a time as text, read in blocks of 64 MB (no file in memory)."""
    rest = ""
    with open(path, encoding="utf-8") as f:
        for chunk in iter(lambda: f.read(block), ""):
            text, end = rest + chunk, 0
            for m in _RECORD.finditer(text):
                yield m.group()
                end = m.end()
            rest = text[end:]                                  # the start of a record that goes on in the next block


def _value(text, key):
    """the string value of `"key": "..."` in a record, or None (nuScenes writes `"key": "value"`; "token" does not match inside "sample_token")."""
    start = f'"{key}": "'
    i = text.find(start)
    return None if i < 0 else text[i + len(start):text.index('"', i + len(start))]


def scene_names(dataroot, version):
    with open(osp.join(os.path.expanduser(dataroot), version, "scene.json")) as f:
        return [s["name"] for s in json.load(f)]


# ---------------------------------------------------------------------------------------------------------------- the cache
def cache_dir(version, base=None):
    return osp.join(base or CACHE_BASE, f"light_{version}")


def _sources(root):
    """size and modification time of the big tables: the cache is valid for these and no others"""
    out = {}
    for t in BIG_TABLES:
        s = os.stat(osp.join(root, t + ".json"))
        out[t] = [s.st_size, s.st_mtime_ns]
    return out


def cache_valid(dataroot, version, base=None):
    d = cache_dir(version, base)
    try:
        with open(osp.join(d, "index.json")) as f:
            idx = json.load(f)
        return idx.get("format") == CACHE_FORMAT and idx.get("sources") == _sources(osp.join(os.path.expanduser(dataroot), version))
    except (OSError, ValueError):
        return False


def ensure_cache(dataroot, version, base=None, verbose=True):
    """make sure the per-sequence cache of the big tables exists and fits the data on disk; builds it (one pass over the three tables, about 40 s for trainval) when not.
    Call it once in the main process before any worker starts. -> the folder"""
    if cache_valid(dataroot, version, base):
        return cache_dir(version, base)
    root = osp.join(os.path.expanduser(dataroot), version)
    t0 = time.time()
    with open(osp.join(root, "sample.json")) as f:
        sample_scene = {s["token"]: s["scene_token"] for s in json.load(f)}
    with open(osp.join(root, "scene.json")) as f:
        scene_tokens = [s["token"] for s in json.load(f)]
    by = {s: {"sd": [], "ego": [], "ann": []} for s in scene_tokens}
    ego_scene = {}
    for t in _records(osp.join(root, "sample_data.json")):
        if any(k in t for k in _KEEP):
            rec = json.loads(t)
            sc = sample_scene.get(rec["sample_token"])
            if sc is not None:
                by[sc]["sd"].append(rec)
                ego_scene[rec["ego_pose_token"]] = sc
    for t in _records(osp.join(root, "ego_pose.json")):
        sc = ego_scene.get(_value(t, "token"))
        if sc is not None:
            by[sc]["ego"].append(json.loads(t))
    for t in _records(osp.join(root, "sample_annotation.json")):
        sc = sample_scene.get(_value(t, "sample_token"))
        if sc is not None:
            by[sc]["ann"].append(json.loads(t))
    final = cache_dir(version, base)
    tmp = f"{final}.tmp.{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for sc, rec in by.items():
        with gzip.open(osp.join(tmp, sc + ".pkl.gz"), "wb", compresslevel=3) as f:
            pickle.dump(rec, f, protocol=4)
    with open(osp.join(tmp, "index.json"), "w") as f:
        json.dump({"format": CACHE_FORMAT, "version": version, "sources": _sources(root), "scenes": len(by), "channels": [k.strip("/") for k in _KEEP],
                   "built": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
    old = f"{final}.old.{os.getpid()}"                      # swap, never remove first: a reader sees the old folder or the new one, not half of a removal
    if osp.isdir(final):
        os.replace(final, old)
    os.replace(tmp, final)
    shutil.rmtree(old, ignore_errors=True)
    if verbose:
        size = sum(e.stat().st_size for e in os.scandir(final)) / 1e6
        print(f"light devkit cache of {version}: {len(by)} sequences, {size:.0f} MB, {time.time() - t0:.0f} s -> {final}", flush=True)
    return final


# ---------------------------------------------------------------------------------------------------------------- the devkit object
def light_nuscenes(dataroot, version, scenes=None, verbose=False, cameras=("CAM_FRONT",), radars=True, cache_base=None):
    from nuscenes.nuscenes import NuScenes
    dataroot = os.path.expanduser(dataroot)
    root = osp.join(dataroot, version)
    want = None
    if scenes is not None:
        with open(osp.join(root, "scene.json")) as f:
            want = {s["token"] for s in json.load(f) if s["name"] in set(scenes)}
        with open(osp.join(root, "sample.json")) as f:
            samples = {s["token"] for s in json.load(f) if s["scene_token"] in want}
    keep = tuple(f"/{c}/" for c in cameras) + (("/RADAR_",) if radars else ())     # other cameras only for pictures, no radars for the lane GT
    state = {}
    cached = None
    if want is not None and set(cameras) <= {"CAM_FRONT"} and cache_valid(dataroot, version, cache_base):          # a group from the per-sequence cache: a second, not half a minute
        d = cache_dir(version, cache_base)
        with open(osp.join(root, "scene.json")) as f:
            order = [s["token"] for s in json.load(f) if s["token"] in want]
        cached = {"sd": [], "ego": [], "ann": []}
        try:
            for sc in order:
                with gzip.open(osp.join(d, sc + ".pkl.gz"), "rb") as f:
                    part = pickle.load(f)
                for k in cached:
                    cached[k].extend(part[k])
        except (OSError, EOFError, pickle.UnpicklingError, KeyError):         # the cache was replaced or damaged while it was read: the scan gives the same records
            cached = None

    def kept_sample_data():
        """the CAM_FRONT / RADAR records of the wanted samples: read once, the devkit asks for ego_pose first and needs their pose tokens"""
        if "sd" not in state:
            if cached is not None:
                state["sd"] = [r for r in cached["sd"] if any(k in r["filename"] for k in keep)]
            else:
                state["sd"] = [json.loads(t) for t in _records(osp.join(root, "sample_data.json"))
                               if any(k in t for k in keep) and (want is None or _value(t, "sample_token") in samples)]
            state["ego"] = {r["ego_pose_token"] for r in state["sd"]}
        return state["sd"]

    class _Light(NuScenes):
        def __load_table__(self, table_name):
            path = osp.join(root, f"{table_name}.json")
            if table_name == "sample_data":
                return kept_sample_data()
            if table_name == "ego_pose":
                kept_sample_data()
                if cached is not None:
                    return [r for r in cached["ego"] if r["token"] in state["ego"]]
                return [json.loads(t) for t in _records(path) if _value(t, "token") in state["ego"]]
            if table_name == "sample_annotation" and want is not None:
                if cached is not None:
                    return list(cached["ann"])
                return [json.loads(t) for t in _records(path) if _value(t, "sample_token") in samples]
            with open(path) as f:
                tab = json.load(f)
            if want is not None and table_name == "scene":
                tab = [r for r in tab if r["token"] in want]
            if want is not None and table_name == "sample":
                tab = [r for r in tab if r["scene_token"] in want]
            return tab

    return _Light(version=version, dataroot=dataroot, verbose=verbose)
