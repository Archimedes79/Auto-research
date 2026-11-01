"""A memory-light nuScenes devkit object for the GT generator: the full devkit of v1.0-trainval needs 8+ GB of RAM (3.5 million
sample_data records, 1.3 million ego poses, 1.2 million annotations) and pushes a shared machine into swapping. autoGT only reads
the CAM_FRONT and the five RADAR channels, and a run works through the sequences group by group, so:

    nusc = light_nuscenes("~/data/nuscenes_full", "v1.0-trainval", scenes=["scene-0001", ...])   # ~0.7 GB for 85 sequences
    names = scene_names("~/data/nuscenes_full", "v1.0-trainval")                              # all 850 names, instantly

The big tables (sample_data, ego_pose, sample_annotation) are scanned in blocks and only the records of the requested
sequences are kept (for sample_data only the CAM_FRONT / RADAR channels); `scenes=None` keeps all sequences (about 6 GB for
trainval). Everything else is the unchanged devkit, so `get`, the reverse indices, `sample["data"]` (CAM_FRONT and the radars only)
and the map API work as usual; the `prev` / `next` chains of the kept channels stay intact because a chain never leaves its sensor.
"""
import json
import os
import os.path as osp
import re

_KEEP = ("/CAM_FRONT/", "/RADAR_")
_RECORD = re.compile(r"\{[^{}]*\}")                     # a record of these tables is a flat dict: no braces inside


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


def light_nuscenes(dataroot, version, scenes=None, verbose=False):
    from nuscenes.nuscenes import NuScenes
    dataroot = os.path.expanduser(dataroot)
    root = osp.join(dataroot, version)
    want = None
    if scenes is not None:
        with open(osp.join(root, "scene.json")) as f:
            want = {s["token"] for s in json.load(f) if s["name"] in set(scenes)}
        with open(osp.join(root, "sample.json")) as f:
            samples = {s["token"] for s in json.load(f) if s["scene_token"] in want}
    state = {}

    def kept_sample_data():
        """the CAM_FRONT / RADAR records of the wanted samples: read once, the devkit asks for ego_pose first and needs their pose tokens"""
        if "sd" not in state:
            state["sd"] = [json.loads(t) for t in _records(osp.join(root, "sample_data.json"))
                           if any(k in t for k in _KEEP) and (want is None or _value(t, "sample_token") in samples)]
            state["ego"] = {r["ego_pose_token"] for r in state["sd"]}
        return state["sd"]

    class _Light(NuScenes):
        def __load_table__(self, table_name):
            path = osp.join(root, f"{table_name}.json")
            if table_name == "sample_data":
                return kept_sample_data()
            if table_name == "ego_pose":
                kept_sample_data()
                return [json.loads(t) for t in _records(path) if _value(t, "token") in state["ego"]]
            if table_name == "sample_annotation" and want is not None:
                return [json.loads(t) for t in _records(path) if _value(t, "sample_token") in samples]
            with open(path) as f:
                tab = json.load(f)
            if want is not None and table_name == "scene":
                tab = [r for r in tab if r["token"] in want]
            if want is not None and table_name == "sample":
                tab = [r for r in tab if r["scene_token"] in want]
            return tab

    return _Light(version=version, dataroot=dataroot, verbose=verbose)
