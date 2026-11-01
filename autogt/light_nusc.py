"""A memory-light nuScenes devkit object for the GT generator: the full devkit of v1.0-trainval needs 8+ GB of RAM (3.5 million
sample_data records, 1.3 million ego poses, 1.2 million annotations) and pushes a shared machine into swapping. autoGT only reads
the CAM_FRONT and the five RADAR channels, and a run works through the sequences group by group, so:

    nusc = light_nuscenes("~/data/nuscenes_full", "v1.0-trainval", scenes=["scene-0001", ...])   # ~0.7 GB for 85 sequences
    names = scene_names("~/data/nuscenes_full", "v1.0-trainval")                              # all 850 names, instantly

The big tables (sample_data, ego_pose, sample_annotation) are scanned line by line and only the records of the requested
sequences are kept (for sample_data only the CAM_FRONT / RADAR channels); `scenes=None` keeps all sequences (about 6 GB for
trainval). Everything else is the unchanged devkit, so `get`, the reverse indices, `sample["data"]` (CAM_FRONT and the radars only)
and the map API work as usual; the `prev` / `next` chains of the kept channels stay intact because a chain never leaves its sensor.
"""
import json
import os
import os.path as osp

_KEEP = ("/CAM_FRONT/", "/RADAR_")


def _records(path):
    """records of a pretty-printed nuScenes table, one at a time as (list of stripped lines, joined text), no file in memory."""
    buf = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s == "{":
                buf = [s]
            elif s in ("}", "},"):
                buf.append("}")
                yield buf, "".join(buf)
                buf = []
            elif buf:
                buf.append(s)


def _value(buf, key):
    """the string value of `"key": "..."` among the first lines of a record (nuScenes writes token first), or None."""
    pre = f'"{key}"'
    for ln in buf[1:6]:
        if ln.startswith(pre):
            return ln.split('"')[3]
    return None


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

    class _Light(NuScenes):
        def __load_table__(self, table_name):
            path = osp.join(root, f"{table_name}.json")
            if table_name == "sample_data":
                recs = []
                for buf, t in _records(path):
                    if any(k in t for k in _KEEP) and (want is None or _value(buf, "sample_token") in samples):
                        recs.append(json.loads(t))
                state["ego"] = {r["ego_pose_token"] for r in recs}
                return recs
            if table_name == "ego_pose":
                if "ego" not in state:                              # the devkit loads ego_pose before sample_data: read both
                    state["ego"] = set()
                    for buf, t in _records(osp.join(root, "sample_data.json")):
                        if any(k in t for k in _KEEP) and (want is None or _value(buf, "sample_token") in samples):
                            state["ego"].add(json.loads(t)["ego_pose_token"])
                return [json.loads(t) for buf, t in _records(path) if _value(buf, "token") in state["ego"]]
            if table_name == "sample_annotation" and want is not None:
                return [json.loads(t) for buf, t in _records(path) if _value(buf, "sample_token") in samples]
            with open(path) as f:
                tab = json.load(f)
            if want is not None and table_name == "scene":
                tab = [r for r in tab if r["token"] in want]
            if want is not None and table_name == "sample":
                tab = [r for r in tab if r["scene_token"] in want]
            return tab

    return _Light(version=version, dataroot=dataroot, verbose=verbose)
