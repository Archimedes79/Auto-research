"""A memory-light nuScenes devkit object for the GT generator: the full devkit of v1.0-trainval needs 8+ GB of RAM (3.5 million
sample_data records, 1.3 million ego poses) and fails on a shared machine. autoGT only reads the CAM_FRONT and the five RADAR
channels, so the two big tables are scanned line by line and only those records are kept (about 40 % of sample_data, and the
ego poses they refer to); everything else is the unchanged devkit, so `get`, the reverse indices, `sample["data"]` (now for
CAM_FRONT and the radars only) and the map API work as usual. The `prev` / `next` chains of the kept channels stay intact because
a chain never leaves its sensor.

    nusc = light_nuscenes("~/data/nuscenes_full", "v1.0-trainval")      # ~4 GB instead of 8+ GB, ~1 min
"""
import json
import os
import os.path as osp

_KEEP = ("/CAM_FRONT/", "/RADAR_")


def _records(path):
    """records of a pretty-printed nuScenes table, one at a time as (token, text), without loading the file."""
    buf = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s == "{":
                buf = [s]
            elif s in ("}", "},"):
                buf.append("}")
                yield buf[1].split('"')[3] if len(buf) > 1 and buf[1].startswith('"token"') else None, "".join(buf)
                buf = []
            elif buf:
                buf.append(s)


def light_nuscenes(dataroot, version, verbose=False):
    from nuscenes.nuscenes import NuScenes

    class _Light(NuScenes):
        _needed_ego = None

        def __load_table__(self, table_name):
            path = osp.join(self.table_root, f"{table_name}.json")
            if table_name == "sample_data":
                recs = [json.loads(t) for _, t in _records(path) if any(k in t for k in _KEEP)]
                self._needed_ego = {r["ego_pose_token"] for r in recs}
                return recs
            if table_name == "ego_pose":
                if self._needed_ego is None:                         # the devkit loads ego_pose before sample_data: read both
                    need = set()
                    for _, t in _records(osp.join(self.table_root, "sample_data.json")):
                        if any(k in t for k in _KEEP):
                            need.add(json.loads(t)["ego_pose_token"])
                    self._needed_ego = need
                return [json.loads(t) for tok, t in _records(path) if tok in self._needed_ego]
            with open(path) as f:
                return json.load(f)

    return _Light(version=version, dataroot=os.path.expanduser(dataroot), verbose=verbose)
