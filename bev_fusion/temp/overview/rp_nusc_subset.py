"""scratch: a nuScenes devkit loader restricted to a few scenes, so that the evaluation adapter's dataset fits in ~1 GB instead of 7-8 GB (the full trainval tables:
2.6 M sample_data + 2.6 M ego_pose + 1.2 M annotations as python dicts). `restrict(scene_names)` patches NuScenes.__load_table__ IN MEMORY (the repository and the data are
untouched): the big tables are streamed record by record and only the records of the kept scenes are materialised; the small tables are loaded whole.
Kept: scene (by name), sample (scene_token), sample_data (sample_token; the sweeps of a keyframe carry its sample_token), ego_pose (tokens of the kept sample_data),
sample_annotation (sample_token), instance (first annotation kept); all other tables whole. For the kept scenes every record is the stored one."""
import json
import os

KEEP = {"scenes": set(), "memo": {}}


def stream(path, chunk=1 << 24):
    """records of a json file that is one big array of objects, one at a time (memory: one chunk)"""
    dec = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buf = f.read(chunk)
        buf = buf[buf.index("[") + 1:]
        pos = 0
        while True:
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buf):
                    break
                more = f.read(chunk)
                if not more:
                    return
                buf, pos = more, 0
            if buf[pos] == "]":
                return
            while True:
                try:
                    obj, end = dec.raw_decode(buf, pos)
                    break
                except json.JSONDecodeError:
                    more = f.read(chunk)
                    if not more:
                        raise
                    buf, pos = buf[pos:] + more, 0
            pos = end
            yield obj


def restrict(scene_names):
    from nuscenes.nuscenes import NuScenes
    KEEP["scenes"] = set(scene_names)
    memo = KEEP["memo"] = {}

    def load(self, table_name):
        root = self.table_root
        p = lambda t: os.path.join(root, f"{t}.json")                        # noqa: E731
        if table_name in memo:
            return memo[table_name]
        if table_name == "scene":
            sc = [r for r in stream(p("scene")) if r["name"] in KEEP["scenes"]]
            memo["_scene_tokens"] = {r["token"] for r in sc}
            memo["scene"] = sc
        elif table_name == "sample":
            st = memo["_scene_tokens"] if "_scene_tokens" in memo else load(self, "scene") and memo["_scene_tokens"]
            memo["sample"] = [r for r in stream(p("sample")) if r["scene_token"] in st]
            memo["_sample_tokens"] = {r["token"] for r in memo["sample"]}
        elif table_name == "sample_data":
            if "sample" not in memo:
                load(self, "scene")
                load(self, "sample")
            sm = memo["_sample_tokens"]
            memo["sample_data"] = [r for r in stream(p("sample_data")) if r["sample_token"] in sm]
            memo["_ego_tokens"] = {r["ego_pose_token"] for r in memo["sample_data"]}
        elif table_name == "ego_pose":
            if "sample_data" not in memo:
                load(self, "scene")
                load(self, "sample")
                load(self, "sample_data")
            eg = memo["_ego_tokens"]
            memo["ego_pose"] = [r for r in stream(p("ego_pose")) if r["token"] in eg]
        elif table_name == "sample_annotation":
            if "sample" not in memo:
                load(self, "scene")
                load(self, "sample")
            sm = memo["_sample_tokens"]
            memo["sample_annotation"] = [r for r in stream(p("sample_annotation")) if r["sample_token"] in sm]
            memo["_ann_tokens"] = {r["token"] for r in memo["sample_annotation"]}
        elif table_name == "instance":
            if "sample_annotation" not in memo:
                load(self, "scene")
                load(self, "sample")
                load(self, "sample_annotation")
            an = memo["_ann_tokens"]
            memo["instance"] = [r for r in stream(p("instance")) if r["first_annotation_token"] in an]
        else:
            with open(p(table_name)) as f:
                memo[table_name] = json.load(f)
        return memo[table_name]

    NuScenes.__load_table__ = load      # dunder-ended names are not mangled
