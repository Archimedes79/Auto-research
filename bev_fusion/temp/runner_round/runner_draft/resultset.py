"""resultset - what a network produced on frames, stored so that the evaluation never needs the network.

    w = ResultWriter(root, identity, frames, dts, taps)       # a folder root/<id>.partial; the id comes from the content (identity, frames, dts, taps)
    w.add("scene-0003/000", outputs, taps)                     # one frame, written at once (memory stays small); outputs = {"dt": {dt: {...}}}, see OUTPUTS
    rs = w.finish()                                            # result.json written, the folder renamed to root/<id>: from now on it is never changed
    rs = ResultSet.open("runs/t20_base_s0/results/3f9a0c1d2e")
    rs.id, rs.identity, rs.dts, rs.taps, rs.keys()             # what it is, which frames it holds
    rs.load("scene-0003/000")                                  # -> {"dt": {dt: {...}}, "taps": {name: array}}   (the same structure that was added)
    rs.subset(["scene-0003/000", ...])                         # a view on some of the frames (same files, an id of its own)
    rs.check()                                                 # -> list of problems, empty = sound

OUTPUTS, per dt (seconds, a float): `prob` {layer: (X, Y) probability map}, `dist` {layer: (X, Y) distance in metres}, `dir` {layer: (2, X, Y)}, `mass`, `vhat`, `objvel` (arrays or
None), `dets` (a list of detections, dicts of numbers and text), `detr` (the refined detections or None). Arrays are stored as float32, nothing is rounded. The structure is the
`outputs["dt"]` of a live subject of the evaluation, so a stored result and a live network feed the same metrics.

Project-independent: numpy and the standard library only. A result set is immutable and has an id from its content; the same content gives the same id, a new one a new folder.
File layout:  <id>/result.json (format, id, identity, frames, dts, taps, created)   <id>/frames/<scene>_<index>.npz (one frame; keys `<dt>::prob::<layer>`, `<dt>::mass`, `tap::<name>` ...)
"""
import datetime
import hashlib
import json
import os
import shutil

import numpy as np

FORMAT = 1
SEP = "::"                                   # inside a key of the .npz: <dt>::<group>::<layer>, <dt>::<name>, tap::<name>
GROUPS = ("prob", "dist", "dir")             # dicts of layer -> array
ARRAYS = ("mass", "vhat", "objvel")          # one array or None
LISTS = ("dets", "detr")                     # lists of detections (stored as json text) or None


class ResultSetError(Exception):
    """what the person has to fix; the message says what and where"""


def digest(obj, n=10):
    """the first n hex digits of the sha1 of a json-able object (keys sorted): the id of a result set, the hash of a frame list"""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:n]


def dt_name(dt):
    """a dt in seconds as text: +0, +0.5, -0.5"""
    return f"{float(dt):+g}"


def frame_key(scene, index):
    """the key of a frame: 'scene-0003/012' (the scene and the position inside it, in time order)"""
    return f"{scene}/{index:03d}"


def _file(key):
    return key.replace("/", "_") + ".npz"


def _json_default(x):
    if hasattr(x, "item"):
        return x.item()
    raise TypeError(f"{type(x).__name__} is not json-able")


def pack(outputs, taps=None):
    """outputs + taps -> {npz key: array}"""
    out = {}
    for dt, o in outputs["dt"].items():
        k = dt_name(dt)
        for g in GROUPS:
            for layer, a in (o.get(g) or {}).items():
                if SEP in layer:
                    raise ResultSetError(f"layer name {layer!r} contains {SEP!r}")
                out[f"{k}{SEP}{g}{SEP}{layer}"] = np.asarray(a, dtype=np.float32)
        for n in ARRAYS:
            if o.get(n) is not None:
                out[f"{k}{SEP}{n}"] = np.asarray(o[n], dtype=np.float32)
        for n in LISTS:
            if o.get(n) is not None:
                out[f"{k}{SEP}{n}"] = np.array(json.dumps(o[n], default=_json_default))
    for name, a in (taps or {}).items():
        out[f"tap{SEP}{name}"] = np.asarray(a)
    return out


def unpack(z):
    """an opened .npz -> {"dt": {dt: outputs}, "taps": {name: array}}; a dt has every field (None / empty when it was not stored)"""
    dts, taps = {}, {}
    for key in z.files:
        p = key.split(SEP)
        if p[0] == "tap":
            taps[p[1]] = z[key]
            continue
        o = dts.setdefault(float(p[0]), {**{g: {} for g in GROUPS}, **{n: None for n in ARRAYS + LISTS}})
        if len(p) == 3:
            o[p[1]][p[2]] = z[key]
        elif p[1] in LISTS:
            o[p[1]] = json.loads(str(z[key]))
        else:
            o[p[1]] = z[key]
    return {"dt": dict(sorted(dts.items())), "taps": taps}


def make_id(identity, tokens, dts, taps):
    """(id, identity with the frames' hash) of a result set: the same identity, frames, dts and taps always give the same id"""
    identity = dict(identity, frames=digest(list(tokens), 12), n_frames=len(tokens))
    return digest({"identity": identity, "dts": [float(d) for d in dts], "taps": sorted(taps)}), identity


def _write_json(path, obj):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(path + ".tmp", path)


class ResultWriter:
    """writes a result set frame by frame into root/<id>.partial and renames it to root/<id> in finish(). `frames`: [{"scene", "index", "token"}] in the order they will be added
    (time order inside a scene); `identity`: a json-able dict that says what produced the result (network, checkpoint, code, memory ...)."""

    def __init__(self, root, identity, frames, dts, taps=()):
        self.frames = [dict(f, key=frame_key(f["scene"], f["index"])) for f in frames]
        self._by_key = {f["key"]: f for f in self.frames}
        if len(self._by_key) != len(self.frames):
            raise ResultSetError("two frames have the same scene and index")
        self.dts, self.taps = [float(d) for d in dts], sorted(taps)
        self.id, self.identity = make_id(identity, [f["token"] for f in self.frames], self.dts, self.taps)
        self.root, self.final, self.folder = root, os.path.join(root, self.id), os.path.join(root, self.id + ".partial")
        if os.path.exists(self.final):
            raise ResultSetError(f"{self.final} exists: a result set is never overwritten, open it (ResultSet.open)")
        if os.path.exists(self.folder):                              # a crashed earlier attempt is moved aside, never deleted (hard rule 5)
            aside = os.path.join(root, "_to_delete")
            os.makedirs(aside, exist_ok=True)
            shutil.move(self.folder, os.path.join(aside, f"{self.id}.partial.{datetime.datetime.now():%Y%m%d_%H%M%S}"))
        os.makedirs(os.path.join(self.folder, "frames"))
        self._added = set()

    def add(self, key, outputs, taps=None, info=None):
        """store one frame; its dts and taps must be the ones announced; `info` is a small free dict kept in result.json (losses, timings, health numbers)"""
        frame = self._by_key.get(key)
        if frame is None:
            raise ResultSetError(f"{key} is not one of the announced frames")
        if key in self._added:
            raise ResultSetError(f"{key} was added twice")
        if sorted(float(d) for d in outputs["dt"]) != sorted(self.dts) or sorted(taps or {}) != self.taps:
            raise ResultSetError(f"{key}: dts {sorted(outputs['dt'])} / taps {sorted(taps or {})} differ from the announced {self.dts} / {self.taps}")
        np.savez_compressed(os.path.join(self.folder, "frames", _file(key)), **pack(outputs, taps))
        if info:
            frame["info"] = info
        self._added.add(key)

    def finish(self):
        """write result.json and make the folder final; refuses when a frame is missing -> the ResultSet"""
        missing = [f["key"] for f in self.frames if f["key"] not in self._added]
        if missing:
            raise ResultSetError(f"{len(missing)} frame(s) were not added, e.g. {missing[:3]}")
        _write_json(os.path.join(self.folder, "result.json"),
                    {"format": FORMAT, "id": self.id, "identity": self.identity, "dts": self.dts, "taps": self.taps, "frames": self.frames,
                     "created": datetime.datetime.now().isoformat(timespec="seconds")})
        os.rename(self.folder, self.final)
        return ResultSet.open(self.final)


class ResultSet:
    """a finished result set, or a view on some of its frames (subset)"""

    def __init__(self, folder, info, keys=None, view_of=None):
        self.folder, self.info, self.view_of = folder, info, view_of
        self.identity, self.dts, self.taps = info["identity"], info["dts"], info["taps"]
        self._frames = {f["key"]: f for f in info["frames"]}
        self._keys = list(keys) if keys is not None else [f["key"] for f in info["frames"]]
        self.id = info["id"] if view_of is None else f"{info['id']}+{digest(self._keys, 6)}"

    @classmethod
    def open(cls, folder):
        path = os.path.join(folder, "result.json")
        if not os.path.isfile(path):
            raise ResultSetError(f"{folder} has no result.json (not a result set, or not finished)")
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
        if info.get("format") != FORMAT:
            raise ResultSetError(f"{folder}: format {info.get('format')!r}, this code reads format {FORMAT}")
        return cls(folder, info)

    def keys(self):
        """the keys of the frames in this set, in the order they were produced"""
        return list(self._keys)

    def scenes(self):
        return list(dict.fromkeys(self._frames[k]["scene"] for k in self._keys))

    def load(self, key):
        """one frame -> {"dt": {dt: outputs}, "taps": {name: array}}"""
        if key not in self._keys:
            raise ResultSetError(f"{key} is not in this result set ({len(self._keys)} frames, e.g. {self._keys[:2]})")
        with np.load(os.path.join(self.folder, "frames", _file(key)), allow_pickle=False) as z:
            return unpack(z)

    def subset(self, keys):
        """a view on some of the frames (they keep the order of this set); the files are the same"""
        unknown = [k for k in keys if k not in self._keys]
        if unknown:
            raise ResultSetError(f"not in this result set: {unknown[:3]}")
        return ResultSet(self.folder, self.info, [k for k in self._keys if k in set(keys)], view_of=self.id)

    def summary(self):
        """a short dict for lists: id, network, step, memory, number of frames, scenes, dts, taps, folder"""
        i = self.identity
        return {"id": self.id, "network": i.get("network"), "step": i.get("step"), "memory": i.get("memory"), "frames": len(self._keys), "scenes": len(self.scenes()),
                "dts": self.dts, "taps": self.taps, "created": self.info["created"], "folder": self.folder}

    def check(self):
        """every file there and readable, every number finite, the same dts / layers / taps in every frame, the frames of a scene in time order -> list of problems"""
        problems, layers = [], None
        last = {}
        for k in self._keys:
            f = self._frames[k]
            if f["index"] <= last.get(f["scene"], -1):
                problems.append(f"{k}: not in time order inside its scene")
            last[f["scene"]] = f["index"]
            try:
                d = self.load(k)
            except (OSError, ValueError, KeyError, ResultSetError) as e:
                problems.append(f"{k}: cannot be read ({e})")
                continue
            if sorted(d["dt"]) != sorted(self.dts) or sorted(d["taps"]) != self.taps:
                problems.append(f"{k}: dts {sorted(d['dt'])} / taps {sorted(d['taps'])} differ from result.json")
            seen = {(dt, g, n) for dt, o in d["dt"].items() for g in GROUPS for n in o[g]}
            layers = seen if layers is None else layers
            if seen != layers:
                problems.append(f"{k}: other layers than the first frame")
            for dt, o in d["dt"].items():
                bad = [f"{g}/{n}" for g in GROUPS for n, a in o[g].items() if not np.isfinite(a).all()] + [n for n in ARRAYS if o[n] is not None and not np.isfinite(o[n]).all()]
                if bad:
                    problems.append(f"{k} dt {dt:+g}: not finite: {bad[:3]}")
        return problems
