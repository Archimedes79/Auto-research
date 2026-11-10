"""ResultSet: what a Subject predicted, kept on disk so that frames can be scored again (another ground truth, another metric, another subset) without the network.

    rs = ResultSet.create(path, identity)           # the set of this identity (or the stored one again: a stopped writer resumes)
    rs.add(ref, prediction)                         # one frame; atomic, so a killed writer leaves no half frame
    rs.has(ref, taps);  rs.load(ref, taps) -> Prediction;  rs.refs();  rs.close()

A ResultSet belongs to an IDENTITY: everything the predictions depend on and nothing else (the network and its checkpoint, the inputs it was fed, the memory mode). Its folder is named by the
hash of the identity, so the same network on the same data is one set however many different questions are asked of it: a question is only a list of frames, and any frame already stored
is reused (a subset of an earlier selection costs no network). A frame's prediction depends on the frames before it in its scene (the memory is carried: D70), but those are fixed by the
identity, so a stored frame is valid whatever else was asked. The dataset key of a frame is stored with it and checked on `load`: a data set that was rebuilt so that index k is another
keyframe is an error, not a wrong score.
Layout (`format` 2 in the manifest): `<path>/manifest.json`, `<path>/frames/<scene>_<index>.npz` and, for the intermediate results asked for with the predictions (taps),
`<path>/taps/<scene>_<index>/<tap>.npy`. A frame file is an ordinary numpy .npz (zlib level 1; `np.load(file)` reads it, no pickle anywhere): the arrays of the prediction as `a0.npy`, `a1.npy`,
... and `tree.json`, the nested outputs / info / key as json in which an array is `{"$a": i}`, a dict is `{"$d": [[key, value], ...]}` (keys keep their type: 0.0 is a float key), a tuple
`{"$t": [...]}`, a numpy scalar `{"$s": i}` and a float that is not finite `{"$f": "nan"}`. A frame stored without a tap does not have it: `has(ref, taps)` says no, and the frame is predicted
again to get it.
`writer_lock` (flock) keeps two writers (two chats asking the same question) out of one folder. The core knows nothing about the network or the data."""
from __future__ import annotations

import contextlib
import hashlib
import json
import io
import os
import re
import socket
import time
import zipfile
import zlib
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

try:
    import fcntl
except ImportError:                                                    # Windows
    fcntl = None

from .contracts import FrameRef, Prediction

FORMAT = 2
FRAME_FILE = re.compile(r"(.+)_(\d+)\.npz")                             # <scene>_<index>.npz: a frame (anything else in frames/ is not)


def encode(obj: Any):
    """(tree, arrays): obj as json-able data and the list of its numpy arrays (see the module text for the grammar); a type that is not stored raises TypeError, an object array too"""
    arrays: List[np.ndarray] = []

    def enc(o):
        if o is None or isinstance(o, (bool, str)):
            return o
        if isinstance(o, int):
            return int(o)
        if isinstance(o, float):
            return float(o) if np.isfinite(o) else {"$f": repr(float(o))}
        if isinstance(o, np.ndarray):
            if o.dtype == object:
                raise TypeError("an object array cannot be stored")
            arrays.append(o if o.flags["C_CONTIGUOUS"] else np.array(o, order="C"))      # (ascontiguousarray would turn a 0-d array into a 1-d one)
            return {"$a": len(arrays) - 1}
        if isinstance(o, np.generic):
            arrays.append(np.asarray(o))
            return {"$s": len(arrays) - 1}
        if isinstance(o, dict):
            return {"$d": [[enc(k), enc(v)] for k, v in o.items()]}
        if isinstance(o, tuple):
            return {"$t": [enc(x) for x in o]}
        if isinstance(o, list):
            return [enc(x) for x in o]
        raise TypeError(f"cannot store a {type(o).__name__}")
    return enc(obj), arrays


def decode(tree: Any, arrays: Sequence[np.ndarray]) -> Any:
    """the inverse of encode"""
    def dec(o):
        if isinstance(o, list):
            return [dec(x) for x in o]
        if isinstance(o, dict):
            (tag, val), = o.items()
            if tag == "$a":
                return arrays[val]
            if tag == "$s":
                return arrays[val][()]
            if tag == "$d":
                return {dec(k): dec(v) for k, v in val}
            if tag == "$t":
                return tuple(dec(x) for x in val)
            if tag == "$f":
                return float(val)
            raise ValueError(f"unknown tag {tag}")
        return o
    return dec(tree)


def digest(obj: Any, n: int = 12) -> str:
    """n hex digits of the sha1 of the json of obj (keys sorted)"""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def result_id(identity: Dict[str, Any]) -> str:
    """'r' + 12 hex digits of the identity"""
    return "r" + digest(identity)


def _n_arrays(tree: Any) -> int:
    """how many arrays a frame's tree refers to (the highest placeholder index + 1)"""
    if isinstance(tree, list):
        return max((_n_arrays(x) for x in tree), default=0)
    if isinstance(tree, dict):
        (tag, val), = tree.items()
        if tag in ("$a", "$s"):
            return val + 1
        if tag == "$d":
            return max((_n_arrays(x) for kv in val for x in kv), default=0)
        if tag == "$t":
            return max((_n_arrays(x) for x in val), default=0)
    return 0


def write_atomic(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)              # (package.py clean removes the emptied frames / taps folders of a set it keeps the manifest of)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)                                  # a reader (or a killed writer) never sees a partial file
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class Locked(RuntimeError):
    """another process is writing this folder"""


@contextlib.contextmanager
def writer_lock(path: str) -> Iterator[None]:
    """one writer per folder: an exclusive `flock` on the file `<path>.lock` (the kernel drops it when the holder dies, so there is no stale lock to take over); `pid host` of the holder is
    written into the file for the message. The file itself stays. Without `fcntl` (Windows Python) there is no lock: run the suite from WSL"""
    if fcntl is None:
        yield
        return
    lock = os.path.normpath(path) + ".lock"
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = os.read(fd, 200).decode("utf-8", "replace").strip() or "another process"
            raise Locked(f"{path} is being written by {holder} ({lock}); wait for it") from None
        os.ftruncate(fd, 0)
        os.write(fd, f"pid {os.getpid()} on {socket.gethostname()}".encode())
        yield
    finally:
        os.close(fd)                                                  # releases the lock


class ResultSet:
    def __init__(self, path: str, manifest: Dict[str, Any], mode: str):
        self.path, self.manifest, self.mode = path, manifest, mode
        self._have: Optional[set] = None
        self._since_write = 0

    # ------------------------------------------------------------------------------------------ create / open
    @classmethod
    def create(cls, path: str, identity: Dict[str, Any]) -> "ResultSet":
        """the set of this identity, new or stored (its frames are kept); the folder must be named by the id"""
        rid = result_id(identity)
        if os.path.basename(os.path.normpath(path)) != rid:
            raise ValueError(f"the folder of a result set is named by its id: {rid}, not {os.path.basename(os.path.normpath(path))}")
        if os.path.exists(os.path.join(path, "manifest.json")):
            return cls.open(path, mode="a")
        os.makedirs(os.path.join(path, "frames"), exist_ok=True)
        rs = cls(path, {"format": FORMAT, "id": rid, "identity": identity, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "health": {}, "wall_s": 0.0}, "a")
        rs.save()
        return rs

    @classmethod
    def open(cls, path: str, mode: str = "r") -> "ResultSet":
        with open(os.path.join(path, "manifest.json"), encoding="utf-8") as f:
            m = json.load(f)
        if m.get("format") != FORMAT:
            raise ValueError(f"{path}: result set format {m.get('format')}, this code reads {FORMAT}")
        return cls(path, m, mode)

    # ------------------------------------------------------------------------------------------ reading
    @property
    def id(self) -> str:
        return self.manifest["id"]

    def _file(self, ref: FrameRef) -> str:
        return os.path.join(self.path, "frames", f"{ref.slug}.npz")

    def _listing(self) -> set:
        if self._have is None:
            folder = os.path.join(self.path, "frames")
            self._have = {f for f in os.listdir(folder) if FRAME_FILE.fullmatch(f)} if os.path.isdir(folder) else set()
        return self._have

    def tap_path(self, ref: FrameRef, name: str) -> str:
        """the file of a tap of a frame (a '/' in the name becomes '_')"""
        return os.path.join(self.path, "taps", ref.slug, name.replace("/", "_") + ".npy")

    def has(self, ref: FrameRef, taps: Sequence[str] = ()) -> bool:
        """the frame is stored (with these taps, when asked)"""
        return os.path.basename(self._file(ref)) in self._listing() and all(os.path.exists(self.tap_path(ref, t)) for t in taps)

    def n_frames(self) -> int:
        return len(self._listing())

    def tap_names(self, ref: FrameRef) -> List[str]:
        folder = os.path.join(self.path, "taps", ref.slug)
        return sorted(f[:-4] for f in os.listdir(folder) if f.endswith(".npy")) if os.path.isdir(folder) else []

    def refs(self) -> List[FrameRef]:
        """the stored frames (scene and index; the dataset key is in the file, not in its name), in time order"""
        return sorted(FrameRef(m.group(1), int(m.group(2))) for m in map(FRAME_FILE.fullmatch, self._listing()))

    def load(self, ref: FrameRef, taps: Sequence[str] = ()) -> Prediction:
        """the stored prediction; `taps` = the intermediate results to load with it (a KeyError for one that was not stored); an IOError for a file that cannot be read or that belongs to
        another dataset key than `ref` (the data changed under the stored index)"""
        if not self.has(ref, taps):
            raise KeyError(f"{ref.label} is not in result set {self.id}" + (f" with the taps {list(taps)}" if taps else ""))
        try:
            with zipfile.ZipFile(self._file(ref)) as z:
                tree = json.loads(z.read("tree.json").decode("utf-8"))
                arrs = [np.lib.format.read_array(io.BytesIO(z.read(f"a{i}.npy")), allow_pickle=False) for i in range(_n_arrays(tree))]
            d = decode(tree, arrs)
            arrays = {t: np.load(self.tap_path(ref, t), allow_pickle=False) for t in taps}
            outputs, info, key = d["outputs"], d["info"], d.get("key", "")
        except (OSError, ValueError, KeyError, IndexError, EOFError, TypeError, zlib.error, zipfile.BadZipFile) as e:
            raise IOError(f"{self._file(ref)} (or a tap of it) is unreadable ({type(e).__name__}: {e}); remove the file, the next run predicts the frame again") from None
        if ref.key and key and ref.key != key:
            raise IOError(f"{ref.label} was predicted for the frame {key}, the data now has {ref.key} at that index: the dataset changed; remove {self._file(ref)} (the next run predicts it again)")
        return Prediction(outputs=outputs, taps=arrays, info=info)

    # ------------------------------------------------------------------------------------------ writing
    def add(self, ref: FrameRef, pred: Prediction) -> None:
        """store a frame, with its dataset key; the taps of the prediction go next to it. A frame that is stored keeps its outputs (a second call only adds taps)"""
        if self.mode == "r":
            raise IOError("result set opened read-only")
        if not self.has(ref):
            tree, arrays = encode({"outputs": pred.outputs, "info": pred.info, "key": ref.key})
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
                z.writestr("tree.json", json.dumps(tree).encode("utf-8"))
                for i, a in enumerate(arrays):
                    with z.open(f"a{i}.npy", "w", force_zip64=True) as f:
                        np.lib.format.write_array(f, a, allow_pickle=False)
            write_atomic(self._file(ref), buf.getvalue())
            self._listing().add(os.path.basename(self._file(ref)))
        for name, arr in pred.taps.items():
            buf = io.BytesIO()
            np.save(buf, np.asarray(arr))
            write_atomic(self.tap_path(ref, name), buf.getvalue())

    def close(self, wall_s: float = 0.0, cost: Optional[Dict[str, Any]] = None) -> None:
        """record the seconds this writer took and the cost of its predictions (kept per session: the last one that ran frames)"""
        if self.mode == "r":
            return
        self.manifest["wall_s"] = round(self.manifest.get("wall_s", 0.0) + wall_s, 1)
        if cost:
            self.manifest["cost"] = cost
        self.save()

    def save(self) -> None:
        """write the manifest now (close does it too; a long run saves it after each scene, so a killed writer keeps what it recorded)"""
        write_atomic(os.path.join(self.path, "manifest.json"), json.dumps(self.manifest, indent=1, default=str).encode("utf-8"))

    def size_mb(self) -> float:
        folder = os.path.join(self.path, "frames")
        return round(sum(os.path.getsize(os.path.join(folder, f)) for f in self._listing() if os.path.exists(os.path.join(folder, f))) / 1e6, 1)
