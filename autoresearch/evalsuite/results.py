"""ResultSet: what a Subject predicted, kept on disk so that frames can be scored again (another ground truth, another metric, another subset) without the network.

    rs = ResultSet.create(path, identity)           # the set of this identity (or the stored one again: a stopped writer resumes)
    rs.add(ref, prediction)                         # one frame; atomic, so a killed writer leaves no half frame
    rs.has(ref, taps);  rs.load(ref, taps) -> Prediction;  rs.refs();  rs.close()

A ResultSet belongs to an IDENTITY: everything the predictions depend on and nothing else (the network and its checkpoint, the inputs it was fed, the memory mode). Its folder is named by the
hash of the identity, so the same network on the same data is one set however many different questions are asked of it: a question is only a list of frames, and any frame already stored
is reused (a subset of an earlier selection costs no network). A frame's prediction depends on the frames before it in its scene (the memory is carried: D70), but those are fixed by the
identity, so a stored frame is valid whatever else was asked. The dataset key of a frame is stored with it and checked on `load`: a data set that was rebuilt so that index k is another
keyframe is an error, not a wrong score.
Layout (`format` 1 in the manifest): `<path>/manifest.json`, `<path>/frames/<scene>_<index>.pkl` (a zlib level 1 compressed pickle of {outputs, info, key}: the program's own cache, not an
exchange format, `pickle.load` cannot read it directly, use `load`; open a ResultSet only from a folder this program made) and, for the intermediate results asked for with the predictions
(taps), `<path>/taps/<scene>_<index>/<tap>.npy`. A frame stored without a tap does not have it: `has(ref, taps)` says no, and the frame is predicted again to get it.
`writer_lock` keeps two writers (two chats asking the same question) out of one folder. The core knows nothing about the network or the data."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pickle
import socket
import time
import uuid
import zlib
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

from .contracts import FrameRef, Prediction

FORMAT = 1


def digest(obj: Any, n: int = 12) -> str:
    """n hex digits of the sha1 of the json of obj (keys sorted)"""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def result_id(identity: Dict[str, Any]) -> str:
    """'r' + 12 hex digits of the identity"""
    return "r" + digest(identity)


def _write_atomic(path: str, data: bytes) -> None:
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


def _alive(pid: int) -> bool:
    if os.name == "nt":                                        # os.kill(pid, 0) would terminate the process there: a lock of another process on Windows is never taken over
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


@contextlib.contextmanager
def writer_lock(path: str) -> Iterator[None]:
    """one writer per folder: a file `<path>.lock` made exclusively, holding `pid host token`. A lock of a dead process on this host is taken over (the file is renamed away first: only one
    contender gets it); an unreadable or empty one after 5 s; any other is refused with `Locked`. A holder removes only the lock that holds its own token."""
    lock = os.path.normpath(path) + ".lock"
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    host, token = socket.gethostname(), uuid.uuid4().hex
    mine = f"{os.getpid()} {host} {token}"
    for _ in range(200):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, mine.encode())
            os.close(fd)
            break
        except FileExistsError:
            text = _read(lock)
            parts = text.split()
            if len(parts) < 3:                                 # just being created by its owner, or damaged
                try:
                    old = time.time() - os.path.getmtime(lock) > 5
                except OSError:
                    continue
                if old:
                    _take_over(lock, text)
                else:
                    time.sleep(0.05)
                continue
            if parts[1] == host and parts[0].isdigit() and not _alive(int(parts[0])):
                _take_over(lock, text)
                continue
            raise Locked(f"{path} is being written by pid {parts[0]} on {parts[1]} ({lock}); wait for it, or remove the lock file if that process is gone")
    else:
        raise Locked(f"{path}: could not take the lock {lock}")
    try:
        yield
    finally:
        if _read(lock) == mine:
            try:
                os.remove(lock)
            except OSError:
                pass


def _take_over(lock: str, seen: str) -> None:
    """remove the stale lock whose text we read: rename it away (atomic, one contender wins); when what we renamed is not what we read (a live lock was made in between) put it back"""
    away = f"{lock}.stale.{uuid.uuid4().hex}"
    try:
        os.rename(lock, away)
    except OSError:
        return
    if _read(away) == seen:
        os.remove(away)
        return
    try:
        os.link(away, lock)                                    # fails when somebody made a new lock again: then that one stays
    except OSError:
        pass
    os.remove(away)


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
        rs._write_manifest()
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
        return os.path.join(self.path, "frames", f"{ref.slug}.pkl")

    def _listing(self) -> set:
        if self._have is None:
            folder = os.path.join(self.path, "frames")
            self._have = {f for f in os.listdir(folder) if f.endswith(".pkl")} if os.path.isdir(folder) else set()
        return self._have

    def _tap_file(self, ref: FrameRef, name: str) -> str:
        return os.path.join(self.path, "taps", ref.slug, name.replace("/", "_") + ".npy")

    def has(self, ref: FrameRef, taps: Sequence[str] = ()) -> bool:
        """the frame is stored (with these taps, when asked)"""
        return os.path.basename(self._file(ref)) in self._listing() and all(os.path.exists(self._tap_file(ref, t)) for t in taps)

    def n_frames(self) -> int:
        return len(self._listing())

    def tap_names(self, ref: FrameRef) -> List[str]:
        folder = os.path.join(self.path, "taps", ref.slug)
        return sorted(f[:-4] for f in os.listdir(folder) if f.endswith(".npy")) if os.path.isdir(folder) else []

    def refs(self) -> List[FrameRef]:
        """the stored frames (scene and index; the dataset key is in the file, not in its name), in time order"""
        out = []
        for f in self._listing():
            scene, _, idx = f[:-4].rpartition("_")
            out.append(FrameRef(scene, int(idx)))
        return sorted(out)

    def load(self, ref: FrameRef, taps: Sequence[str] = ()) -> Prediction:
        """the stored prediction; `taps` = the intermediate results to load with it (a KeyError for one that was not stored); an IOError for a file that cannot be read or that belongs to
        another dataset key than `ref` (the data changed under the stored index)"""
        if not self.has(ref, taps):
            raise KeyError(f"{ref.label} is not in result set {self.id}" + (f" with the taps {list(taps)}" if taps else ""))
        try:
            with open(self._file(ref), "rb") as f:
                d = pickle.loads(zlib.decompress(f.read()))
            arrays = {t: np.load(self._tap_file(ref, t)) for t in taps}
            outputs, info, key = d["outputs"], d["info"], d.get("key", "")
        except (OSError, ValueError, KeyError, AttributeError, ImportError, zlib.error, pickle.UnpicklingError, EOFError) as e:
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
            _write_atomic(self._file(ref), zlib.compress(pickle.dumps({"outputs": pred.outputs, "info": pred.info, "key": ref.key}, protocol=4), 1))
            self._listing().add(os.path.basename(self._file(ref)))
        for name, arr in pred.taps.items():
            os.makedirs(os.path.dirname(self._tap_file(ref, name)), exist_ok=True)
            tmp = self._tap_file(ref, name) + f".tmp.{os.getpid()}"
            with open(tmp, "wb") as f:
                np.save(f, np.asarray(arr))
            os.replace(tmp, self._tap_file(ref, name))

    def close(self, wall_s: float = 0.0, cost: Optional[Dict[str, Any]] = None) -> None:
        """record the seconds this writer took and the cost of its predictions (kept per session: the last one that ran frames)"""
        if self.mode == "r":
            return
        self.manifest["wall_s"] = round(self.manifest.get("wall_s", 0.0) + wall_s, 1)
        if cost:
            self.manifest["cost"] = cost
        self._write_manifest()

    def _write_manifest(self) -> None:
        _write_atomic(os.path.join(self.path, "manifest.json"), json.dumps(self.manifest, indent=1, default=str).encode("utf-8"))

    def size_mb(self) -> float:
        folder = os.path.join(self.path, "frames")
        return round(sum(os.path.getsize(os.path.join(folder, f)) for f in self._listing() if os.path.exists(os.path.join(folder, f))) / 1e6, 1)
