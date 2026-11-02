"""RunStore: what a quantitative run leaves on disk, so every number can be recomputed and traced without the network.

    <run dir>/manifest.json            who ran what (source / subject / task descriptions, arguments, metric names, time, commit)
    <run dir>/frames.jsonl             one line per frame: label, scene, index, key, scores, support, flags, info, chunk, pos
    <run dir>/counts_<metric>_<chunk>.npz   the Counts of `CHUNK` consecutive frames of one metric (arrays '<pos>|<name>')

Frames are appended in chunks (counts file first, then the frames.jsonl lines of that chunk), so a run that dies leaves a
consistent prefix and `RunStore.open(...).done()` tells what to skip when it is resumed (a frames.jsonl line cut off by the crash is dropped
on open, and cut away when the store is opened for writing; a run directory has ONE writer: opening it for writing while another process still
appends would cut that process's half-written line). Any shapes are allowed per frame
(fixed or ragged); the store never interprets a count.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .contracts import Counts, FrameRef

CHUNK = 64


def _git_commit(path: str) -> str:
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                                     # noqa: BLE001
        return ""


_IDENTITY_KEYS = ("state", "source", "subject", "task", "selection")
_VOLATILE = {"subject": ("device",)}


def _identity_differences(stored: Dict[str, Any], new: Dict[str, Any], keys=_IDENTITY_KEYS, gained: Optional[List[str]] = None,
                          either: bool = False, ignore: Optional[Dict[str, Any]] = None) -> List[str]:
    """the manifest entries that make a resume (or a benchmark attached to a run) another run (what differs, stored -> new).
    A key of a description (source / subject / task dict) that is NEW in `new` and absent from `stored` is not a difference: the
    stored run predates it (an adapter that later adds e.g. a ground-truth signature must not orphan its older stores); it is
    appended to `gained` ("source: gt_signature") for a warning. either=True (two stored runs compared) tolerates a key that only
    one side has, in both directions. A key present on both sides with another value always differs, unless `ignore` names it
    ({entry: keys}: inputs that two compared arms vary on purpose, e.g. a feature cache)."""
    out = []
    for k in keys:
        a, b = stored.get(k), new.get(k)
        if isinstance(a, dict) and isinstance(b, dict):
            drop = tuple(_VOLATILE.get(k, ())) + tuple((ignore or {}).get(k, ()))
            a, b = {x: v for x, v in a.items() if x not in drop}, {x: v for x, v in b.items() if x not in drop}
            only = sorted(set(b) - set(a)) + (sorted(set(a) - set(b)) if either else [])
            if only and gained is not None:
                gained.append(f"{k}: {', '.join(map(str, only))}")
            bad = sorted(x for x in set(a) | set(b) if x not in only and a.get(x) != b.get(x))
            if bad:
                out.append(f"{k}: " + ", ".join(f"{x} {a.get(x)!r} -> {b.get(x)!r}" for x in bad[:3]))
        elif a != b:
            out.append(f"{k}: {a!r} -> {b!r}")
    return out


def warn_gained(gained: List[str], where: str) -> None:
    """the warning for tolerated description keys (stderr: PYTHONWARNINGS=ignore must not hide it)"""
    if gained:
        print(f"evalsuite warning: {where}: description gained keys: {'; '.join(gained)}; the stored run predates them (compared on the "
              "keys both have)", file=sys.stderr, flush=True)


def _read_frames(fp: str, repair: bool) -> List[Dict[str, Any]]:
    """the frames of frames.jsonl. A cut-off LAST line (the run died inside a flush: a line is written with its newline, so a cut-off one is
    the text after the last newline) is dropped; with repair the file is cut back to the last whole line, so the next flush appends cleanly
    and the dropped frame is run again. Any other bad line (not JSON, not an object, or newline-terminated) is a damaged file: an error."""
    with open(fp, "rb") as f:
        raw = f.read()
    frames, end, offset = [], 0, 0                                        # end: the byte after the last whole line
    lines = raw.split(b"\n")
    for n, line in enumerate(lines, 1):
        offset += len(line) + 1
        if line.strip(b" \t\r\0"):
            try:
                rec = json.loads(line)
            except ValueError:
                if n == len(lines):                                       # nothing after it, not even a newline: cut off by the crash
                    break
                raise ValueError(f"{fp}: line {n} is not valid JSON: the file is damaged") from None
            if not isinstance(rec, dict) or not {"label", "chunk", "pos"} <= rec.keys():
                raise ValueError(f"{fp}: line {n} is not a frame record: the file is damaged")
            frames.append(rec)
        end = min(offset, len(raw))
    if repair and end < len(raw):
        with open(fp, "r+b") as f:
            f.truncate(end)
    elif repair and raw and not raw.endswith(b"\n"):
        with open(fp, "ab") as f:
            f.write(b"\n")
    return frames


class RunStore:
    def __init__(self, path: str, manifest: Dict[str, Any], mode: str):
        self.path, self.manifest, self.mode = path, manifest, mode
        self._frames: List[Dict[str, Any]] = []
        self._buf: List[Dict[str, Any]] = []                              # frames waiting for the next chunk
        self._cache: Dict[str, Any] = {}
        self._n_chunks = 0

    # ------------------------------------------------------------------------------------------ create / open
    @classmethod
    def create(cls, path: str, manifest: Dict[str, Any], resume: bool = True, force: bool = False) -> "RunStore":
        """a new run directory, or (resume) the existing one: frames already stored are kept. A resume must be the SAME run (state,
        source, subject, task, selection as in the stored manifest, the device aside): otherwise the stored counts would be mixed
        into the numbers of another checkpoint / seed / frame set. force=True skips that check. A description key that is new in
        `manifest` and absent from the stored run (an adapter that added e.g. a ground-truth signature) is tolerated with a warning; the
        stored descriptions are then kept and the keys listed under `predates_keys`."""
        if os.path.exists(os.path.join(path, "manifest.json")):
            if not resume:
                raise FileExistsError(f"{path} already holds a run (resume it or choose another name)")
            st = cls.open(path, mode="a", repair=False)                   # nothing is touched before it is clear that this is the same run
            gained: List[str] = []
            diff = [] if force else _identity_differences(st.manifest, manifest, gained=gained)
            if diff:
                raise ValueError(f"{path} holds a different run ({'; '.join(diff)}): choose another name, or force=True to resume it anyway")
            if os.path.exists(os.path.join(path, "frames.jsonl")):
                st._frames = _read_frames(os.path.join(path, "frames.jsonl"), repair=True)
            warn_gained(gained, f"resume of {path}")
            if any("frames_sha" in g for g in gained):
                print(f"evalsuite warning: resume of {path}: the stored run predates the frame-list check (selection: frames_sha), its frames are not verified "
                      "against this selection", file=sys.stderr, flush=True)
            old = {k: st.manifest.get(k) for k in _IDENTITY_KEYS}
            st.manifest.update({k: v for k, v in manifest.items() if k not in ("created",)})
            if gained and not force:                                       # the stored frames predate the new keys: keep the stored
                for k, v in old.items():                                   # descriptions, so the manifest does not claim them
                    if isinstance(v, dict):
                        st.manifest[k] = v
                st.manifest["predates_keys"] = sorted(set(st.manifest.get("predates_keys", [])) | set(gained))
            st.manifest["resumed"] = st.manifest.get("resumed", 0) + 1
            return st
        os.makedirs(path, exist_ok=True)
        m = dict(manifest)
        m.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
        m.setdefault("commit", _git_commit(os.path.dirname(os.path.abspath(__file__))))
        st = cls(path, m, "a")
        st._write_manifest()
        open(os.path.join(path, "frames.jsonl"), "a").close()
        return st

    @classmethod
    def open(cls, path: str, mode: str = "r", repair: Optional[bool] = None) -> "RunStore":
        """repair (default: when opened for writing): cut a torn last line of frames.jsonl away (_read_frames)"""
        with open(os.path.join(path, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        st = cls(path, manifest, mode)
        fp = os.path.join(path, "frames.jsonl")
        if os.path.exists(fp):
            st._frames = _read_frames(fp, repair=mode != "r" if repair is None else repair)
        st._n_chunks = 1 + max((fr["chunk"] for fr in st._frames), default=-1)
        return st

    # ------------------------------------------------------------------------------------------ writing
    def add(self, ref: FrameRef, counts: Dict[str, Counts], scores: Dict[str, float], support: Dict[str, float],
            flags: Sequence[Dict[str, Any]] = (), info: Optional[Dict[str, Any]] = None) -> None:
        if self.mode == "r":
            raise IOError("store opened read-only")
        self._buf.append({"label": ref.label, "scene": ref.scene, "index": ref.index, "key": ref.key,
                          "scores": scores, "support": support, "flags": list(flags), "info": info or {}, "_counts": counts})
        if len(self._buf) >= CHUNK:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        chunk = self._n_chunks
        by_metric: Dict[str, Dict[str, np.ndarray]] = {}
        for pos, fr in enumerate(self._buf):
            for metric, counts in fr["_counts"].items():
                by_metric.setdefault(metric, {})                         # a metric with empty counts for the whole chunk still gets its file
                for k, v in counts.items():
                    by_metric.setdefault(metric, {})[f"{pos}|{k}"] = np.asarray(v)
        for metric, arrays in by_metric.items():
            tmp = os.path.join(self.path, f"counts_{metric}_{chunk:05d}.tmp.npz")
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, os.path.join(self.path, f"counts_{metric}_{chunk:05d}.npz"))
        with open(os.path.join(self.path, "frames.jsonl"), "a", encoding="utf-8") as f:
            for pos, fr in enumerate(self._buf):
                line = {k: v for k, v in fr.items() if k != "_counts"}
                line.update(chunk=chunk, pos=pos, metrics=sorted(fr["_counts"]))
                f.write(json.dumps(line, allow_nan=True) + "\n")
                self._frames.append(line)
        self._buf = []
        self._n_chunks += 1
        self._write_manifest()

    def close(self, finished: bool = True) -> None:
        self.flush()
        self.manifest["finished"] = bool(finished)
        self.manifest["n_frames"] = len(self._frames)
        self.manifest["scenes"] = sorted({fr["scene"] for fr in self._frames})
        self._write_manifest()

    def set_cost(self, key: str, value: Dict[str, Any]) -> None:
        """store a cost entry (run_ms, bench, ops; see cost.py) in the manifest of this run and write it"""
        self.manifest.setdefault("cost", {})[key] = value
        self._write_manifest()

    def _write_manifest(self) -> None:
        tmp = os.path.join(self.path, "manifest.tmp.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=1, default=str)
        os.replace(tmp, os.path.join(self.path, "manifest.json"))

    # ------------------------------------------------------------------------------------------ reading
    @property
    def frames(self) -> List[Dict[str, Any]]:
        """the stored frames in the order they were run (flush() first when writing)"""
        return self._frames

    def done(self) -> set:
        return {fr["label"] for fr in self._frames}

    def labels(self) -> List[str]:
        return [fr["label"] for fr in self._frames]

    def scene_names(self) -> List[str]:
        seen: Dict[str, None] = {}
        for fr in self._frames:
            seen.setdefault(fr["scene"], None)
        return list(seen)

    def _chunk(self, metric: str, chunk: int) -> Dict[int, Counts]:
        key = f"{metric}|{chunk}"
        if key not in self._cache:
            if len(self._cache) > 6:
                self._cache.pop(next(iter(self._cache)))
            out: Dict[int, Counts] = {}
            fn = os.path.join(self.path, f"counts_{metric}_{chunk:05d}.npz")
            if os.path.exists(fn):                                           # missing (a run stored before 2026-10-31 whose metric gave {} for a
                with np.load(fn) as z:                                       # whole chunk): the frames of the chunk have empty counts
                    for name in z.files:
                        pos, k = name.split("|", 1)
                        out.setdefault(int(pos), {})[k] = z[name]
            self._cache[key] = out
        return self._cache[key]

    def counts(self, metric: str, labels: Optional[Iterable[str]] = None) -> List[Counts]:
        """Counts of one metric for the frames in `labels` (default: all), in stored order."""
        want = None if labels is None else set(labels)
        out = []
        for fr in self._frames:
            if want is not None and fr["label"] not in want:
                continue
            out.append(self._chunk(metric, fr["chunk"]).get(fr["pos"], {}))
        return out

    def by_scene(self, metric: str) -> Dict[str, List[Counts]]:
        res: Dict[str, List[Counts]] = {}
        for fr, c in zip(self._frames, self.counts(metric)):
            res.setdefault(fr["scene"], []).append(c)
        return res

    def frame(self, label: str) -> Dict[str, Any]:
        """the stored frame by its label; 'scene/3', 'scene:3' and 'scene/003' name the same frame (FrameRef.parse)"""
        for fr in self._frames:
            if fr["label"] == label:
                return fr
        parsed = FrameRef.parse(str(label))
        if parsed is not None:
            for fr in self._frames:
                if (fr["scene"], int(fr["index"])) == parsed:
                    return fr
        raise KeyError(f"{label!r}: no such frame in this run (labels look like {self._frames[0]['label']!r})" if self._frames else f"{label!r}: the run has no frames")
