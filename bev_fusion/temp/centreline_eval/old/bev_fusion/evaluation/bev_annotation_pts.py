"""Lidar / radar point counts of every nuScenes annotation (box), as a small lookup table: token -> (num_lidar_pts, num_radar_pts).

Why: our rig is a front camera plus radar. Whether a GT box is observed by the RADAR (at least one radar return inside it) is the label that separates "things the
camera can see" from "things camera and radar can see" in the evaluation (evaluation/bev_coverage.py); the GT cache does not store the counts (it stores only
`observed` = lidar OR radar). The counts are fields of the nuScenes annotation file, read once, kept as derived data (not in git):

    python evaluation/bev_annotation_pts.py build        reads <NUSC_ROOT>/v1.0-trainval/sample_annotation.json in chunks (about 600 MB: minutes, I/O bound) ->
                                                         data/cache/annotation_pts.npz (a few tens of MB)
    python evaluation/bev_annotation_pts.py check        the table against the GT cache: every box token of 200 cached frames must be found
    from evaluation.bev_annotation_pts import load; pts = load(); pts("<annotation token>") -> (lidar, radar) or None

NUSC_ROOT defaults to ~/data/nuscenes_full (the same WSL view the rest of the code reads). `num_radar_pts` counts the points of ALL five radars of the keyframe
sweep inside the box, so a box can have 0 although a radar looked at it (small or far objects, pedestrians: no return)."""
from __future__ import annotations

import os
import re
import sys
from typing import Optional, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "data", "cache", "annotation_pts.npz")
PAT = re.compile(r'"token": "([0-9a-f]{32})".{0,2500}?"num_lidar_pts": (\d+),\s*"num_radar_pts": (\d+)', re.S)


def source_path(version: str = "v1.0-trainval") -> str:
    return os.path.join(os.environ.get("NUSC_ROOT", os.path.expanduser("~/data/nuscenes_full")), version, "sample_annotation.json")


def build(path: Optional[str] = None, out: str = OUT) -> int:
    path = path or source_path()
    found = {}
    with open(path, "r", encoding="utf-8") as fh:
        tail = ""
        while True:
            chunk = fh.read(1 << 25)
            if not chunk:
                break
            buf = tail + chunk
            for m in PAT.finditer(buf):
                found[m.group(1)] = (int(m.group(2)), int(m.group(3)))
            tail = buf[-6000:]
    keys = sorted(found)
    tok = np.array([bytes.fromhex(k) for k in keys], dtype="S16")
    lid = np.array([found[k][0] for k in keys], np.uint16)
    rad = np.array([found[k][1] for k in keys], np.uint16)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out + ".tmp.npz", tokens=tok, lidar=lid, radar=rad)
    os.replace(out + ".tmp.npz", out)
    print(f"{len(keys)} annotations -> {out}")
    return len(keys)


def load(path: str = OUT):
    d = np.load(path)
    raw, lid, rad = d["tokens"], d["lidar"], d["radar"]
    by = np.frombuffer(raw.tobytes(), np.uint8).reshape(-1, 16)               # (numpy's S16 comparison strips trailing zero bytes: tokens ending in "00" would not be found)
    hi = by[:, :8].copy().view(">u8").ravel()                                  # big-endian halves: the table is sorted by bytes, i.e. by (hi, lo)
    lo = by[:, 8:].copy().view(">u8").ravel()

    def pts(token: str) -> Optional[Tuple[int, int]]:
        b = np.frombuffer(bytes.fromhex(token), ">u8")
        i = int(np.searchsorted(hi, b[0], side="left"))
        while i < len(hi) and hi[i] == b[0]:
            if lo[i] == b[1]:
                return int(lid[i]), int(rad[i])
            i += 1
        return None
    pts.n = len(hi)
    return pts


def check(n_frames: int = 200) -> None:
    import glob
    sys.path[:0] = [ROOT]
    import evaluation.bev_gtcheck as G
    pts = load()
    files = sorted(glob.glob(os.path.join(G.cache_dir(), f"*_{G.resolve_sig()}.npz")))[:n_frames]
    total = miss = obs_bad = 0
    for f in files:
        g = G.load(f).get("cur")
        for b in (g["meta"]["boxes"] if g else []):
            total += 1
            v = pts(b["token"])
            if v is None:
                miss += 1
            elif bool(v[0] + v[1] > 0) != bool(b.get("observed", True)):
                obs_bad += 1                                          # the GT's `observed` must be lidar + radar > 0
    for tok, want in (("30f65d58c9c34488991774bec0ff0500", (3, 1)), ("5bb38a1b262649debccfd4fc00c1c600", (11, 1)), ("c28b310e5a3f4f258f1ef52e0f3c3700", (1, 2))):
        assert pts(tok) == want, (tok, pts(tok), want)         # values read from sample_annotation.json by hand: (lidar, radar) in this order; the first token ends in "00"
    print(f"{pts.n} annotations in the table; {total} GT boxes of {len(files)} frames: {miss} not found, {obs_bad} with an `observed` flag that disagrees with the counts")
    assert miss == 0 and obs_bad == 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build()
    elif cmd == "check":
        check()
    else:
        print(__doc__)
