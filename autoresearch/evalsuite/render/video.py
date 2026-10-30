"""Frames -> a video that plays everywhere (H.264, yuv420p, mp4 with the index up front), checked by reading it back.

    encode(frames, out_mp4, fps)  ->  {"path", "codec", "frames", "size", "method"}; raises if nothing plays

frames: image files (png / jpg) or arrays (H, W, 3 uint8) in order. Why not cv2.VideoWriter('mp4v'): that is MPEG-4 part 2, which
browsers, the Claude app and most players refuse (2026-10-30: the first scene video did not play). H.264 needs an ffmpeg with
libx264, looked up in $FFMPEG, the PATH and the ffmpeg.exe builds installed on the Windows side (called through WSL interop,
`CANDIDATES`). Without any ffmpeg an animated GIF is written instead and the result says so. After encoding the file is decoded
with OpenCV: frame count and a non-black first frame must match, otherwise the call fails (a video that does not play is never
reported as done).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Sequence, Union

import numpy as np

CANDIDATES = ("/mnt/c/Program Files/DownloadHelper CoApp/ffmpeg.exe",
              "/mnt/c/Program Files (x86)/Digiarty/VideoProc Converter AI/ffmpeg.exe")


def find_ffmpeg():
    """path of an ffmpeg with libx264, or None"""
    for c in [os.environ.get("FFMPEG"), shutil.which("ffmpeg"), *CANDIDATES]:
        if c and os.path.exists(c):
            try:
                out = subprocess.run([c, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30).stdout
            except Exception:                                    # noqa: BLE001
                continue
            if "libx264" in out:
                return c
    return None


def _native(path: str) -> str:
    """a path the ffmpeg binary understands: the Windows form for a .exe under WSL interop"""
    return subprocess.run(["wslpath", "-w", path], capture_output=True, text=True).stdout.strip() or path


def verify(path: str, n_frames: int) -> dict:
    """decode the file again: all frames present, the first frame not black, the codec H.264"""
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    ok, frame = cap.read()
    cap.release()
    codec = "".join(chr((fcc >> 8 * i) & 255) for i in range(4)).lower()
    if not ok or frame is None or frame.max() == 0 or n != n_frames:
        raise RuntimeError(f"{path}: read back {n} frames (expected {n_frames}), first frame {'ok' if ok else 'unreadable'}")
    if codec not in ("h264", "avc1"):         # OpenCV decodes MPEG-4 part 2 ('FMP4') as well, which browsers refuse
        raise RuntimeError(f"{path}: codec {codec!r}, not H.264 - it would not play in a browser")
    return {"frames": n, "size": (frame.shape[1], frame.shape[0])}


def _as_files(frames: Sequence[Union[str, np.ndarray]], tmp: str):
    from PIL import Image
    out = []
    for i, f in enumerate(frames):
        if isinstance(f, str):
            out.append(f)
        else:
            p = os.path.join(tmp, f"in_{i:05d}.png")
            Image.fromarray(np.asarray(f, np.uint8)).save(p)
            out.append(p)
    return out


def encode(frames: Sequence[Union[str, np.ndarray]], out: str, fps: float = 2.0, crf: int = 20) -> dict:
    frames = list(frames)
    if not frames:
        raise ValueError("no frames")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ff = find_ffmpeg()
    with tempfile.TemporaryDirectory(dir=os.path.dirname(out)) as tmp:
        files = _as_files(frames, tmp)
        if ff:
            seq = os.path.join(tmp, "seq")
            os.makedirs(seq)
            for i, f in enumerate(files):                          # a numbered copy: ffmpeg's image2 pattern
                if os.name == "posix" and not ff.endswith(".exe"):
                    os.symlink(os.path.abspath(f), os.path.join(seq, f"{i:05d}.png"))
                else:
                    shutil.copyfile(f, os.path.join(seq, f"{i:05d}.png"))
            win = ff.endswith(".exe")
            cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps), "-i",
                   _native(os.path.join(seq, "%05d.png")) if win else os.path.join(seq, "%05d.png"),
                   "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", "-c:v", "libx264", "-crf", str(crf),
                   "-preset", "medium", "-movflags", "+faststart", _native(out) if win else out]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(out):
                raise RuntimeError(f"ffmpeg failed: {r.stderr[-400:]}")
            return {"path": out, "codec": "h264", "method": os.path.basename(ff), **verify(out, len(files))}
        from PIL import Image
        ims = [Image.open(f).convert("RGB") for f in files]
        gif = os.path.splitext(out)[0] + ".gif"
        ims[0].save(gif, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=True)
        return {"path": gif, "codec": "gif", "method": "PIL (no ffmpeg with libx264 found)", "frames": len(ims),
                "size": ims[0].size}
