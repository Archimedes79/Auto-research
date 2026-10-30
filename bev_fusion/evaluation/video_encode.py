"""Frames -> a video that plays everywhere (H.264, yuv420p, mp4 with the index up front), checked by reading it back.

    encode(frames, out_mp4, fps)           -> {"path", "codec", "frames", "size", "method"}; raises if nothing plays
    wsl ~/miniconda3/envs/gcar/bin/python evaluation/video_encode.py <frames dir or *.png list> out.mp4 [fps]
    wsl ~/miniconda3/envs/gcar/bin/python evaluation/video_encode.py --selftest

Why: cv2.VideoWriter('mp4v') makes MPEG-4 part 2, which browsers, the Claude app and most players refuse (2026-10-30: the first
scene video did not play). H.264 needs an ffmpeg with libx264: looked up in $FFMPEG, the PATH and the ffmpeg.exe builds that are
installed on the Windows side (called through WSL interop). Without any ffmpeg an animated WebP / GIF is written instead (plays
in browsers, bigger) and the result says so. After encoding the file is decoded with OpenCV: frame count and a non-black
first frame must match, otherwise the call fails (a video that does not play is never reported as done).
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

CANDIDATES = ("/mnt/c/Program Files/DownloadHelper CoApp/ffmpeg.exe",
              "/mnt/c/Program Files (x86)/Digiarty/VideoProc Converter AI/ffmpeg.exe")


def find_ffmpeg():
    """path of an ffmpeg with libx264, or None."""
    cands = [os.environ.get("FFMPEG"), shutil.which("ffmpeg"), *CANDIDATES]
    for c in cands:
        if c and os.path.exists(c):
            try:
                out = subprocess.run([c, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30).stdout
            except Exception:                                    # noqa: BLE001
                continue
            if "libx264" in out:
                return c
    return None


def _native(path):
    """a path the ffmpeg binary understands: Windows form for a .exe under WSL interop."""
    return subprocess.run(["wslpath", "-w", path], capture_output=True, text=True).stdout.strip() or path


def encode(frames, out, fps=2.0, crf=20):
    """frames: list of png/jpg paths in order -> out (.mp4). -> dict(path, codec, frames, size, method)."""
    frames = list(frames)
    if not frames:
        raise ValueError("no frames")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ff = find_ffmpeg()
    if ff:
        with tempfile.TemporaryDirectory(dir=os.path.dirname(out)) as tmp:        # a numbered copy: ffmpeg's image2 pattern
            for i, f in enumerate(frames):
                os.symlink(os.path.abspath(f), os.path.join(tmp, f"{i:05d}.png")) if os.name == "posix" and not ff.endswith(".exe") \
                    else shutil.copyfile(f, os.path.join(tmp, f"{i:05d}.png"))
            cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps), "-i",
                   _native(os.path.join(tmp, "%05d.png")) if ff.endswith(".exe") else os.path.join(tmp, "%05d.png"),
                   "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", "-c:v", "libx264", "-crf", str(crf),
                   "-preset", "medium", "-movflags", "+faststart",
                   _native(out) if ff.endswith(".exe") else out]
            r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"ffmpeg failed: {r.stderr[-400:]}")
        info = verify(out, len(frames))
        return {"path": out, "codec": "h264", "method": os.path.basename(ff), **info}
    # fallback: animated WebP / GIF
    from PIL import Image
    ims = [Image.open(f).convert("RGB") for f in frames]
    gif = os.path.splitext(out)[0] + ".gif"
    ims[0].save(gif, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return {"path": gif, "codec": "gif", "method": "PIL (no ffmpeg with libx264 found)", "frames": len(ims), "size": ims[0].size}


def verify(path, n_frames):
    """decode the file again: all frames present, first frame not black."""
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None or frame.max() == 0 or n != n_frames:
        raise RuntimeError(f"{path}: read back {n} frames (expected {n_frames}), first frame {'ok' if ok else 'unreadable'}")
    return {"frames": n, "size": (frame.shape[1], frame.shape[0])}


def _selftest():
    import numpy as np
    import cv2
    with tempfile.TemporaryDirectory() as d:
        fr = []
        for i in range(6):
            im = np.full((135, 241, 3), 40 + 30 * i, np.uint8)                  # odd size on purpose
            cv2.putText(im, str(i), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            p = os.path.join(d, f"f{i}.png")
            cv2.imwrite(p, im)
            fr.append(p)
        r = encode(fr, os.path.join(d, "t.mp4"), fps=2.0)
        print(r)
        assert r["frames"] == 6 and r["size"][0] % 2 == 0 and r["size"][1] % 2 == 0, r
        if r["codec"] == "h264":
            head = open(r["path"], "rb").read(4096)
            assert b"avc1" in head or b"avc1" in open(r["path"], "rb").read(), "not H.264"
    print("video_encode self-test passed")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--selftest":
        _selftest()
    else:
        src = a[0]
        files = sorted(glob.glob(os.path.join(src, "*.png"))) if os.path.isdir(src) else a[:-1]
        print(encode(files, a[1] if os.path.isdir(src) else a[-1], float(a[2]) if len(a) > 2 and os.path.isdir(src) else 2.0))
