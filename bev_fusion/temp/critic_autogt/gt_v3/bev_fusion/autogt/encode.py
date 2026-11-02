"""Frames -> a video that plays everywhere (H.264, yuv420p, mp4 with the index up front), checked by reading it back.

cv2.VideoWriter('mp4v') makes MPEG-4 part 2, which browsers, the Claude app and most players refuse (2026-10-30: the first GT
videos did not play). H.264 needs an ffmpeg with libx264: looked up in $FFMPEG, the PATH and the ffmpeg.exe builds installed on
the Windows side (called through WSL interop). Without one an animated GIF is written and the result says so. The file is decoded
again after encoding (frame count, first frame not black); a video that does not play is never reported as done.
(Same method as evaluation/video_encode.py, kept here so that autogt stays standalone.)
"""
import os
import shutil
import subprocess
import tempfile

CANDIDATES = ("/mnt/c/Program Files/DownloadHelper CoApp/ffmpeg.exe",
              "/mnt/c/Program Files (x86)/Digiarty/VideoProc Converter AI/ffmpeg.exe")


def find_ffmpeg():
    """path of an ffmpeg with libx264, or None."""
    for c in (os.environ.get("FFMPEG"), shutil.which("ffmpeg"), *CANDIDATES):
        if c and os.path.exists(c):
            try:
                if "libx264" in subprocess.run([c, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30).stdout:
                    return c
            except Exception:                                    # noqa: BLE001
                continue
    return None


def _native(path):
    return subprocess.run(["wslpath", "-w", path], capture_output=True, text=True).stdout.strip() or path


def verify(path, n_frames):
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None or frame.max() == 0 or n != n_frames:
        raise RuntimeError(f"{path}: read back {n} frames (expected {n_frames}), first frame {'ok' if ok else 'unreadable'}")
    return {"frames": n, "size": (frame.shape[1], frame.shape[0])}


def encode(frames, out, fps=12.0, crf=20):
    """frames: list of image paths in order -> out (.mp4); returns dict(path, codec, frames, size, method)."""
    frames = list(frames)
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ff = find_ffmpeg()
    if ff:
        exe = ff.endswith(".exe")
        with tempfile.TemporaryDirectory(dir=os.path.dirname(out)) as tmp:
            for i, f in enumerate(frames):                       # a numbered copy: ffmpeg's image2 pattern
                shutil.copyfile(f, os.path.join(tmp, f"{i:05d}.png"))
            pat = os.path.join(tmp, "%05d.png")
            cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps), "-i", _native(pat) if exe else pat,
                   "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", "-c:v", "libx264", "-crf", str(crf),
                   "-preset", "medium", "-movflags", "+faststart", _native(out) if exe else out]
            r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"ffmpeg failed: {r.stderr[-400:]}")
        return {"path": out, "codec": "h264", "method": os.path.basename(ff), **verify(out, len(frames))}
    from PIL import Image
    ims = [Image.open(f).convert("RGB") for f in frames]
    gif = os.path.splitext(out)[0] + ".gif"
    ims[0].save(gif, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return {"path": gif, "codec": "gif", "method": "PIL (no ffmpeg with libx264 found)", "frames": len(ims), "size": ims[0].size}
