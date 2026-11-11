"""Video: driving through the GT of one scene at the 12 Hz front-camera rate: python -m GtGenerator video SCENE OUT.mp4 [seconds]

Left the camera image with the GT projected (map lines: lane divider yellow, road divider orange, centre-line cyan; 3D boxes:
vehicle red, pedestrian magenta, two-wheeler orange, white rim = stationary, grey = no lidar / radar point); right the BEV GT
in the ego frame at that frame (semantic layers dimmed outside the camera view, boxes with heading, speed in km/h).
- map layers: the exact generator at the pose of EVERY camera frame (keyframes and sweeps)
- boxes: annotated at 2 Hz only; GtGenerator.flow interpolates the tracks to the camera time of every frame (stationary boxes use
  their one fixed world pose) and gives the radar-corrected velocity
H.264 (`encode` below: ffmpeg with libx264, read back after encoding), 12 fps.
"""
import math
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image

from . import flow as flowmod
from .grid import GridSpec
from .nuscenes_adapter import NuScenesScene
from .suite import GtGenerator
from .view import EDGES, densify, semantic_rgb

# ---- frames -> a video that plays everywhere (H.264, yuv420p, mp4 with the index up front), checked by reading it back
# cv2.VideoWriter('mp4v') makes MPEG-4 part 2, which browsers, the Claude app and most players refuse (2026-10-30: the first GT videos
# did not play). H.264 needs an ffmpeg with libx264: looked up in $FFMPEG, the PATH and the ffmpeg.exe builds installed on the Windows
# side (called through WSL interop). Without one an animated GIF is written and the result says so. The file is decoded again after
# encoding (frame count, first frame not black); a video that does not play is never reported as done. (Same method as
# autoresearch/Evaluation/render/video.py, kept here so that GtGenerator stays standalone.)
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
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("latin-1").lower()
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None or frame.max() == 0 or n != n_frames:
        raise RuntimeError(f"{path}: read back {n} frames (expected {n_frames}), first frame {'ok' if ok else 'unreadable'}")
    if fourcc not in ("h264", "avc1"):        # OpenCV also decodes MPEG-4 part 2 (mp4v / fmp4), which browsers refuse
        raise RuntimeError(f"{path}: codec {fourcc!r}, not H.264")
    return {"frames": n, "size": (frame.shape[1], frame.shape[0]), "fourcc": fourcc}


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
    ims = [Image.open(f).convert("RGB") for f in frames]
    gif = os.path.splitext(out)[0] + ".gif"
    ims[0].save(gif, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return {"path": gif, "codec": "gif", "method": "PIL (no ffmpeg with libx264 found)", "frames": len(ims), "size": ims[0].size}


COL = {"vehicle": (50, 50, 240), "pedestrian": (230, 60, 230), "two_wheeler": (30, 150, 255)}        # BGR
LCOL = {"lane_divider": (25, 230, 255), "road_divider": (0, 128, 255), "centreline": (230, 230, 25)}
PXM = 9.0                                                                                           # BEV pixels per metre


def _seg(img, K, a, b, color, thick, near=0.5):
    if a[2] < near and b[2] < near:
        return
    if a[2] < near or b[2] < near:
        t = (near - a[2]) / (b[2] - a[2])
        c = a + t * (b - a)
        a, b = (c, b) if a[2] < near else (a, c)
    uv = [(int(round(K[0, 0] * p[0] / p[2] + K[0, 2])), int(round(K[1, 1] * p[1] / p[2] + K[1, 2]))) for p in (a, b)]
    cv2.line(img, uv[0], uv[1], color, thick, cv2.LINE_AA)


def _px(x, y):
    return int(round((26.0 - y) * PXM)), int(round((60.0 - x) * PXM))


def _bev_image(rgb):
    """a (nx, ny, 3) map in 0..1 -> the BGR picture of the video's bird's-eye panel: x forward = up, +y left = left (the overlay uses _px)."""
    return cv2.resize((np.clip(np.flip(np.flip(rgb, 0), 1), 0, 1) * 255).astype(np.uint8)[..., ::-1],
                      (int(52 * PXM), int(60 * PXM)), interpolation=cv2.INTER_NEAREST)


def _corners(o, z):
    """3D corners of an ObjectFlow in the world frame."""
    c, s = math.cos(o.yaw), math.sin(o.yaw)
    w, l, h = o.size
    loc = np.array([[sx * l / 2, sy * w / 2, sz * h / 2] for sz in (-1, 1) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    return np.stack([o.xy[0] + c * loc[:, 0] - s * loc[:, 1], o.xy[1] + s * loc[:, 0] + c * loc[:, 1], z + loc[:, 2]], 1)


def render(layers, scene, sd, kf_token, header, root, use_radar):
    grid = GridSpec()
    gt = layers.keyframe(kf_token, grid, sd=sd)
    meta = gt["meta"]
    T_g2e = np.linalg.inv(meta["T_e2g"])
    K, T_e2c = meta["K"], np.linalg.inv(meta["T_c2e"])
    t = sd["timestamp"] * 1e-6
    fg = flowmod.estimate_flow(scene.tracks, scene.ego, t, scene.radar(t) if use_radar else ())
    by_id = {tr.id: tr for tr in scene.tracks}
    img = cv2.cvtColor(np.asarray(Image.open(os.path.join(root, sd["filename"])).convert("RGB")), cv2.COLOR_RGB2BGR)
    for layer, pls in meta["vectors"].items():
        for pl in pls:
            d = densify(pl, 1.0)
            keep = (d[:, 0] > -5) & (d[:, 0] < 70) & (np.abs(d[:, 1]) < 30)
            P = (np.c_[d, np.zeros(len(d)), np.ones(len(d))] @ T_e2c.T)[:, :3]
            for i in range(len(d) - 1):
                if keep[i] and keep[i + 1]:
                    _seg(img, K, P[i], P[i + 1], LCOL[layer], 2 if layer != "centreline" else 1)
    foot = []
    for o in fg.objects:
        tr = by_id[o.id]
        z = float(np.interp(t, tr.t, tr.extra["trans"][:, 2])) if "trans" in tr.extra else 0.0
        cw = _corners(o, z)
        ce = (np.c_[cw, np.ones(8)] @ T_g2e.T)[:, :3]
        foot.append((o, ce))
        if ce[:, 0].max() < 1 or ce[:, 0].min() > 80:
            continue
        P = (np.c_[ce, np.ones(8)] @ T_e2c.T)[:, :3]
        col = COL[o.group] if o.observed else (170, 170, 170)
        for i, j in EDGES:
            if o.stationary:
                _seg(img, K, P[i], P[j], (255, 255, 255), 4)
            _seg(img, K, P[i], P[j], col, 2)
    cam_panel = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
    rgb = semantic_rgb(gt) * (1 - 0.55 * (1 - gt["camera"]))[..., None]
    bev = _bev_image(rgb)
    Rinv = T_g2e[:2, :2]
    for o, ce in foot:
        fp = ce[:4, :2]
        if fp[:, 0].max() < 0 or fp[:, 0].min() > 60 or np.abs(fp[:, 1]).min() > 27:
            continue
        pts = np.array([_px(x, y) for x, y in fp], np.int32)
        cv2.fillPoly(bev, [pts], COL[o.group] if o.observed else (150, 150, 150))
        cv2.polylines(bev, [pts], True, (255, 255, 255) if o.stationary else (20, 20, 20), 1, cv2.LINE_AA)
        cx, cy = fp.mean(0)
        front = fp[:2].mean(0)
        cv2.line(bev, _px(cx, cy), _px(front[0], front[1]), (255, 255, 255), 1, cv2.LINE_AA)
        if o.observed and np.isfinite(o.v).all() and o.speed > 1.0 and 0 < cx < 60 and abs(cy) < 26:
            ve = Rinv @ o.v
            cv2.arrowedLine(bev, _px(cx, cy), _px(cx + ve[0], cy + ve[1]), (255, 255, 0), 1, cv2.LINE_AA, tipLength=0.25)
            cv2.putText(bev, f"{o.speed * 3.6:.0f}", _px(cx + 1.5, cy + 1.5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(bev, _px(0, 0), 5, (255, 255, 255), -1)
    out = np.full((540 + 44, 960 + bev.shape[1], 3), 255, np.uint8)
    out[44:, :960] = cam_panel
    out[44:44 + bev.shape[0], 960:] = bev
    cv2.putText(out, header[0], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(out, header[1], (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    return out[:out.shape[0] - out.shape[0] % 2, :out.shape[1] - out.shape[1] % 2]


def run(scene_name, out_path, max_s=0.0, dataroot="~/data/nuscenes", version="v1.0-mini", use_radar=True):
    root = os.path.expanduser(dataroot)
    # map layers per frame; velocities come from GtGenerator.flow; the memory-light devkit of this one sequence (the same GT, and trainval fits)
    layers = GtGenerator(root, version, use_flow=False, light=True, scenes=[scene_name])
    scene = NuScenesScene(layers.nusc, root, scene_name)
    toks = layers.keyframes(scene_name)
    key_t = [layers.nusc.get("sample", t)["timestamp"] * 1e-6 for t in toks]
    track = layers.camera_track(toks[0])
    frames = [sd for sd in track if key_t[0] <= sd["timestamp"] * 1e-6 <= key_t[-1]]
    if max_s:
        frames = frames[:int(max_s * 12)]
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=out_dir)
    try:                                                                  # a rendering or encoding error leaves no temporary pictures behind
        paths = []
        for n, sd in enumerate(frames):
            t = sd["timestamp"] * 1e-6
            kf = toks[int(np.argmin([abs(t - kt) for kt in key_t]))]
            i = track.index(sd)
            sp = 0.0
            if i + 1 < len(track):
                rel = np.linalg.inv(layers.ego_pose(sd)) @ layers.ego_pose(track[i + 1])
                sp = rel[0, 3] / max((track[i + 1]["timestamp"] - sd["timestamp"]) * 1e-6, 1e-3)
            head = (f"{scene_name}: {scene.scene['description'][:60]} | t = {t - key_t[0]:5.1f} s | ego {sp * 3.6:3.0f} km/h | "
                    f"{'KEYFRAME' if sd['is_key_frame'] else 'sweep'}",
                    "map layers exact per frame; boxes: 2 Hz annotations interpolated, radar-corrected velocity (km/h); white rim = stationary")
            frame = render(layers, scene, sd, kf, head, root, use_radar)
            paths.append(os.path.join(tmp, f"f{n:05d}.png"))
            if not cv2.imwrite(paths[-1], frame):                         # imwrite reports a failed write by its return value, not by an exception
                raise OSError(f"could not write {paths[-1]}")
            if n % 24 == 0:
                print(f"{n}/{len(frames)} frames", flush=True)
        res = encode(paths, out_path, fps=12)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("saved", res["path"], f"{res['frames']} frames, {res['frames'] / 12:.1f} s, {res['codec']} via {res['method']}")
    return res
