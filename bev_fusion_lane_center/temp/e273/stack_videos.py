"""one-off: comparison videos for the report: the best existing graph d (top) and the E273 head seed 0 (bottom) on the same frames, stacked, 1280 px wide; WebM (VP9) and MP4 (FMP4)

    wsl bash tools/env/_py14.sh stack_videos.py
reads results/E273.report/d_frames/<scene>/scene-XXXX_recall_sp0_dp.mp4 and results/E273.p_s0/videos/scene-XXXX_e273_p_s0.mp4, writes results/E273.report/vid/compare_<scene>.webm and results/E273.report/compare/*.mp4
"""
import glob
import os

import cv2
import numpy as np

R = os.environ["BFLC_ROOT"]
OUT = os.path.join(R, "results", "E273.report")
os.makedirs(os.path.join(OUT, "compare"), exist_ok=True)
W = 1280
for dv in sorted(glob.glob(os.path.join(OUT, "d_frames", "*", "scene-*_recall_sp0_dp.mp4"))):
    name = os.path.basename(dv).split("_")[0]
    ev = os.path.join(R, "results", "E273.p_s0", "videos", f"{name}_e273_p_s0.mp4")
    if not os.path.exists(ev):
        continue
    a, b = cv2.VideoCapture(dv), cv2.VideoCapture(ev)
    h = int(round(1080 * W / 1920 / 2) * 2)
    outs = []
    for ext, code, opts in (("webm", "vp09", "video_codec;libvpx-vp9|crf;36|b;0|deadline;good|cpu-used;4"), ("mp4", "mp4v", "")):
        if opts:
            os.environ["OPENCV_FFMPEG_WRITER_OPTIONS"] = opts
        else:
            os.environ.pop("OPENCV_FFMPEG_WRITER_OPTIONS", None)
        dst = os.path.join(OUT, "vid", f"compare_{name}.webm") if ext == "webm" else os.path.join(OUT, "compare", f"compare_{name}.mp4")
        outs.append((cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*code), 2.0, (W, 2 * h)), dst))
    n = 0
    while True:
        ok1, fa = a.read()
        ok2, fb = b.read()
        if not (ok1 and ok2):
            break
        fa, fb = (cv2.resize(f, (W, h), interpolation=cv2.INTER_AREA) for f in (fa, fb))
        cv2.putText(fa, "best graph so far (recall_sp0_dp)", (W - 420, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 160), 1, cv2.LINE_AA)
        cv2.putText(fb, "E273 lane head, seed 0", (W - 300, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 60, 0), 1, cv2.LINE_AA)
        st = np.vstack([fa, fb])
        for w, _ in outs:
            w.write(st)
        n += 1
    for w, dst in outs:
        w.release()
        print(name, n, "frames", os.path.basename(dst), round(os.path.getsize(dst) / 1e6, 2), "MB", flush=True)
