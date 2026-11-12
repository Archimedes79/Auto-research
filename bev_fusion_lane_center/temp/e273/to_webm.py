"""one-off: transcode the FMP4 videos of the E273 result folders to WebM (VP9) so that a browser can play them; usage: to_webm.py <in.mp4> <out.webm> [width] [crf]"""
import os, sys
os.environ.setdefault("OPENCV_FFMPEG_WRITER_OPTIONS", "video_codec;libvpx-vp9|crf;34|b;0|deadline;good|cpu-used;4")
import cv2, numpy as np

src, dst = sys.argv[1], sys.argv[2]
W = int(sys.argv[3]) if len(sys.argv) > 3 else 1600
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 2.0
w0, h0 = int(cap.get(3)), int(cap.get(4))
H = int(round(h0 * W / w0 / 2) * 2)
wr = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"vp09"), fps, (W, H))
n = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    wr.write(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA))
    n += 1
wr.release()
print(os.path.basename(dst), n, "frames", W, "x", H, round(os.path.getsize(dst) / 1e6, 2), "MB (source", round(os.path.getsize(src) / 1e6, 2), "MB)")
