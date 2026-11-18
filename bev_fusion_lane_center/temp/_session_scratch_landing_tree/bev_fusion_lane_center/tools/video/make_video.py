"""Result videos for lane graphs: one panel per frame (prediction red over ground truth green), written as mp4 (OpenCV, codec mp4v) or gif (imageio / PIL).

  frames_to_video(frames, out, fps=10)                 list of RGB arrays -> mp4 / gif (by file extension)
  graphs_to_video(preds, gts, out, fps=10, every=1)    renders every frame with tools/viz/render.frame_array and writes the video
  side_by_side(arrays_left, arrays_right)              concatenate two frame lists horizontally (e.g. rule graph | learned graph)
No ffmpeg binary is needed (not installed in the gcar env); mp4v plays in VLC / Windows Media Player, use gif for slides.
Usage: tools/env/_py14.sh video/make_video.py --pred pred.pkl --gt gt.pkl --out run.mp4 --fps 8 --every 2 --limit 60
"""
import argparse
import os
import pickle

import numpy as np

import render as RD  # tools/viz is on PYTHONPATH (tools/env/_py14.sh)


def frames_to_video(frames, out, fps=10):
    if not len(frames):
        raise ValueError("no frames to write into " + out)
    frames = [np.ascontiguousarray(f[..., :3]) for f in frames]
    h, w = frames[0].shape[:2]
    h, w = h - h % 2, w - w % 2
    frames = [f[:h, :w] for f in frames]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if out.lower().endswith(".gif"):
        import imageio

        imageio.mimsave(out, frames, duration=1.0 / fps, loop=0)
        return out
    import cv2

    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open " + out)
    for f in frames:
        vw.write(f[..., ::-1])  # RGB -> BGR
    vw.release()
    return out


def side_by_side(left, right):
    """each right picture cut or padded with white to the height of its left one (its width stays), then joined to its right"""
    out = []
    for lf, rt in zip(left, right):
        h = lf.shape[0]
        fit = np.full((h,) + rt.shape[1:], 255, rt.dtype)
        fit[: min(h, rt.shape[0])] = rt[:h]
        out.append(np.concatenate([lf, fit], axis=1))
    return out


def graphs_to_video(preds, gts, out, fps=10, every=1, limit=None, titles=None, metrics=None, size=5.0):
    idx = list(range(0, len(preds), every))[:limit]
    frames = [RD.frame_array(preds[i], gts[i] if gts is not None else None, title=(titles[i] if titles else f"frame {i}"), metrics=(metrics[i] if metrics else None), size=size) for i in idx]
    return frames_to_video(frames, out, fps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    P = pickle.load(open(a.pred, "rb"))
    G = pickle.load(open(a.gt, "rb")) if a.gt else None
    print("written", graphs_to_video(P, G, a.out, a.fps, a.every, a.limit))
