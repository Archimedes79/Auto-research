"""Results video of one system: front camera + lane graph per frame (CPU, cached raw outputs). Made in loop 21, moved from temp/loop21/ on 2026-10-31 (D28, second use).

    wsl bash tools/env/_py14.sh make_results_video.py [registry name]        (default: recall)

Per frame: front camera image (left, top), the lane graph in the ego frame (right: prediction red over ground truth green, +-50 m), a text block (system, scene, frame, per-frame F1@1.0).
Stand-alone: the Boston and the Singapore scene of VAL whose scene Q is closest to the median scene Q of the system (not the best ones), after the stored card
runs/eval/<name>/val.json is reproduced. Output: reports/weekly/Lane_graph_<name>_val.mp4 and a poster png.
As a library (lane_prototype.py --video): `frame_table(split)` (scene, location, timestamp, camera info of every frame of a split) and `video(...)` for any frames.
Layer 0: every frame is matched to its camera info by timestamp (nearest, within 60 ms, one-to-one on all 1706 frames; the poses file and the infos are ordered
differently), so image and graph belong to the same keyframe.
"""
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
import evaluate as EV  # noqa: E402
import sparse_metrics as SM  # noqa: E402

LOCN = {0: "Boston seaport", 1: "Singapore One-North", 2: "Singapore Holland Village", 3: "Singapore Queenstown"}
_INFOS = []  # the camera infos of the nuScenes val scenes, loaded once (15 s)


def frame_table(split):
    """per frame position of the split: scene id, location id, timestamp (s) and the camera info (scene_name, sample_token, cam_infos ...), matched by timestamp"""
    if not _INFOS:
        _INFOS.extend(pickle.load(open(f"{EV.HOME}/data/nuscenes_full/nuscenes_infos_val_crn.pkl", "rb")))
    info_ts = np.array([i["timestamp"] for i in _INFOS], dtype=np.float64) / 1e6
    pz = np.load(f"{EV.HOME}/exp_gcar_b123/poses_val.npz")
    near = np.array([int(np.argmin(np.abs(info_ts - x))) for x in pz["ts"]])
    assert len(set(near.tolist())) == len(near), "the nearest-timestamp match is not one-to-one"
    assert np.all(np.abs(info_ts[near] - pz["ts"]) < 0.06), "no camera info within 60 ms of a frame"  # poses carry a sensor timestamp 33 - 42 ms off the sample timestamp; keyframes are 0.5 s apart
    rows = np.array(json.load(open(f"{EV.R}/data/splits.json"))[EV.SPLITS[split][1]])
    return dict(scene=pz["scene"][rows], loc=pz["loc"][rows], ts=pz["ts"][rows], info=[_INFOS[j] for j in near[rows]])


def median_scenes(g, gts, ft):
    """the Boston and the Singapore One-North scene whose scene Q is closest to the median scene Q"""
    sq = {}
    for s in np.unique(ft["scene"]):
        ix = np.nonzero(ft["scene"] == s)[0]
        sq[int(s)] = (SM.quality(SM.card_any([g[i] for i in ix], [gts[i] for i in ix])), len(ix), int(ft["loc"][ix[0]]))
    med = float(np.median([v[0] for v in sq.values()]))
    pick = [min((abs(v[0] - med), s) for s, v in sq.items() if v[2] == want)[1] for want in (0, 1)]
    print("scene Q median", round(med, 4), "| chosen scenes", [(s, round(sq[s][0], 4), sq[s][1], LOCN[sq[s][2]]) for s in pick])
    return pick


def video(name, g, gts, ft, groups, header, out, split="val"):
    """mp4 (+ poster png) of the frames `groups` = [(scene id, [frame positions in time order])]; g / gts indexed by frame position; header = lines about the numbers of the set"""
    import make_video as MV
    import render as RD
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        fontb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = fontb = ImageFont.load_default()
    W, H = 1328, 720
    frames, poster = [], None
    for s, ix in groups:
        for n, i in enumerate(ix):
            info = ft["info"][i]
            cam = Image.open(f"{EV.HOME}/data/nuscenes_full/{info['cam_infos']['CAM_FRONT']['filename']}").convert("RGB").resize((640, 360))
            try:
                f1 = SM.card_any([g[i]], [gts[i]])["F1@1.0"]
            except Exception:  # noqa: BLE001
                f1 = float("nan")
            bev = Image.fromarray(RD.frame_array(g[i], gts[i], title="", size=7.2, dpi=100)).resize((688, 720))
            img = Image.new("RGB", (W, H), (255, 255, 255))
            img.paste(cam, (0, 0))
            img.paste(bev, (640, 0))
            d = ImageDraw.Draw(img)
            d.text((16, 380), "Lane graph from camera + radar", fill=(20, 30, 60), font=fontb)
            d.text((16, 420), f"system: {name} (registry name), {split.upper()} scene {s}", fill=(40, 40, 40), font=font)
            d.text((16, 452), f"{LOCN[int(ft['loc'][i])]}   frame {n + 1} of {len(ix)}", fill=(40, 40, 40), font=font)
            d.text((16, 496), f"this frame: F1@1.0 {f1:.2f}   vertices {len({k for nd in g[i]['nodes'] for k in nd})}", fill=(40, 40, 40), font=font)
            for k, line in enumerate(header[:2]):
                d.text((16, 540 + 32 * k), line, fill=(40, 40, 40), font=font)
            d.text((16, 626), "red = prediction   green = ground truth", fill=(120, 20, 20), font=font)
            d.text((16, 658), "camera image shown for orientation only", fill=(110, 110, 110), font=font)
            arr = np.asarray(img)
            frames.append(arr)
            if poster is None and n == len(ix) // 2:
                poster = arr
    MV.frames_to_video(frames, out, fps=3)
    Image.fromarray(poster).save(out.replace(".mp4", "_poster.png"))
    print(f"{len(frames)} frames -> {os.path.relpath(out, EV.R)}")
    return out


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "recall"
    g, gts, _ = EV.graphs_of(name, EV.registry()[name], "val", require_cache=True)
    stored = json.load(open(f"{EV.R}/runs/eval/{name}/val.json", encoding="utf-8-sig"))["metrics"]
    m = SM.card_any(g, gts, extra=True)
    assert abs(m["Q"] - stored["Q"]) <= EV.TOL, "stored card not reproduced"
    print(f"{name}: Q {m['Q']:.4f} vert100 {m['vert100']:.2f} lanes found {m['lanes_found']:.3f} (stored card reproduced)")
    ft = frame_table("val")
    groups = [(s, sorted(np.nonzero(ft["scene"] == s)[0], key=lambda i: ft["ts"][i])) for s in median_scenes(g, gts, ft)]
    header = [f"VAL {len(gts)} frames: Q {m['Q']:.3f}  lanes found {m['lanes_found']:.2f}", f"vertices per 100 m {m['vert100']:.1f}"]
    video(name, g, gts, ft, groups, header, f"{EV.R}/reports/weekly/Lane_graph_{name}_val.mp4")


if __name__ == "__main__":
    main()
