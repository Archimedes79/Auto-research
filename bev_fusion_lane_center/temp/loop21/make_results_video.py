"""Loop 21: results video of one system on VAL scenes (CPU, cached raw outputs, 1 thread).

    wsl bash tools/env/_py14.sh temp/loop21/make_results_video.py [registry name]        (default: recall)

Per frame: front camera image (left, top), the lane graph in the ego frame (right: prediction red over ground truth green, +-50 m), a text block (system, scene, frame, per-frame F1@1.0).
Scenes: the Boston and the Singapore scene of VAL whose scene Q is closest to the median scene Q of the system (not the best ones). Output: reports/weekly/Lane_graph_<name>_val.mp4
and a poster png. Layer 0: every used frame is matched to its camera info by timestamp (nearest, within 60 ms, one-to-one on all 1706 frames; the poses file and the infos are ordered differently), so image and graph belong to the same keyframe.
"""
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop20"))
import e255_matched as E  # noqa: E402
import evaluate as EV  # noqa: E402
import make_video as MV  # noqa: E402
import p16_infer as PI  # noqa: E402
import render as RD  # noqa: E402
import sparse_metrics as SM  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "recall"
OUT = f"{EV.R}/reports/weekly/Lane_graph_{name}_val.mp4"
E.load(name)
S = E.S
gts = S["gts"]
g = [PI.assemble(r, None, S["tau"], S["cfg"], **S["asm"]) for r in S["raw"]]
stored = json.load(open(f"{EV.R}/runs/eval/{name}/val.json", encoding="utf-8-sig"))["metrics"]
m_all = SM.card_any(g, gts, extra=True)
assert abs(m_all["Q"] - stored["Q"]) <= 5e-4, "stored card not reproduced"
print(f"{name}: Q {m_all['Q']:.4f} vert100 {m_all['vert100']:.2f} lanes found {m_all['lanes_found']:.3f} (stored card reproduced)")

sp = json.load(open(f"{EV.R}/data/splits.json"))
rows = np.array(sp["val"])
pz = np.load(f"{EV.HOME}/exp_gcar_b123/poses_val.npz")
scene, loc, ts = pz["scene"][rows], pz["loc"][rows], pz["ts"][rows]
infos = pickle.load(open(f"{EV.HOME}/data/nuscenes_full/nuscenes_infos_val_crn.pkl", "rb"))
INFO_TS = np.array([i["timestamp"] for i in infos], dtype=np.float64) / 1e6  # the poses file is ordered differently from the infos: match by timestamp
NEAR = np.array([int(np.argmin(np.abs(INFO_TS - x))) for x in pz["ts"]])
assert len(set(NEAR.tolist())) == len(NEAR), "the nearest-timestamp match is not one-to-one"
LOCN = {0: "Boston seaport", 1: "Singapore One-North", 2: "Singapore Holland Village", 3: "Singapore Queenstown"}

# scene Q and the choice of two representative scenes
sq = {}
for s in np.unique(scene):
    ix = np.nonzero(scene == s)[0]
    sq[int(s)] = (SM.quality(SM.card_any([g[i] for i in ix], [gts[i] for i in ix])), len(ix), int(loc[ix[0]]))
med = float(np.median([v[0] for v in sq.values()]))
pick = []
for want in (0, 1):  # Boston, Singapore One-North
    cand = [(abs(v[0] - med), s) for s, v in sq.items() if v[2] == want]
    pick.append(min(cand)[1])
print("scene Q median", round(med, 4), "| chosen scenes", [(s, round(sq[s][0], 4), sq[s][1], LOCN[sq[s][2]]) for s in pick])

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    FONTB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
except OSError:
    FONT = FONTB = ImageFont.load_default()
W, H = 1328, 720
frames = []
poster = None
for s in pick:
    ix = [i for i in np.nonzero(scene == s)[0]]
    ix.sort(key=lambda i: ts[i])
    for n, i in enumerate(ix):
        j = int(np.argmin(np.abs(INFO_TS - ts[i])))
        assert abs(INFO_TS[j] - ts[i]) < 0.06, "no camera info within 60 ms of this frame"  # poses carry a sensor timestamp 33 - 42 ms off the sample timestamp; keyframes are 0.5 s apart
        info = infos[j]
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
        d.text((16, 380), "Lane graph from camera + radar", fill=(20, 30, 60), font=FONTB)
        d.text((16, 420), f"system: {name} (registry name), VAL scene {s}", fill=(40, 40, 40), font=FONT)
        d.text((16, 452), f"{LOCN[int(loc[i])]}   frame {n + 1} of {len(ix)}", fill=(40, 40, 40), font=FONT)
        d.text((16, 496), f"this frame: F1@1.0 {f1:.2f}   vertices {len({k for nd in g[i]['nodes'] for k in nd})}", fill=(40, 40, 40), font=FONT)
        d.text((16, 540), f"VAL 873 frames: Q {m_all['Q']:.3f}  lanes found {m_all['lanes_found']:.2f}", fill=(40, 40, 40), font=FONT)
        d.text((16, 572), f"vertices per 100 m {m_all['vert100']:.1f}", fill=(40, 40, 40), font=FONT)
        d.text((16, 626), "red = prediction   green = ground truth", fill=(120, 20, 20), font=FONT)
        d.text((16, 658), "camera image shown for orientation only", fill=(110, 110, 110), font=FONT)
        arr = np.asarray(img)
        frames.append(arr)
        if poster is None and n == len(ix) // 2:
            poster = arr
MV.frames_to_video(frames, OUT, fps=3)
Image.fromarray(poster).save(OUT.replace(".mp4", "_poster.png"))
print(f"{len(frames)} frames -> {OUT}")
