"""t11 box velocity, step 0b: velocity skill of the stored detections, with the evaluation's own meter (evaluation/metrics.py DetMeter:
front field of view, true positives at TP_THRESHOLD, moving = GT speed >= MOVING_SPEED) on the (detections, GT) pairs of
runs/<run>/eval/scene_records*.pkl. ave0 = the error of always saying 0 m/s on the same pairs; skill = 1 - ave / ave0.
Run (from bev_fusion/): ~/miniconda3/envs/gcar/bin/python temp/box_vel/stored_ave.py runs/t1_r0/eval/scene_records.pkl ...
"""
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch"), ROOT):
    sys.path.insert(0, p)
from evaluation.metrics import DetMeter                                   # noqa: E402

CLASSES = ["vehicle", "two_wheeler", "pedestrian"]


def score(path, dts=(0.0, 0.5, -0.5)):
    d = pickle.load(open(path, "rb"))
    m = DetMeter(classes=CLASSES)
    out = {}
    for dt in dts:
        fr = [m.frame_velocity(m.keep(ds), m.keep(gs), CLASSES) for sc in d["det"].values() for ds, gs in sc.get(dt, [])]
        if fr:
            out[dt] = m.reduce_velocity(fr, CLASSES)["vehicle"]
    return out


if __name__ == "__main__":
    keys = ("ave_moving", "ave0_moving", "ave_skill_moving", "n_tp_moving", "ave_static", "ave0_static", "n_tp_static")
    print(f"{'pickle':52s} {'dt':>5s} " + " ".join(f"{k:>16s}" for k in keys))
    for p in sys.argv[1:]:
        for dt, v in score(p).items():
            print(f"{os.path.relpath(p, 'runs'):52s} {dt:5.1f} " + " ".join(f"{float(v.get(k, float('nan'))):16.3f}" for k in keys))
