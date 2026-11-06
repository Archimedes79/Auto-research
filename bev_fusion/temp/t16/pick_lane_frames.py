"""scratch: lane-divider quality per frame from the stored evaluation store of a run, and the frames to draw (best / median / worst, with GT lane cells).
    wsl python3 bev_fusion/temp/t16/pick_lane_frames.py t16_full_s0 > bev_fusion/temp/t16/lane_frames_t16_full_s0.txt"""
import json
import math
import sys

run = sys.argv[1]
p = f"bev_fusion/runs/{run}/eval/suite/quant/eval_curves_fusion_vel_n1000/frames.jsonl"
rows = []
for line in open(p, encoding="utf-8"):
    d = json.loads(line.replace("NaN", "null"))
    s, sup = d["scores"], d["support"]
    iou, f1 = s.get("seg.lane_divider_iou"), s.get("seg.lane_divider_f1")
    n = sup.get("seg.lane_divider_iou") or 0
    rows.append((d["label"], iou, f1, n))
have = [r for r in rows if r[1] is not None and r[3] and r[3] >= 30]
print(f"{run}: {len(rows)} frames, {len(have)} with at least 30 GT lane-divider cells")
if not have:
    sys.exit()
v = sorted(r[1] for r in have)
q = lambda a: v[min(len(v) - 1, int(a * len(v)))]
print("lane-divider IoU over those frames: mean %.3f  q10 %.3f  q25 %.3f  median %.3f  q75 %.3f  q90 %.3f  share with IoU < 0.1: %.2f  share >= 0.5: %.2f" % (
    sum(v) / len(v), q(0.1), q(0.25), q(0.5), q(0.75), q(0.9), sum(x < 0.1 for x in v) / len(v), sum(x >= 0.5 for x in v) / len(v)))
have.sort(key=lambda r: r[1])


def distinct(rs, n):
    """the first n rows of different scenes"""
    out, seen = [], set()
    for r in rs:
        sc = r[0].split("/")[0]
        if sc not in seen:
            seen.add(sc)
            out.append(r)
        if len(out) == n:
            break
    return out


mid = len(have) // 2
around = sorted(have[mid - 60:mid + 60], key=lambda r: abs(r[1] - v[len(v) // 2]))
pick = {"worst": distinct(have, 4), "median": distinct(around, 4), "best": distinct(have[::-1], 4)}
for k, rs in pick.items():
    for r in rs:
        print(f"PICK {k:6s} {r[0]}  lane_iou {r[1]:.3f}  f1 {r[2] if r[2] is None else round(r[2], 3)}  gt_cells {int(r[3])}")
