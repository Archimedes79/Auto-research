"""scratch: the plain-text body of the report mail (D46) for Mirko, from the stored tables and the digests (numbers are read, never typed), plus the two narrative files.
    wsl python3 bev_fusion/temp/t16/build_mail2.py > bev_fusion/results/BEV_result_report/mail.txt   (from the repository root)"""
import json
import os
import subprocess
import sys

sys.path.insert(0, "bev_fusion/temp/t16")
from bigr_lib import md_tables, read_table  # noqa: E402

RUNS = [("t16_full_s0", "baseline"), ("t17_data", "more data"), ("t18_fix", "bundle")]
ROWS = [("drivable area IoU", "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0"), ("walkway IoU", "seg_fov/seg/dt+0.00/walkway/all/all/iou@0"), ("car park IoU", "seg_fov/seg/dt+0.00/carpark_area/all/all/iou@0"),
        ("pedestrian crossing IoU", "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0"), ("lane divider IoU (0.5 m tolerance)", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5"),
        ("road divider IoU (1 m tolerance)", "seg_fov/seg/dt+0.00/road_divider/all/all/iou@1"), ("lane divider zero-crossing F1 (0.5 m)", "seg_fov/seg/dt+0.00/lane_divider:zc/all/all/f1@0.5"),
        ("mAP (fusion area)", "detf/dt+0.00/mean/all/mAP"), ("vehicle AP@2 m", "detf/dt+0.00/vehicle/all/ap@2"), ("pedestrian AP@2 m", "detf/dt+0.00/pedestrian/all/ap@2"),
        ("velocity error, moving box centres (m/s, lower better)", "vel/dt+0.00/centre/field/moving/epe"), ("velocity error, static cells (m/s, lower better)", "vel/dt+0.00/cells/static/epe")]


def timing_lines():
    """ms per frame of the three runs from bev_fusion/runs/<run>/eval/timing_modules.json (only a result marked idle)"""
    res = {}
    for r, l in RUNS:
        p = f"bev_fusion/runs/{r}/eval/timing_modules.json"
        if os.path.exists(p):
            res[r] = json.load(open(p, encoding="utf-8"))
    if not res:
        return ["(no idle-GPU timing of the large runs was obtained)"]
    out = []
    for r, l in RUNS:
        d = res.get(r)
        if not d:
            out.append(f"  {l:10s} no timing")
            continue
        inf = d["inference_ms"]
        mods = sorted(inf["per_frame_ms"].items(), key=lambda kv: -kv[1])[:5]
        out.append(f"  {l:10s} {inf['total_ms']:6.1f} ms per frame ({'idle GPU' if d.get('idle') else 'NOT idle: do not quote'}); largest modules: " + ", ".join(f"{k} {v:.1f}" for k, v in mods))
    return out


def main():
    have = [(r, l, read_table(r)) for r, l in RUNS if read_table(r)]
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = []
    narr = "bev_fusion/temp/t16/mail_narrative2.txt"
    out.append(open(narr, encoding="utf-8").read().rstrip("\n") if os.path.exists(narr) else "(narrative missing)")
    out.append("")
    out.append("=== TABLE (validation set: 1000 frames of 82 scenes, ground truth v8, maps on front-camera cells and detection on the fusion area, ONE training seed per run; baseline = the delivered system, more data = 700 scenes, bundle = module fixes + smaller learning rate of the motion block) ===")
    w = max(len(n) for n, _ in ROWS) + 2
    out.append(" " * w + "".join(f"{l:>12s}" for _, l, _ in have))
    for name, key in ROWS:
        out.append(f"{name:{w}s}" + "".join(f"{t[key]:12.3f}" if key in t else f"{'-':>12s}" for _, _, t in have))
    out.append("")
    out.append("Cost (batch 1, three time offsets, cached chain, idle RTX 3060; training: 3.6 s per optimiser step):")
    out += timing_lines()
    out.append("")
    out.append("4. AGAINST THE STATE OF THE ART (same frames, same ground truth, same metric code; details and caveats in the report, section 6.3)")
    q = "bev_fusion/temp/t16/soa/soa_maps_v8_t16_full_s0.md"
    if os.path.exists(q):
        t = md_tables(open(q, encoding="utf-8").read())[0][2]
        names = {"drivable_area": "drivable area", "walkway": "walkway", "carpark_area": "car park", "ped_crossing": "pedestrian crossing", "lane_divider": "lane divider", "road_divider": "road divider"}
        out.append("Maps, ours / GaussianCaR (6 cameras + radar, 360 degrees), 384 validation frames, threshold 0.5 (plain IoU for areas, 0.5 m tolerance for lines):")
        for r in t:
            if len(r) >= 7 and ((r[0] in ("drivable_area", "walkway", "carpark_area", "ped_crossing") and r[1] == "iou@0") or (r[0] in ("lane_divider", "road_divider") and r[1] == "iou@0.5")):
                out.append(f"  {names[r[0]]:22s} {r[4]} / {r[5]}   (ours as a share: {r[6]})")
    f = "bev_fusion/temp/t16/soa/front_v8_n1000.md"
    if os.path.exists(f):
        fr = md_tables(open(f, encoding="utf-8").read())[0][2]
        out.append(f"Detection, front sector 0 - 50 m, 1000 frames: camera-only DualBEV mAP2 {fr[0][1]}, our baseline {fr[1][1]} (difference {fr[1][2]}); the radar is the difference, see the report.")
    out.append("")
    end = "bev_fusion/temp/t16/mail_narrative2_end.txt"
    out.append(open(end, encoding="utf-8").read().rstrip("\n") if os.path.exists(end) else "(closing narrative missing)")
    out.append("")
    out.append(f"Repository: github.com/Archimedes79/Auto-research, main at {head} when this mail was written.")
    print("\n".join(out))


if __name__ == "__main__":
    main()
