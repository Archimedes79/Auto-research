import json, os, re
B = "/mnt/c/Auto-research/bev_fusion/"
keys = {"detf mAP": "detf/dt+0.00/mean/all/mAP", "det mAP": "det/dt+0.00/mean/all/mAP",
        "lane_div IoU@0.5": "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5", "road_div IoU@0.5": "seg_fov/seg/dt+0.00/road_divider/all/all/iou@0.5",
        "drivable IoU@0": "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0", "walkway IoU@0": "seg_fov/seg/dt+0.00/walkway/all/all/iou@0",
        "ped_crossing IoU@0": "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0", "carpark IoU@0": "seg_fov/seg/dt+0.00/carpark_area/all/all/iou@0",
        "vel EPE moving": "vel/dt+0.00/centre/field/moving/epe"}
def card(p):
    d = json.load(open(B + p)); return d.get("metrics", d), d
st, _ = card("runs/t20_base_s0/eval/card.json")
for tag in ("eval_fusion_n1000", "eval_fusion_nall"):
    pass
import glob
cands = sorted(glob.glob(B + "temp/t26/base_run/eval/**/card*.json", recursive=True))
print("cards:", [c.replace(B, "") for c in cands])
for c in cands:
    m, d = card(c.replace(B, ""))
    print("==", c.replace(B, ""), "n =", d.get("n"), "gt =", (d.get("protocol") or {}).get("gt_signature") or d.get("gt_signature"))
    print(f"{'metric':22s}{'stored v9 (1000)':>18s}{'now':>10s}{'shift':>9s}")
    for k, key in keys.items():
        a, b = st.get(key), m.get(key)
        print("%-22s%18s%10s%9s" % (k, "-" if a is None else round(a,4), "-" if b is None else round(b,4), "" if (a is None or b is None) else round(b-a,4)))
