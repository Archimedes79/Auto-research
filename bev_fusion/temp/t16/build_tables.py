"""scratch of the t16 report (D46): markdown tables from the stored evaluation tables of runs (numbers from the files, never typed).
    python bev_fusion/temp/t16/build_tables.py --n 1000 name1[:label1] name2[:label2] ...   (run folder names under bev_fusion/runs; the first is the reference of the delta column)
Reads runs/<name>/eval/by_n/table_n<N>.txt (the suite's `eval` table), eval/timing_modules.json, runs/<name>/config.json, log.jsonl."""
import argparse
import json
import os
import re

ROOT = "bev_fusion/runs"
MAPS = [("drivable_area", "iou@0"), ("walkway", "iou@0"), ("carpark_area", "iou@0"), ("ped_crossing", "iou@0"), ("lane_divider", "iou@0.5"), ("road_divider", "iou@1")]
LAYERS = ["drivable_area", "walkway", "carpark_area", "ped_crossing", "lane_divider", "road_divider"]


def load_table(run, n):
    d = {}
    p = f"{ROOT}/{run}/eval/by_n/table_n{n}.txt"
    if not os.path.exists(p):
        return d
    for line in open(p, encoding="utf-8"):
        m = re.match(r"^(\S+)\s+(-?[0-9.]+(?:e-?[0-9]+)?|nan)\s*$", line.rstrip())
        if m:
            d[m.group(1)] = float(m.group(2))
    return d


def cell(v, ref=None, lower=False, fmt="{:.3f}"):
    if v is None:
        return "-"
    s = fmt.format(v)
    return s


def delta(v, ref, lower=False):
    if v is None or ref is None:
        return ""
    d = v - ref
    good = (d < 0) if lower else (d > 0)
    return f" ({d:+.3f}{'' if abs(d) < 1e-9 else (' +' if good else ' -')})"


def rows(spec, tabs, labels):
    out = ["| metric | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for name, key, lower in spec:
        vals = [t.get(key) for t in tabs]
        cells = [cell(vals[0])] + [cell(v) + delta(v, vals[0], lower) for v in vals[1:]]
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    names = [r.split(":")[0] for r in a.runs]
    labels = [r.split(":")[1] if ":" in r else r.split(":")[0] for r in a.runs]
    tabs = [load_table(r, a.n) for r in names]
    sp = []
    for lay, thr in MAPS:
        sp.append((f"{lay} IoU {thr} (dt 0)", f"seg_fov/seg/dt+0.00/{lay}/all/all/{thr}", False))
    sp += [("drivable IoU@0 at dt -0.5 s", "seg_fov/seg/dt-0.50/drivable_area/all/all/iou@0", False),
           ("drivable IoU@0 at dt +0.5 s", "seg_fov/seg/dt+0.50/drivable_area/all/all/iou@0", False)]
    sp += [(f"{lay} ROC-AUC", f"curve/dt+0.00/{lay}/roc_auc", False) for lay in LAYERS]
    sp += [(f"{lay} AP", f"curve/dt+0.00/{lay}/ap", False) for lay in LAYERS]
    print("### Maps (VAL, fusion-area scoring `seg_fov`, thresholded IoU; curves over all cells)\n")
    print(rows(sp, tabs, labels))
    sd = [("mAP, fusion area (detf)", "detf/dt+0.00/mean/all/mAP", False), ("mAP, fusion area + partial (detfp)", "detfp/dt+0.00/mean/all/mAP", False),
          ("vehicle AP@2 m", "detf/dt+0.00/vehicle/all/ap@2", False), ("pedestrian AP@2 m", "detf/dt+0.00/pedestrian/all/ap@2", False),
          ("two-wheeler AP@2 m", "detf/dt+0.00/two_wheeler/all/ap@2", False), ("vehicle translation error (ate, m)", "detf/dt+0.00/vehicle/all/ate", True),
          ("vehicle IoU@0 (dense output)", "seg_obj/seg/dt+0.00/vehicle/all/all/iou@0", False)]
    print("\n### Objects (VAL)\n")
    print(rows(sd, tabs, labels))
    sv = [("velocity EPE at moving box centres (m/s)", "vel/dt+0.00/centre/field/moving/epe", True), ("zero-predictor EPE at the same centres", "vel/dt+0.00/centre/zero/moving/epe", True),
          ("EPE vehicle_moving cells", "vel/dt+0.00/cells/vehicle_moving/epe", True), ("EPE pedestrian_moving cells", "vel/dt+0.00/cells/pedestrian_moving/epe", True),
          ("EPE static cells", "vel/dt+0.00/cells/static/epe", True)]
    print("\n### Velocity (VAL; lower is better)\n")
    print(rows(sv, tabs, labels))
    sl = [(f"val loss {k}", f"val_loss/{k}", True) for k in ("group.areas", "group.lines", "group.objects", "group.motion", "total")]
    print("\n### Validation loss terms (lower is better)\n")
    print(rows(sl, tabs, labels))
    # cost
    print("\n### Cost\n")
    head = ["| item | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    cost = {"inference ms per frame (batch 1, 3 dts, idle GPU)": [], "training step s (median of the log, accum 2)": [], "parameters trained (M)": []}
    for r in names:
        tj = f"{ROOT}/{r}/eval/timing_modules.json"
        t = json.load(open(tj)) if os.path.exists(tj) else {}
        tot = (t.get("inference_ms") or {}).get("total")
        cost["inference ms per frame (batch 1, 3 dts, idle GPU)"].append("-" if tot is None else f"{tot:.1f}")
        lg = f"{ROOT}/{r}/log.jsonl"
        sps = []
        if os.path.exists(lg):
            for ln in open(lg):
                try:
                    d = json.loads(ln)
                except ValueError:
                    continue
                if "t_step" in d:
                    sps.append(d["t_step"])
        sps = sorted(sps[len(sps) // 3:])           # the cold first third is left out
        cost["training step s (median of the log, accum 2)"].append("-" if not sps else f"{sps[len(sps) // 2]:.2f}")
        pj = t.get("params") or {}
        cost["parameters trained (M)"].append("-" if not pj else f"{sum(pj.values()) / 1e6:.2f}")
    for k, v in cost.items():
        head.append(f"| {k} | " + " | ".join(v) + " |")
    print("\n".join(head))


if __name__ == "__main__":
    main()
