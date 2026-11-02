"""Results of the 6000-step run t10_dual_long_s0 next to its controls, read from the stored eval files (cold VAL 1000, seg_fov, dt 0).
Nothing is typed by hand except the predictions of the run record (card 3c rev 2) and the thresholds of the card's rule.
Run from bev_fusion (WSL, stdlib only): python3 temp/t10/facts_long.py  ->  temp/t10/results_long.md, temp/t10/facts_long.json"""
import json
import os
import re
import statistics

RUNS = ["t10_dual_s0", "t10_dual_s1", "t11_vel_hurdle_acc2", "t10_dual_long_s0"]
CONTROLS = ["t10_dual_s0", "t10_dual_s1"]
VARIANT = "t10_dual_long_s0"
LAYERS = [("drivable", "drivable_area", "iou@0"), ("walkway", "walkway", "iou@0"), ("carpark", "carpark_area", "iou@0"),
          ("crossing", "ped_crossing", "iou@0"), ("lane divider @0.5 m", "lane_divider", "iou@0.5"), ("road divider @0.5 m", "road_divider", "iou@0.5")]
VEL = [("vehicle skill, moving", "det/dt+0.00/vehicle/all/ave_skill_moving"), ("vehicle error, static (m/s)", "det/dt+0.00/vehicle/all/ave_static"),
       ("vehicle error, moving (m/s)", "det/dt+0.00/vehicle/all/ave_moving"), ("vehicle AVE (m/s)", "det/dt+0.00/vehicle/all/ave"),
       ("mAVE (m/s)", "det/dt+0.00/mean/all/mAVE"), ("det mAP", "det/dt+0.00/mean/all/mAP")]
# the run record's prediction (card 3c rev 2): value, low, high
PRED = {"lane divider": (0.39, 0.355, 0.44), "walkway": (0.37, 0.34, 0.41), "crossing": (0.20, 0.16, 0.26), "drivable": (0.775, 0.76, 0.795),
        "front mAP2": (0.355, 0.33, 0.385), "vehicle skill": (0.55, 0.35, 0.70), "static error": (0.25, 0.15, 0.35)}
FRONT_MD = "experiments/t10_dual_features/results_x50_long.md"
BAND, GAIN_CONTINUE, GAIN_DROP, SKILL_MIN, STATIC_MAX, FRONT_MIN = 0.029, 0.04, 0.02, 0.50, 0.30, 0.307


def load(run):
    p = f"runs/{run}/eval/metrics.json"
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    d = m.get("metrics", m)
    cfg = {}
    if os.path.exists(f"runs/{run}/config.json"):
        cfg = json.load(open(f"runs/{run}/config.json"))
    return {"d": d, "sig": m.get("gt_signature", ""), "weights": m.get("weights", ""), "ckpt": m.get("checkpoint", ""), "cfg": cfg}


def seg(r, layer, tol):
    return r["d"].get(f"seg_fov/seg/dt+0.00/{layer}/all/all/{tol}")


def front(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\|\s*(\S+)\s*\|\s*([0-9.]+)\s*\|\s*([^|]*)\|", line)
        if m and m.group(1) not in ("system", "---"):
            out.setdefault(m.group(1), (float(m.group(2)), m.group(3).strip()))
    return out


def fmt(v, n=3):
    return "-" if v is None else f"{v:.{n}f}"


def sgn(v, n=3):
    return "-" if v is None else f"{v:+.{n}f}"


def main():
    data = {r: load(r) for r in RUNS}
    fr = front(FRONT_MD)
    out, facts = [], {"runs": {}}
    out.append("# Result of the 6000-step run next to its controls (cold VAL, 1000 frames of 82 scenes, `seg_fov`, dt 0; all numbers from the stored eval files)\n")
    out.append("| run | steps x accum | start from | step in file | weights | GT signature |")
    out.append("|---|---|---|---|---|---|")
    for r in RUNS:
        x = data[r]
        if x is None:
            out.append(f"| {r} | (no eval file yet) | | | | |")
            continue
        c = x["cfg"]
        out.append(f"| {r} | {c.get('steps', '?')} x {c.get('accum', 1)} | {c.get('init_from') or 'scratch'} | {x['d'].get('info/step', '?')} | {x['weights']} | {x['sig']} |")
    sig_ctl = {data[r]["sig"] for r in CONTROLS if data[r]}
    if data[VARIANT] and sig_ctl and {data[VARIANT]["sig"]} != sig_ctl:
        out.append(f"\n**WARNING: GT signature of the variant {data[VARIANT]['sig']} differs from the controls {sorted(sig_ctl)}: not comparable.**")
    out.append("\n## Maps (IoU; drivable, walkway, carpark, crossing at tolerance 0, dividers at 0.5 m)\n")
    out.append("| run | " + " | ".join(n for n, _, _ in LAYERS) + " |")
    out.append("|---|" + "---|" * len(LAYERS))
    maps = {}
    for r in RUNS:
        x = data[r]
        if x is None:
            continue
        maps[r] = [seg(x, layer, tol) for _, layer, tol in LAYERS]
        out.append(f"| {r} | " + " | ".join(fmt(v) for v in maps[r]) + " |")
        facts["runs"][r] = {"maps": dict(zip([n for n, _, _ in LAYERS], maps[r]))}
    ctl = [maps[r] for r in CONTROLS if r in maps]
    ctl_mean = [statistics.mean(c[i] for c in ctl) for i in range(len(LAYERS))] if ctl else None
    if ctl_mean:
        out.append("| control mean (s0, s1) | " + " | ".join(fmt(v) for v in ctl_mean) + " |")
        facts["control_mean"] = dict(zip([n for n, _, _ in LAYERS], ctl_mean))
    gain = None
    if VARIANT in maps and ctl_mean:
        gain = [a - b for a, b in zip(maps[VARIANT], ctl_mean)]
        out.append("| **long minus control mean** | " + " | ".join(sgn(v) for v in gain) + " |")
        rel = [100 * (a / b - 1) for a, b in zip(maps[VARIANT], ctl_mean)]
        out.append("| relative to the control mean | " + " | ".join(f"{v:+.1f} %" for v in rel) + " |")
        facts["gain"] = dict(zip([n for n, _, _ in LAYERS], gain))
        facts["gain_relative_percent"] = dict(zip([n for n, _, _ in LAYERS], rel))
    out.append("\n## Velocity and detection (vehicles; the hurdle of the card: skill on moving cars >= 0.50 and static error <= 0.30 m/s)\n")
    out.append("| run | " + " | ".join(n for n, _ in VEL) + " | front mAP2 |")
    out.append("|---|" + "---|" * (len(VEL) + 1))
    for r in RUNS:
        x = data[r]
        if x is None:
            continue
        vals = [x["d"].get(k) for _, k in VEL]
        facts["runs"][r]["velocity"] = dict(zip([n for n, _ in VEL], vals))
        f2 = fr.get(r)
        facts["runs"][r]["front_mAP2"] = f2[0] if f2 else None
        out.append(f"| {r} | " + " | ".join(fmt(v) for v in vals) + f" | {fmt(f2[0]) if f2 else '-'} |")
    if fr:
        out.append("\nFront-sector comparison (`bev_suite.py front`, paired scene bootstrap) from `" + FRONT_MD + "`: " + "; ".join(f"{k} {v[0]:.3f} ({v[1]})" for k, v in fr.items()))
    out.append("\n## The rule of the card, evaluated by this script (the verdict itself comes from `decide.py`)\n")
    lines = []
    if gain is not None:
        g = gain[[n for n, _, _ in LAYERS].index("lane divider @0.5 m")]
        walk = gain[1]
        cross = gain[3]
        lines.append(f"- lane divider gain over the control mean: {g:+.4f} (band {BAND}); >= {GAIN_CONTINUE:+.2f} = continue provisionally, below it INCONCLUSIVE; one seed never KEEPs: **{'continue provisionally' if g >= GAIN_CONTINUE else 'INCONCLUSIVE'}**")
        lines.append(f"- 'SOA reachable on this hardware' is dropped only if the equal-budget arm WITHOUT object_moving also gains < {GAIN_DROP:+.2f}; this run has the hurdle on, so it cannot drop it alone (gain {g:+.4f})")
        lines.append(f"- guard: walkway ({walk:+.4f}) and crossing ({cross:+.4f}) have the same sign as the lane divider gain: **{'yes' if (walk > 0) == (g > 0) and (cross > 0) == (g > 0) else 'NO'}**")
    x = data[VARIANT]
    if x:
        sk = x["d"].get("det/dt+0.00/vehicle/all/ave_skill_moving")
        st = x["d"].get("det/dt+0.00/vehicle/all/ave_static")
        if sk is not None and st is not None:
            lines.append(f"- hurdle (skill >= {SKILL_MIN}, static <= {STATIC_MAX}): skill {sk:.3f}, static {st:.3f}: **{'holds' if sk >= SKILL_MIN and st <= STATIC_MAX else 'does NOT hold'}**")
        f2 = fr.get(VARIANT)
        if f2:
            lines.append(f"- guard front mAP2 >= {FRONT_MIN}: {f2[0]:.3f} ({f2[1]}): **{'ok' if f2[0] >= FRONT_MIN else 'VIOLATED'}**")
        else:
            lines.append(f"- guard front mAP2 >= {FRONT_MIN}: not yet measured (`bev_suite.py front ...` of temp/t10/after_long.sh)")
        # predictions
        meas = {"lane divider": seg(x, "lane_divider", "iou@0.5"), "walkway": seg(x, "walkway", "iou@0"), "crossing": seg(x, "ped_crossing", "iou@0"),
                "drivable": seg(x, "drivable_area", "iou@0"), "front mAP2": fr.get(VARIANT, (None,))[0], "vehicle skill": x["d"].get("det/dt+0.00/vehicle/all/ave_skill_moving"),
                "static error": x["d"].get("det/dt+0.00/vehicle/all/ave_static")}
        lines.append("\nPrediction of the run record against the result:\n")
        lines.append("| quantity | predicted (range) | measured | in range |")
        lines.append("|---|---|---|---|")
        hits = 0
        facts["prediction"] = {}
        for k, (v, lo, hi) in PRED.items():
            mv = meas.get(k)
            ok = None if mv is None else (lo <= mv <= hi)
            hits += 1 if ok else 0
            where = "-" if ok is None else ("yes" if ok else ("below" if mv < lo else "above"))
            lines.append(f"| {k} | {v} ({lo} - {hi}) | {fmt(mv)} | {where} |")
            facts["prediction"][k] = {"pred": v, "low": lo, "high": hi, "measured": mv, "in_range": ok}
        lines.append(f"\n{hits} of {len(PRED)} predicted quantities inside their range (front mAP2 counts only after the front comparison).")
    out.extend(lines)
    text = "\n".join(out) + "\n"
    open("temp/t10/results_long.md", "w", encoding="utf-8").write(text)
    json.dump(facts, open("temp/t10/facts_long.json", "w"), indent=1)
    print(text)


if __name__ == "__main__":
    main()
