"""t11 vel cause, input side: how much of a vehicle's velocity do the radar returns inside its GT box carry?

The question behind the literature comparison (CenterFusion / RCBEVDet / CRN use the radar Doppler for velocity; BEVDet4D, VideoBEV,
StreamPETR get it from aligned history): before porting anything, does the INPUT of our chain hold the velocity? For the observed GT vehicles
of the front sector in n val frames (the dataset's own radar field, exactly what the radar branch receives: ego-compensated radial velocity
`vr` along the ray `psi` of the vehicle frame, positions shifted along the ray to the camera time, returns of the last 0.25 s), and with
the association by the GT box (footprint + 1 m), so these are UPPER bounds of what a detector-side association could do:
  radial check    corr / median error of vr against v_gt . u over the returns inside moving boxes (the sign convention and the compensation)
  coverage        share of moving / static vehicles with >= 1 return
  Doppler + heading prior   v_est = (vr / (h . u)) h with h = the box axis (cos yaw, sin yaw), median over the returns with |h . u| >= 0.3,
                  0 where no return qualifies: AVE of v_est against v_gt on the moving vehicles, ave0 = mean |v_gt|, skill = 1 - AVE / ave0
  moving / static the median |vr| > 0.5 m/s as a moving test: true / false positive rate
Run (from bev_fusion/, CPU, one light devkit ~1.5 GB):
  OMP_NUM_THREADS=2 ~/miniconda3/envs/gcar/bin/python temp/box_vel/doppler_at_boxes.py runs/t8_ctl_s0 val 300 30 temp/box_vel/out/doppler_at_boxes_val.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [ROOT, HERE]

MOVING = 0.5          # m/s, evaluation/metrics.py MOVING_SPEED
MARGIN = 1.0          # m around the footprint
MAX_AGE = 0.25        # s, radar sweeps older than this are not used
MIN_COS = 0.3         # |h . u| below this: the motion is nearly across the line of sight, no Doppler estimate
BINS = ((0.0, 20.0), (20.0, 40.0), (40.0, 50.0))


def frame_rows(item, in_front, nusc):
    rad = {k: np.asarray(v) for k, v in item["radar"].items()}
    R_g2e = np.asarray(item["gt"]["cur"]["meta"]["T_e2g"], float)[:2, :2].T
    rows = []
    for b in item["gt"]["cur"]["meta"]["boxes"]:
        if b["group"] != "vehicle" or not b.get("observed", True) or not b.get("vel_known", True):
            continue
        cx, cy = float(b["center"][0]), float(b["center"][1])
        if not in_front({"cx": cx, "cy": cy}):
            continue
        w, ln, yaw = float(b["size"][0]), float(b["size"][1]), float(b["yaw"])
        vg = np.asarray(b["vel"], float)[:2]
        dx, dy = rad["x"] - cx, rad["y"] - cy
        c, s = math.cos(yaw), math.sin(yaw)
        u, v = dx * c + dy * s, -dx * s + dy * c
        k = (np.abs(u) <= ln / 2 + MARGIN) & (np.abs(v) <= w / 2 + MARGIN) & (rad["dt"] <= MAX_AGE)
        try:                                                  # the devkit velocity: differences of the annotated positions, no radar
            vdk = R_g2e @ np.asarray(nusc.box_velocity(b["token"]), float)[:2]
            vdk = vdk.tolist() if np.isfinite(vdk).all() else None
        except Exception:                                      # noqa: BLE001
            vdk = None
        row = {"x": cx, "speed": float(np.hypot(*vg)), "vg": vg.tolist(), "vdk": vdk, "src": b.get("vel_source", "?"), "n": int(k.sum()),
               "stationary": bool(b.get("stationary", False))}
        if k.any():
            ux, uy = np.cos(rad["psi"][k]), np.sin(rad["psi"][k])
            vr = rad["vr"][k].astype(float)
            row["vr"], row["u"] = vr.tolist(), np.stack([ux, uy], 1).tolist()
            cosv = c * ux + s * uy
            use = np.abs(cosv) >= MIN_COS
            if use.any():
                sest = float(np.median(vr[use] / cosv[use]))
                row["v_est"] = [sest * c, sest * s]
        rows.append(row)
    return rows


def with_ref(rows, ref):
    """the rows that have the reference velocity `ref` ('vg' = autogt's, partly fused with the radar Doppler; 'vdk' = the devkit's, from the
    annotated positions only), with speed and the reference radial velocity recomputed against it"""
    out = []
    for r in rows:
        v = r.get(ref)
        if v is None:
            continue
        r2 = dict(r, vg=v, speed=float(np.hypot(*v)))
        if r["n"]:
            r2["vr_gt"] = (np.asarray(r["u"]) @ np.asarray(v)).tolist()
        out.append(r2)
    return out


def summarise(rows):
    mv = [r for r in rows if r["speed"] >= MOVING]
    st = [r for r in rows if r["speed"] < MOVING]
    out = {"n_vehicles": len(rows), "n_moving": len(mv), "n_static": len(st)}

    def ave(rs):
        e = [np.hypot(r["v_est"][0] - r["vg"][0], r["v_est"][1] - r["vg"][1]) if "v_est" in r else r["speed"] for r in rs]
        return float(np.mean(e)), float(np.mean([r["speed"] for r in rs]))
    if mv:
        out["coverage_moving"] = float(np.mean([r["n"] > 0 for r in mv]))
        out["estimate_moving"] = float(np.mean(["v_est" in r for r in mv]))
        if any(r["n"] for r in mv):
            a = np.concatenate([r["vr"] for r in mv if r["n"]])
            g = np.concatenate([r["vr_gt"] for r in mv if r["n"]])
            out["radial_corr_moving"] = float(np.corrcoef(a, g)[0, 1])
            out["radial_median_abs_err_moving"] = float(np.median(np.abs(a - g)))
            out["radial_median_abs_gt_moving"] = float(np.median(np.abs(g)))
            med = lambda r: float(np.median(np.abs(r["vr"])))                                # noqa: E731
            out["moving_test_tpr"] = float(np.mean([med(r) > MOVING for r in mv if r["n"]]))
        e, e0 = ave(mv)
        out.update(ave_moving=e, ave0_moving=e0, skill_moving=1 - e / e0)
        sub = [r for r in mv if "v_est" in r]
        if sub:
            e, e0 = ave(sub)
            out.update(ave_moving_with_estimate=e, ave0_moving_with_estimate=e0, skill_moving_with_estimate=1 - e / e0, n_with_estimate=len(sub))
    if st:
        out["coverage_static"] = float(np.mean([r["n"] > 0 for r in st]))
        if any(r["n"] for r in st):
            a0 = np.concatenate([r["vr"] for r in st if r["n"]])
            out["radial_median_abs_vr_static"] = float(np.median(np.abs(a0)))
            out["radial_p90_abs_vr_static"] = float(np.quantile(np.abs(a0), 0.9))
            out["moving_test_fpr"] = float(np.mean([float(np.median(np.abs(r["vr"]))) > MOVING for r in st if r["n"]]))
        sub = [r for r in st if "v_est" in r]
        out["static_false_speed_mean"] = float(np.mean([np.hypot(*r["v_est"]) for r in sub])) if sub else float("nan")
        out["static_false_speed_ge1"] = float(np.mean([np.hypot(*r["v_est"]) >= 1.0 for r in sub])) if sub else float("nan")
    return out


def reference_floor(rows):
    """how far autogt's velocity is from the devkit's on the vehicles both have, moving by the devkit: the noise of the independent reference"""
    both = [r for r in rows if r.get("vdk") is not None and np.hypot(*r["vdk"]) >= MOVING]
    if not both:
        return {}
    e = float(np.mean([np.hypot(r["vg"][0] - r["vdk"][0], r["vg"][1] - r["vdk"][1]) for r in both]))
    e0 = float(np.mean([np.hypot(*r["vdk"]) for r in both]))
    return {"n": len(both), "ave_autogt_vs_devkit": e, "ave0_devkit": e0, "skill": 1 - e / e0}


def main(run, split, n, n_scenes, out_path):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    from evaluation.bev_adapter import _chain, setup_paths
    setup_paths(run)
    ns = _chain()
    from evaluation.metrics import in_front
    from probe_features import datasets
    ds, scene_names = datasets(run, ns.T, split, n_scenes)
    idx = sorted(set(np.linspace(0, len(ds) - 1, min(n, len(ds))).round().astype(int).tolist()))
    print(f"{run} {split}: {len(scene_names)} scenes, {len(ds)} samples, {len(idx)} frames", flush=True)
    rows = []
    for j, i in enumerate(idx):
        rows += frame_rows(ds[i], in_front, ds.nusc)
        if j % 50 == 0:
            print(f"{j + 1}/{len(idx)} frames, {len(rows)} vehicles", flush=True)
    res = {"run": run, "split": split, "frames": len(idx), "scenes": len(scene_names),
           "sources": {s: sum(1 for r in rows if r["src"] == s) for s in sorted({r["src"] for r in rows})}, "reference_floor": reference_floor(rows)}
    for ref, label in (("vg", "autogt"), ("vdk", "devkit")):
        rr = with_ref(rows, ref)
        res[f"{label}_all"] = summarise(rr)
        for lo, hi in BINS:
            res[f"{label}_x_{lo:.0f}_{hi:.0f}"] = summarise([r for r in rr if lo <= r["x"] < hi])
        if ref == "vg":
            for s in res["sources"]:
                res[f"autogt_src_{s}"] = summarise([r for r in rr if r["src"] == s])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=1)
    print("sources", res["sources"], "reference floor", json.dumps({a: round(b, 3) for a, b in res["reference_floor"].items()}))
    for k in res:
        if k.startswith(("autogt_", "devkit_")):
            print(k, json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in res[k].items()}))
    print(f"-> {out_path}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]), int(a[3]), a[4])
