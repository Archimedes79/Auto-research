"""l4_readout.py - scratch (no verdict): the contribution of the lane branch (L4) of t20_base_s0 to the lane_divider map, on the 246 frames, CPU, with the WORKING TREE's training/ code for all
three variants (lane_readout.py run t20_base_s0 --variant live | l4off | nodepth must have been run):
  (a) live     as it is (L4 on, lift PD)                (b) l4off    model.lane.mix weight and bias zeroed after the load: the contribution is exactly 0
  (c) nodepth  ModelConfig.lane_use_depth False (lift P), the t20 weights loaded unchanged
Paired scene-bootstrap differences (a) - (b) and (c) - (a) (lane_readout compare), amplitude of the branch's output, and the check that (a) with the live code equals the stored-snapshot read-out.
-> temp/overview/l4_readout.txt / .json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lane_readout as LR  # noqa: E402

RUN = "t20_base_s0"


def main():
    cp = LR.resolve_ckpt(RUN, None)
    tags = {v: LR.tag_of(RUN, cp, v) for v in ("live", "l4off", "nodepth")}
    snap = LR.tag_of(RUN, cp)
    rd = {}
    for v, t in tags.items():
        if not os.path.exists(os.path.join(HERE, f"readout_{t}.json")):
            raise SystemExit(f"missing read-out of {t}: run  lane_readout.py run {RUN} --variant {v}")
        rd[v] = json.load(open(os.path.join(HERE, f"readout_{t}.json")))
    LR.do_compare(RUN, RUN, None, None, "l4off", "live")          # B - A = live - l4off  = (a) - (b)
    LR.do_compare(RUN, RUN, None, None, "live", "nodepth")        # nodepth - live = (c) - (a)
    cab = json.load(open(os.path.join(HERE, f"readout_compare_{tags['l4off']}_vs_{tags['live']}.json")))
    cca = json.load(open(os.path.join(HERE, f"readout_compare_{tags['live']}_vs_{tags['nodepth']}.json")))
    L, R = [], {"tags": tags, "n_frames": rd["live"]["n_frames"]}
    L.append(f"==== L4 read-out of {RUN} ({os.path.basename(cp)}), {rd['live']['n_frames']} val frames, CPU, memory empty, EMA weights, working-tree training/ code for all three variants ====")
    # reproduction of the earlier numbers with the live code
    sn = os.path.join(LR.MAPS_DIR, snap)
    lv = os.path.join(LR.MAPS_DIR, tags["live"])
    mx, n = 0.0, 0
    for f in os.listdir(lv):
        if f.endswith(".npz") and os.path.exists(os.path.join(sn, f)):
            a_, b_ = np.load(os.path.join(lv, f))["lane_divider"], np.load(os.path.join(sn, f))["lane_divider"]
            mx, n = max(mx, float(np.abs(a_ - b_).max())), n + 1
    L.append(f"(a) with the live code vs the stored-snapshot code (runs/{RUN}/code): lane_divider probability maps of {n} frames, max |difference| = {mx:.2e}")
    snap_json = json.load(open(os.path.join(HERE, f"readout_{snap}.json")))
    L.append("    IoU @0 / 0.5 / 1 m, all cells: live " + " / ".join(f"{rd['live']['iou']['cnt_base'][t]['iou']:.4f}" for t in ("0.0", "0.5", "1.0"))
             + "   snapshot code " + " / ".join(f"{snap_json['iou']['cnt_base'][t]['iou']:.4f}" for t in ("0.0", "0.5", "1.0")) + "   (earlier: 0.1970 / 0.4462 / 0.5613; outside intersections 0.2204 / 0.5091 / 0.6409)")
    R["reproduction"] = dict(max_map_diff=mx, n_frames=n)
    # table of the three variants
    names = {"live": "(a) L4 on, lift PD (as is)", "l4off": "(b) L4 dropped (mix zeroed)", "nodepth": "(c) lift P (lane_use_depth False)"}
    L += ["", "-- lane_divider IoU, precision / recall at 0.5 m; 95% CI over scenes in brackets",
          f"{'variant':36s} {'cells':10s} | {'IoU@0':>7s} {'IoU@0.5':>8s} {'[95% CI]':>15s} {'IoU@1':>7s} | {'prec@0.5':>8s} {'rec@0.5':>8s}"]
    for v in ("live", "l4off", "nodepth"):
        for key, cn in (("cnt_base", "all"), ("cnt_both", "outside"), ("cnt_band", "outside+band")):
            d = rd[v]["iou"][key]
            L.append(f"{names[v]:36s} {cn:10s} | {d['0.0']['iou']:7.4f} {d['0.5']['iou']:8.4f} {'[%.3f, %.3f]' % tuple(d['0.5']['ci']):>15s} {d['1.0']['iou']:7.4f} | {d['0.5']['precision']:8.3f} {d['0.5']['recall']:8.3f}")
    L += ["", "-- ranking of the probability on the fov cells (the real metric's histograms): AP / ROC-AUC",
          *[f"{names[v]:36s} AP {rd[v]['curves']['ap']:.4f} [{rd[v]['curves']['ap_ci'][0]:.3f}, {rd[v]['curves']['ap_ci'][1]:.3f}]   ROC-AUC {rd[v]['curves']['roc_auc']:.4f}" for v in ("live", "l4off", "nodepth")]]
    # paired differences
    def rows(c, label, sign_note):
        L.extend(["", f"-- paired differences {label} (scene bootstrap, 2000 resamples; * = the interval excludes 0)"])
        for k in ("IoU@0.0 all cells", "IoU@0.5 all cells", "IoU@1.0 all cells", "IoU@0.0 outside intersections", "IoU@0.5 outside intersections", "IoU@1.0 outside intersections", "IoU@0.5 outside + band",
                  "precision@0.5 all cells", "recall@0.5 all cells", "precision@0.5 outside intersections", "recall@0.5 outside intersections", "AP", "ROC-AUC"):
            r = c["rows"].get(k)
            if r is None:
                continue
            L.append(f"   {k:40s} {r['diff']:+8.4f}  [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]" + ("  *" if r["ci"][0] > 0 or r["ci"][1] < 0 else "") + f"      ({r['B']:.4f} vs {r['A']:.4f})")
        for k in sorted(c["rows"]):
            if k.startswith("IoU@0.5 all cells p>") and k.endswith(("0.30", "0.70", "0.90")):
                r = c["rows"][k]
                L.append(f"   {k:40s} {r['diff']:+8.4f}  [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
    rows(cab, "(a) L4 on minus (b) L4 dropped", "")
    rows(cca, "(c) lift P minus (a) lift PD", "")
    R["a_minus_b"], R["c_minus_a"] = cab["rows"], cca["rows"]
    # amplitude
    L += ["", "-- amplitude of the lane branch's output (hook on model.lane) against the camera BEV it is added to (hook on the camera temporal block); means over the frames"]
    R["amplitude"] = {}
    for v in ("live", "nodepth", "l4off"):
        a = rd[v].get("l4")
        if a is None:
            L.append(f"   {names[v]:36s} no hook values")
            continue
        L.append(f"   {names[v]:36s} mean |lane| {a['mean_abs_lane']:.5f}  std lane {a['std_lane']:.5f} | mean |cam| {a['mean_abs_cam']:.5f}  std cam {a['std_cam']:.5f} | std ratio lane/cam {a['std_ratio']:.4f}  mean-abs ratio {a['absmean_ratio']:.4f} | "
                 f"fov cells: RMS lane {a['rms_lane_fov']:.4f} vs cam {a['rms_cam_fov']:.4f}, per-cell ratio {a['cell_ratio_fov']:.4f}")
        R["amplitude"][v] = a
    L += ["", "(the camera BEV is the same tensor in (a) and (c) up to the added lane output; (b) adds exactly 0 by construction)"]
    txt = "\n".join(L) + "\n"
    open(os.path.join(HERE, "l4_readout.txt"), "w").write(txt)
    json.dump(R, open(os.path.join(HERE, "l4_readout.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(txt)


if __name__ == "__main__":
    main()
