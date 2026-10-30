p = "evaluation/evaluate_run.py"
s = open(p, encoding="utf-8").read()
old = '''    for cls in ("vehicle", "pedestrian"):
        # the greedy matching of a detection only involves its own frame, so the true-positive flags of every scene are fixed:
        # compute them once, a resample only re-sorts the concatenated flags (equal to re-running det_ap, but fast)
        fl_a, fl_b = scene_flags(fa, sc, cls), scene_flags(fb, sc, cls)
        a0, b0 = ap_from_flags(fl_a, sc), ap_from_flags(fl_b, sc)
        ds = []
        for _ in range(n_boot):
            ix = rng.integers(0, len(sc), len(sc))
            s_ix = [sc[i] for i in ix]
            ds.append(ap_from_flags(fl_b, s_ix) - ap_from_flags(fl_a, s_ix))
        lo, hi = np.percentile(ds, [2.5, 97.5])
        res[f"{cls} AP@2"] = (b0 - a0, lo, hi, a0, b0)'''
new = '''    for cls in ("vehicle", "pedestrian"):
        for thr in (1.0, 2.0):            # AP@1 added 2026-10-29: AP@2 has a high no-sensor floor (t3_improve_loop try 1)
            # the greedy matching of a detection only involves its own frame, so the true-positive flags of every scene are fixed:
            # compute them once, a resample only re-sorts the concatenated flags (equal to re-running det_ap, but fast)
            fl_a, fl_b = scene_flags(fa, sc, cls, thr), scene_flags(fb, sc, cls, thr)
            a0, b0 = ap_from_flags(fl_a, sc), ap_from_flags(fl_b, sc)
            ds = []
            for _ in range(n_boot):
                ix = rng.integers(0, len(sc), len(sc))
                s_ix = [sc[i] for i in ix]
                ds.append(ap_from_flags(fl_b, s_ix) - ap_from_flags(fl_a, s_ix))
            lo, hi = np.percentile(ds, [2.5, 97.5])
            res[f"{cls} AP@{thr:g}"] = (b0 - a0, lo, hi, a0, b0)'''
assert old in s
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))

p = "temp/study/collect.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''KEYS = [("mAP", "det/dt+0.00/mean/all/mAP", 3), ("veh AP@2", "det/dt+0.00/vehicle/all/ap@2", 3),
        ("ped AP@2", "det/dt+0.00/pedestrian/all/ap@2", 3),''', '''KEYS = [("mAP2", "mAP2", 3), ("veh AP@1", "det/dt+0.00/vehicle/all/ap@1", 3), ("ped AP@1", "det/dt+0.00/pedestrian/all/ap@1", 3),
        ("veh AP@2", "det/dt+0.00/vehicle/all/ap@2", 3), ("ped AP@2", "det/dt+0.00/pedestrian/all/ap@2", 3),
        ("mAP3", "det/dt+0.00/mean/all/mAP", 3),''')
s = s.replace('''BOOT = {"mAP": None, "veh AP@2": "vehicle AP@2", "ped AP@2": "pedestrian AP@2",''', '''BOOT = {"mAP2": None, "veh AP@1": "vehicle AP@1", "ped AP@1": "pedestrian AP@1", "veh AP@2": "vehicle AP@2", "ped AP@2": "pedestrian AP@2",''')
s = s.replace('''def load(arm):
    p = f"runs/t2_{arm}/eval/metrics.json"
    return json.load(open(p))["metrics"] if os.path.exists(p) else None''', '''def load(arm):
    p = f"runs/{arm if arm.startswith('t3_') else 't2_' + arm}/eval/metrics.json"
    if not os.path.exists(p):
        return None
    m = json.load(open(p))["metrics"]
    # mAP2: vehicle + pedestrian only (the third class, two_wheeler, has 174 GT in 1000 frames and swings +-0.15 AP between seeds)
    m["mAP2"] = sum(m[f"det/dt+0.00/{c}/all/ap@{t}"] for c in ("vehicle", "pedestrian") for t in (0.5, 1, 2, 4)) / 8
    return m''')
s = s.replace('ARMS = ["ref_s0", "ref_s1", "radar_off",', 'ARMS = ["ref_s0", "ref_s1", "t3_cam_zsum", "t3_cam_adapter", "t3_adapter", "t3_zsum", "t3_ref_long", "radar_off",')
# band: seed spread of the stable headline metrics (not the two_wheeler-driven mAP3)
s = s.replace('''    band = max(0.010, 2 * max(abs(M["ref_s1"][k] - ref[k]) for _, k, _ in KEYS[:1] + KEYS[5:8]))''', '''    band = max(0.010, 2 * max(abs(M["ref_s1"][k] - ref[k]) for _, k, _ in KEYS[:2] + KEYS[7:10]))''')
s = s.replace('print("\n## Differences and verdicts\n")\nprint("| arm | " + " | ".join(k for k, _, _ in KEYS[:8])', 'print("\n## Differences and verdicts\n")\nprint("| arm | " + " | ".join(k for k, _, _ in KEYS[:10])')
s = s.replace('print("|---|" + "---|" * 9)', 'print("|---|" + "---|" * 11)')
s = s.replace("    for lab, key, d in KEYS[:8]:", "    for lab, key, d in KEYS[:10]:")
s = s.replace('''        res, _ = ev.compare("runs/t2_ref_s0", f"runs/t2_{a}", n_boot=300)''', '''        res, _ = ev.compare("runs/t2_ref_s0", f"runs/{a if a.startswith('t3_') else 't2_' + a}", n_boot=300)''')
s = s.replace('''        elif lab == "mAP" and abs(diff) > band:
            (better if diff > 0 else worse).append(lab + " (no bootstrap)")''', '''        elif lab == "mAP2" and abs(diff) > band:
            (better if diff > 0 else worse).append(lab + " (no bootstrap)")''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
