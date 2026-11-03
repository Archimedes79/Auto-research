"""throw-away: apply the second batch of edits to soa_maps_v8.py (exact-text replacements, each must match once)"""
import re
p = "temp/t16/soa_maps_v8.py"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)


rep('ap.add_argument("--rule", choices=("old", "usable"), default="old")',
    'ap.add_argument("--rule", choices=("old", "usable", "old400"), default="old")')
rep('''    if a.rule == "old":
        base = [r for r in old400 if r[1] in pos]''',
    '''    if a.rule == "old400":                                      # all 400 old frames, edge keyframes included: only without the network (the val dataset has no sample for them)
        if not a.no_model:
            sys.exit("--rule old400 needs --no-model (the dataset of the run has no sample for the first / last keyframe of a scene)")
        base = list(old400)
        rule_text = "all 400 frames of the old scoring (the first / last keyframe of a scene included)"
    elif a.rule == "old":
        base = [r for r in old400 if r[1] in pos]''')
rep('old = old_comparison(S) if (len(frames_info) == 400 and same_usable) else None',
    'old = old_comparison(S) if a.rule == "old400" and len(frames_info) == 400 else None')
# aggregate of the stored-count gate
rep('''                    emax = max(emax, abs(acc["err"] - sc[sk][2]))
''', '''                    emax = max(emax, abs(acc["err"] - sc[sk][2]))
                    AGG["mine"][sk] = AGG["mine"].get(sk, 0) + acc["cnt"]
                    AGG["theirs"][sk] = AGG["theirs"].get(sk, 0) + sc[sk][0]
''')
rep('LINES = ("lane_divider", "road_divider")', 'LINES = ("lane_divider", "road_divider")\nAGG = {"mine": {}, "theirs": {}}                      # summed counts of the frames that are in the stored evaluation: ours (CPU) and the stored ones (GPU)')
rep('''                "nan_pattern_equal": all(c["nan_pattern_equal"] for c in checks), "frames": checks}''',
    '''                "nan_pattern_equal": all(c["nan_pattern_equal"] for c in checks),
                "keys_compared": sum(c["keys_compared"] for c in checks), "keys_differing": sum(c["keys_differing"] for c in checks),
                "headline_from_summed_counts": agg_headline(), "frames": checks}''')
rep('def best_of(S, s, layer, b, tol):', '''def agg_headline():
    """headline numbers of ours from the summed counts of the checked frames: CPU (this script) / the stored GPU counts of the suite / difference"""
    def val(c, t, zc):
        ph, pm, gh, gm = c[t]
        if zc:
            p, r = ph / (ph + pm), gh / (gh + gm)
            return 2 * p * r / (p + r)
        tp = (ph + gh) / 2
        return tp / (tp + pm + gm)
    out = {}
    for layer, t, zc, name in (("drivable_area", 0, False, "drivable_area iou@0"), ("ped_crossing", 0, False, "ped_crossing iou@0"), ("walkway", 0, False, "walkway iou@0"),
                               ("carpark_area", 0, False, "carpark_area iou@0"), ("lane_divider", 1, False, "lane_divider iou@0.5"),
                               ("road_divider", 2, False, "road_divider iou@1"), ("lane_divider:zc", 1, True, "lane_divider:zc f1@0.5"), ("road_divider:zc", 1, True, "road_divider:zc f1@0.5")):
        k = f"fov|+0.00|{layer}|all|all"
        if k in AGG["mine"]:
            try:
                m_, s_ = float(val(AGG["mine"][k], t, zc)), float(val(AGG["theirs"][k], t, zc))
            except ZeroDivisionError:
                continue
            out[name] = {"cpu": m_, "stored_gpu": s_, "diff": m_ - s_}
    return out


def best_of(S, s, layer, b, tol):''')
# the tables: a cell set without GT cells is not a number
rep('''                n_gt = S["gcar"][layer][bin_]["0.5"]["n_gt"]
                if best:''', '''                n_gt = S["gcar"][layer][bin_]["0.5"]["n_gt"]
                if n_gt == 0:
                    w(f"| {layer} | {tol} | 0 | - | - | - | - |")
                    continue
                if best:''')
rep('''                cells.append(f"{fmt(ou)} / {fmt(gc)} / {fmt(ratio(ou, gc), 2)} ({S['gcar'][layer][b]['0.5']['n_gt']})")''',
    '''                n_b = S["gcar"][layer][b]["0.5"]["n_gt"]
                cells.append(f"{fmt(ou)} / {fmt(gc)} / {fmt(ratio(ou, gc), 2)} ({n_b})" if n_b else "- (no GT cells)")''')
rep('''        w(f"Sanity: {g['n_frames']} of these frames are in the stored suite evaluation `{g['store']}`; the per-frame counts of ours equal the stored ones "
          f"(max difference {g['max_count_diff']:.3g}, scores {g['max_score_diff']:.3g}, NaN pattern equal: {g['nan_pattern_equal']}).")''',
    '''        w(f"Sanity (ours against the suite's stored numbers): {g['n_frames']} of these frames are in the stored suite evaluation `{g['store']}` (made on the GPU). "
          f"{g['keys_compared'] - g['keys_differing']} of {g['keys_compared']} per-frame count arrays are identical; the others differ by at most {g['max_count_diff']:.3g} cells "
          f"(a cell whose probability lies on the 0.5 threshold, CPU against GPU floats); largest per-frame score difference {g['max_score_diff']:.2g}, NaN pattern equal: {g['nan_pattern_equal']}. "
          "From the summed counts (CPU / stored / difference): "
          + "; ".join(f"{k} {v['cpu']:.4f} / {v['stored_gpu']:.4f} / {v['diff']:+.5f}" for k, v in g["headline_from_summed_counts"].items()) + ".")''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
