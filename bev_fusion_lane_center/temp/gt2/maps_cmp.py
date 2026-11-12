import json, os, sys
def L(r):
    p = f"../../runs/{r}/jmaps_val.json"
    return json.load(open(p)) if os.path.exists(p) and os.path.getsize(p) > 0 else None
def kp(d):
    k = d["keypoints"]
    return (k["kp_ours_tol3.0_noang_ent"]["f1"] + k["kp_ours_tol3.0_noang_ext"]["f1"]) / 2
def rng(d, key):
    return d["range_city_split"][key]["recall"]
arm, ctl = sys.argv[1], sys.argv[2]            # e.g. e307_n e286_j1
for s in (0, 1):
    a, c = L(f"{arm}_s{s}"), L(f"{ctl}_s{s}")
    if not a or not c:
        print(f"s{s}: pending")
        continue
    print(f"s{s}: kp_f1 {kp(a):.3f} vs {kp(c):.3f} ({kp(a) - kp(c):+.3f}); zone IoU {a['zone_iou']:.3f} vs {c['zone_iou']:.3f} ({a['zone_iou'] - c['zone_iou']:+.3f}); core IoU {a['core_iou']:.3f} vs {c['core_iou']:.3f}; "
          f"entry recall crowded {a['crowded_split']['ours_ent']['crowded_recall']:.3f} vs {c['crowded_split']['ours_ent']['crowded_recall']:.3f} ({a['crowded_split']['ours_ent']['crowded_recall'] - c['crowded_split']['ours_ent']['crowded_recall']:+.3f}), isolated {a['crowded_split']['ours_ent']['isolated_recall']:.3f} vs {c['crowded_split']['ours_ent']['isolated_recall']:.3f}; "
          f"entry >35 m {rng(a, 'range_ours_ent_gt35'):.3f} vs {rng(c, 'range_ours_ent_gt35'):.3f} ({rng(a, 'range_ours_ent_gt35') - rng(c, 'range_ours_ent_gt35'):+.3f}); best-thr F1 {a['kp_f1_best_threshold']['ours']:.3f} vs {c['kp_f1_best_threshold']['ours']:.3f}")
