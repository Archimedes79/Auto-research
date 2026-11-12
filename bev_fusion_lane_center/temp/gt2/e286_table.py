import json
arms = ("j0", "j1", "j2")
D = {(a, s): json.load(open(f"runs/e286_{a}_s{s}/jmaps_val.json")) for a in arms for s in (0, 1)}


def kpf1(d, who="ours"):
    kp = d["keypoints"]
    return (kp[f"kp_{who}_tol3.0_noang_ent"]["f1"] + kp[f"kp_{who}_tol3.0_noang_ext"]["f1"]) / 2


def rc(d, key):
    return d["range_city_split"][key]["recall"]


def cr(d, key):
    return d["crowded_split"][key]


def pair(f, who_arm=None):
    return " / ".join("%.3f" % f(D[(who_arm, s)]) for s in (0, 1))


out = []
out.append("| arm | kp_f1 (thr 0.3, 3 m, distance only) s0 / s1 | mean | best-threshold F1 s0 / s1 | zone IoU s0 / s1 | core IoU s0 / s1 | entry recall > 35 m s0 / s1 | entry recall Singapore s0 / s1 | entry recall crowded / isolated (s0) |")
out.append("|---|---|---|---|---|---|---|---|---|")
bs = [kpf1(D[("j1", s)], "B") for s in (0, 1)]
out.append("| **c: B's event heat** | %.3f / %.3f | **%.3f** | %.3f / %.3f | - | - | %.3f / %.3f | %s | %.3f / %.3f |" % (bs[0], bs[1], sum(bs) / 2, D[("j1", 0)]["kp_f1_best_threshold"]["B"], D[("j1", 1)]["kp_f1_best_threshold"]["B"],
           rc(D[("j1", 0)], "range_B_ent_gt35"), rc(D[("j1", 1)], "range_B_ent_gt35"), " / ".join("%.3f" % rc(D[("j1", s)], "city_B_ent_singapore") for s in (0, 1)),
           cr(D[("j1", 0)], "B_ent")["crowded_recall"], cr(D[("j1", 0)], "B_ent")["isolated_recall"]))
for a in arms:
    k = [kpf1(D[(a, s)]) for s in (0, 1)]
    out.append("| **%s** | %.3f / %.3f | **%.3f** | %s | %s | %s | %s | %s | %.3f / %.3f |" % (
        a, k[0], k[1], sum(k) / 2, pair(lambda d: d["kp_f1_best_threshold"]["ours"], a), pair(lambda d: d["zone_iou"], a), pair(lambda d: d["core_iou"], a),
        pair(lambda d: rc(d, "range_ours_ent_gt35"), a), pair(lambda d: rc(d, "city_ours_ent_singapore"), a), cr(D[(a, 0)], "ours_ent")["crowded_recall"], cr(D[(a, 0)], "ours_ent")["isolated_recall"]))
print("\n".join(out))
gain = {a: [kpf1(D[(a, s)]) - bs[s] for s in (0, 1)] for a in arms}
print({a: [round(g, 3) for g in v] for a, v in gain.items()})
