import json, re
R = "/mnt/c/Auto-research/bev_fusion/runs"
runs = ["t10_dual_s0", "t10_dual_s1", "t11_vel_hurdle", "t11_vel_hurdle_acc2"]
pat = re.compile(r"(det/dt\+0\.00/(vehicle|pedestrian)/all/(ave|ave_skill_moving|ave_moving|ave0_moving|ave_static|mAP2|AP2)$)|(^maps/.*(iou|IoU|miou))|(front)", re.I)
mets = {r: json.load(open(f"{R}/{r}/eval/metrics.json")) for r in runs}
def flat(d, p=""):
    out = {}
    for k, v in d.items():
        kk = f"{p}/{k}" if p else k
        if isinstance(v, dict): out.update(flat(v, kk))
        elif isinstance(v, (int, float)): out[kk] = v
    return out
F = {r: flat(m) for r, m in mets.items()}
keys = sorted(k for k in F["t11_vel_hurdle"] if pat.search(k))
print("key".ljust(60), *[r[-12:].rjust(12) for r in runs])
for k in keys:
    print(k[:60].ljust(60), *[(f"{F[r][k]:.4f}" if k in F[r] else "-").rjust(12) for r in runs])
