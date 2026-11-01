import json, sys
def flat(d, p=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict): o.update(flat(v, p + k + "."))
        else: o[p + k] = v
    return o
base = flat(json.load(open("training/configs/t10_dual_s0.json")))
run = flat(json.load(open("runs/t10_dual_s0/config.json")))
for n in ("t11_vel_ctl", "t11_vel_hurdle"):
    c = flat(json.load(open(f"training/configs/{n}.json")))
    print(n, {k: (base.get(k, "<absent>"), c.get(k, "<absent>")) for k in sorted(set(base) | set(c)) if base.get(k, "<absent>") != c.get(k, "<absent>")})
print("run config vs file:", {k: (run.get(k, "<absent>"), base.get(k, "<absent>")) for k in sorted(set(base)) if run.get(k) != base.get(k)})
print("run-only keys (defaults at t10):", {k: run[k] for k in sorted(set(run) - set(base))})
