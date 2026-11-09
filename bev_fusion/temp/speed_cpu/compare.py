import json
import re

import common  # noqa: F401  (chdir to bev_fusion)


def load(tag):
    return json.load(open(f"temp/speed_cpu/profile_{tag}.json"))["warm"]


def norm(rows):
    out = {}
    for k, v in rows.items():
        k2 = re.sub(r"#\d+$", "", k)
        o = out.setdefault(k2, {"fwd": 0, "bwd": 0})
        o["fwd"] += v.get("fwd", 0)
        o["bwd"] += v.get("bwd", 0)
    return out


tags = ["b1", "s_b1n15", "s_b2n15", "s_b1n30"]
D = {t: load(t) for t in tags}
for t in tags:
    print(t, "total fwd+bwd", D[t]["tot"], "=", D[t]["tot"]["fwd"] + D[t]["tot"]["bwd"], "time with profiler %.1f s" % D[t]["time_s"])
N = {t: norm(D[t]["rows"]) for t in tags}
secs = sorted(N["b1"], key=lambda k: -(N["b1"][k]["fwd"] + N["b1"][k]["bwd"]))
print(f"\n{'section (warm keyframe)':34s} {'B1 full':>9s} {'B1 small':>9s} {'B2 small':>9s} {'B2-B1':>7s} {'B1 n30':>8s} {'n30-n15':>8s}")
for s in secs:
    g = lambda t: N[t].get(s, {"fwd": 0, "bwd": 0})
    tot = lambda t: g(t)["fwd"] + g(t)["bwd"]
    print(f"{s:34s} {tot('b1'):9d} {tot('s_b1n15'):9d} {tot('s_b2n15'):9d} {tot('s_b2n15') - tot('s_b1n15'):7d} {tot('s_b1n30'):8d} {tot('s_b1n30') - tot('s_b1n15'):8d}")
