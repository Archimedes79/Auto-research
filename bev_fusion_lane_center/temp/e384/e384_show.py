import json, sys
SP = sys.argv[1]
for s in (0, 1):
    b = json.load(open(f"{SP}/e384_step0b_s{s}.json"))
    c = json.load(open(f"{SP}/e384_step0c_s{s}.json"))
    print("seed", s, "frames", b["frames"], "gt", b["gt"], "uncovered", round(b["uncovered"], 4), "topo share of GT", round(b["topo_share_of_gt"], 4), "uncovered topo (of GT)", round(b["uncovered_topo_share_of_gt"], 4))
    for k in ("p>=0.5", "p>=0.2", "valid(p>0.05)"):
        print("  ", k, "ceiling pts (all / topo) and dropped-true share:", {n: (round(b[k + "|" + n]["ceiling_pts"], 4), round(b[k + "|" + n]["ceiling_topo_pts"], 4), round(b[k + "|" + n]["dropped_true"], 3)) for n in ("pose", "identity", "random")})
    print("   LR existence AP", {k: round(v, 4) for k, v in c["ap"].items()}, "| point delta vs base: pose", round(c["ap_point_delta"]["pose"], 4), "identity", round(c["ap_point_delta"]["identity"], 4), "random", round(c["ap_point_delta"]["random"], 4))
    print("   scene bootstrap (mean, lo, hi):", {k: (round(v["mean"], 4), round(v["lo"], 4), round(v["hi"], 4)) for k, v in c["ap_delta_vs_base"].items()})
    print("   recall of assigned tokens at the precision of p>=0.5 (%.3f):" % c["precision_at_p50_base"], {k: round(v, 4) for k, v in c["recall_at_that_precision"].items()})
