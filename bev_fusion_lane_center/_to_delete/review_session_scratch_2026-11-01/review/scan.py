import glob, json, os, sys
sys.path.insert(0, "/mnt/c/Auto-research/autoresearch/tools")
for proj in ("bev_fusion_lane_center", "bev_fusion"):
    root = "/mnt/c/Auto-research/" + proj
    for f in sorted(glob.glob(root + "/experiments/*/experiment*.json")):
        e = json.load(open(f))
        rule = e.get("rule", [])
        hm = [c for c in rule if isinstance(c, dict) and c.get("metric") == e.get("metric")]
        kinds = ["min" if "min_effect" in c else "max" for c in hm]
        subs = any(c.get("subsets") for c in hm)
        dec = glob.glob(os.path.dirname(f) + "/decision*.json")
        old = json.load(open(dec[0]))["verdict"] if dec else "-"
        new = "-"
        try:
            import decide as DC
            new = DC.decide_experiment(e, root)["verdict"]
        except BaseException as ex:
            new = "ERR " + type(ex).__name__ + " " + str(ex)[:60]
        print(proj, os.path.basename(os.path.dirname(f)), e.get("metric"), e.get("direction"), kinds, "subsets" if subs else "", "old", old, "new", new)
