import json, sys
for f in sys.argv[1:]:
    d = json.load(open(f)); m = d["metrics"]
    print(f.split("/")[-1])
    print("  topo_F1 %.4f  junction_F1 %.4f  F1@1.0 %.4f  lanes_found %.4f  dup_frac %.4f  Q %.4f" % tuple(m[k] for k in ("topo_F1", "junction_F1", "F1@1.0", "lanes_found", "dup_frac", "Q")))
    for k, v in d["info"].items():
        if k.startswith("paired") and isinstance(v, dict):
            t = v.get("topo_F1")
            if t: print("  ", k, "topo_F1 diff %+.4f [%+.4f, %+.4f]" % (t["diff"], t["lo"], t["hi"]))
