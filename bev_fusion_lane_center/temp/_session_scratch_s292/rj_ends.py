import json, os, sys
K = ("topo_F1", "dup_frac", "F1@1.0", "junction_F1", "lanes_found", "vert100_dp")
ONLY = {"_thr0.1", "_snap_thr0.1", "_snap_nogeom_thr0.1", "_snap_nogeom_attach_thr0.1", "_attach_thr0.1", "_e338a_snap_nogeom_attach_thr0.1", "_e338b_snap_thr0.1", "_e338_snap_thr0.1"}
for run in sys.argv[1:]:
    d = f"runs/{run}"
    if not os.path.isdir(d):
        print(run, "MISSING"); continue
    for f in sorted(os.listdir(d)):
        if f.startswith("ends_card") and f.endswith(".json") and f.replace("ends_card", "").replace(".json", "") in ONLY:
            try:
                c = json.load(open(os.path.join(d, f), encoding="utf-8"))
                n = c["numbers"]["head"]
                print(run, f.replace("ends_card", "").replace(".json", ""), {k: round(float(n[k]), 4) for k in K if k in n})
            except Exception as e:
                print(run, f, "ERR", e)
