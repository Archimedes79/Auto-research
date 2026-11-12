import glob, json
print("run thr | topo F1 (d) | lanes d | jF1 d | jR d | dup d | F1@1.0 d | topoP d | topoR d")
for f in sorted(glob.glob("../../temp/gt2/snap/*.json")):
    d = json.load(open(f))
    n, b = d["numbers"]["head"], d["numbers"]["base"]
    print("%-10s %.1f | %.4f (%+.4f) | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f   curves %d" % (d["run"], d["thr"], n["topo_F1"], n["topo_F1"] - b["topo_F1"], n["lanes_found"] - b["lanes_found"], n["junction_F1"] - b["junction_F1"], n["junction_R"] - b["junction_R"], n["dup_frac"] - b["dup_frac"], n["F1@1.0"] - b["F1@1.0"], n["topo_P"] - b["topo_P"], n["topo_R"] - b["topo_R"], d["curves"]))
