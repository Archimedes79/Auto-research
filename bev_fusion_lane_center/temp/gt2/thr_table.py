import glob, json, re
rows = []
for f in sorted(glob.glob("temp/gt2/thr/*.json")):
    d = json.load(open(f))
    n, b = d["numbers"]["head"], d["numbers"]["base"]
    rows.append((d["run"], d["thr"], d["curves"], n["topo_F1"] - b["topo_F1"], n["topo_F1"], n["lanes_found"] - b["lanes_found"], n["junction_F1"] - b["junction_F1"], n["junction_R"] - b["junction_R"], n["dup_frac"] - b["dup_frac"], n["F1@1.0"] - b["F1@1.0"], n["vert100_dp"] - b["vert100_dp"]))
print("run thr curves | topo F1 (d) | lanes d | jF1 d | jR d | dup d | F1@1.0 d | vert d")
for r in sorted(rows, key=lambda x: (x[0], x[1])):
    print("%-12s %.1f %5d | %.4f (%+.4f) | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f | %+.2f" % (r[0], r[1], r[2], r[4], r[3], r[5], r[6], r[7], r[8], r[9], r[10]))
