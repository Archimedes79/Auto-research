import glob, json, sys
d_ = sys.argv[1]
print("file | topo F1 (d) | lanes d | jF1 d | jR d | dup d | F1@1.0 d | vert d | curves")
for f in sorted(glob.glob(f"../../temp/gt2/{d_}/*.json")):
    d = json.load(open(f))
    n, b = d["numbers"]["head"], d["numbers"]["base"]
    print("%-46s | %.4f (%+.4f) | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f | %+.2f | %d wired %d" % (f.split("/")[-1], n["topo_F1"], n["topo_F1"] - b["topo_F1"], n["lanes_found"] - b["lanes_found"], n["junction_F1"] - b["junction_F1"], n["junction_R"] - b["junction_R"], n["dup_frac"] - b["dup_frac"], n["F1@1.0"] - b["F1@1.0"], n["vert100_dp"] - b["vert100_dp"], d["curves"], d["wired_both_ends"]))
