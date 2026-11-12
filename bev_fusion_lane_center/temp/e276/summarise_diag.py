import json, glob, os
for p in sorted(glob.glob("runs/e276diag/*/card.json")):
    c = json.load(open(p)); n = c["p"]["nodes"]; g = c["p"]["graph"]
    print(os.path.basename(os.path.dirname(p)).ljust(26), "frames", c["frames"], "nodes/100m %.1f" % n["nodes_per_100m"], "nodeP %.3f nodeR %.3f" % (n["node_P"], n["node_R"]), "topo_F1 %.3f F1@1 %.3f lanes %.3f" % (g["topo_F1"], g["F1@1.0"], g["lanes_found"]))
