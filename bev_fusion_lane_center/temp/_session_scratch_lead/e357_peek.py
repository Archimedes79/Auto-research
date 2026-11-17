import json,glob,os
for f in sorted(glob.glob("runs/e357_p_s*/tok_card_*.json")):
    m=json.load(open(f))["metrics"]
    print(os.path.basename(os.path.dirname(f)), os.path.basename(f)[9:-5], "topoF1 %.4f P %.3f R %.3f dup %.3f jF1 %.3f F1@1 %.3f lanes %.3f" % (m["topo_F1"],m["topo_P"],m["topo_R"],m["dup_frac"],m["junction_F1"],m["F1@1.0"],m["lanes_found"]))
