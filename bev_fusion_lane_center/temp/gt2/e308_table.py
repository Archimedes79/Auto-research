import json
for s in (0, 1):
    for run in (f"e287_p2_s{s}", f"e308_t_s{s}"):
        h, c = json.load(open(f"runs/{run}/ends_head.json")), json.load(open(f"runs/{run}/ends_card.json"))
        n, d = c["numbers"]["head"], c["difference"]
        print(f"{run:12s} ent/ext {h['ent_recall']:.3f}/{h['ext_recall']:.3f} pair R/P {h['pair_recall']:.3f}/{h['pair_precision']:.3f} topoF1 {n['topo_F1']:.4f} ({d['topo_F1']:+.4f}) lanes {n['lanes_found']:.4f} ({d['lanes_found']:+.4f}) jF1 {n['junction_F1']:.4f} jR {n['junction_R']:.4f} dup {n['dup_frac']:.4f} vert {n['vert100_dp']:.2f}")
