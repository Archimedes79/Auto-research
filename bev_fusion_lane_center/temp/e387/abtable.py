"""a / b of the topo F1 (and the paired diff) for every E387 comparison card (scratch)"""
import glob, json, sys
sys.path.insert(0, "temp/e387")
for s in (0, 1):
    print("seed", s)
    for f in sorted(glob.glob(f"runs/e387_f_s{s}/token_card_*vs*.json")):
        d = json.load(open(f))
        p = d["info"]["paired_vs_ref_run"]
        v = d["info"]["decode_variant"]
        rr = d["info"]["ref_run"]
        cell = f"{v['link_thr']}/{v.get('link_thr_long', v['link_thr'])}/{v['keep_thr']}"
        t = p["topo_F1"]
        print(f"  f at {cell:<12} vs {rr['run']:<14} ref cell {str(rr['cell'])[:60]:<60} topo F1 f {t['a']:.4f} ref {t['b']:.4f} diff {t['diff']:+.4f} [{t['lo']:+.4f}, {t['hi']:+.4f}]  lanes {p['lanes_found']['a']:.4f}/{p['lanes_found']['b']:.4f} dup {p['dup_frac']['a']:.4f}/{p['dup_frac']['b']:.4f} junction {p['junction_F1']['a']:.4f}/{p['junction_F1']['b']:.4f} F1@1 {p['F1@1.0']['a']:.4f}/{p['F1@1.0']['b']:.4f}")
