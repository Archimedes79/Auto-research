import json, os, glob
R = os.environ["BFLC_ROOT"] + "/runs/"
K = ("topo_F1", "F1@1.0", "junction_F1", "dup_frac", "vert100_dp", "lanes_found")
for s in (0, 1):
    for ro, f in (("unsnapped", "ends_card_thr0.1.json"), ("snapped", "ends_card_snap_thr0.1.json")):
        row = {}
        for nm, run in (("E332 a", f"e332_pa_s{s}"), ("E331", f"e331_e_s{s}"), ("E312 c", f"e312_p_s{s}")):
            h = json.load(open(f"{R}{run}/{f}"))
            row[nm] = ({k: round(h["numbers"]["head"][k], 4) for k in K}, h["curves"])
        print(f"s{s} {ro}")
        for nm, (v, cu) in row.items():
            print(f"   {nm:7s} {v} curves {cu}")
    for nm, run in (("maps E332 pair run", f"e332_pa_s{s}_maps"), ("maps E331", f"e331_e_s{s}_maps"), ("maps E312", f"e312_p_s{s}_maps"), ("maps stage 1 e332_ma", f"e332_ma_s{s}"), ("E307", f"e307_n_s{s}")):
        p = f"{R}{run}/jmaps_val.json"
        if os.path.exists(p):
            d = json.load(open(p))
            print(f"   {nm}: kp F1 {d['kp_f1_best_threshold']['ours']:.4f}, zone F1 {d.get('zone_f1')}, core F1 {d.get('core_f1')}")
        else:
            print(f"   {nm}: no jmaps_val.json")
