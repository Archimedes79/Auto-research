import json
for r in ("e312_p_s0","e341_p_s0"):
    for f in ("ends_card_snap_thr0.1.json","ends_card_thr0.1.json","ends_card_cnt_thr0.1.json"):
        try:
            c=json.load(open(f"runs/{r}/{f}"))["numbers"]["head"]
            print(r,f,{k:round(c[k],4) for k in ("topo_F1","topo_P","junction_F1","lanes_found","dup_frac","F1@1.0","vert100_dp")})
        except Exception as e: print(r,f,"ERR",type(e).__name__)
