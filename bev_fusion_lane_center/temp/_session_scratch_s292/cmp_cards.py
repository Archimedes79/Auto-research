import json, os
R = os.environ["BFLC_ROOT"] + "/runs/e312_p_s0/"
a = json.load(open(R + "ends_card_snap_thr0.1.json")); b = json.load(open(R + "ends_card_regress_snap_thr0.1.json"))
print("numbers identical:", a["numbers"] == b["numbers"], "| curves", a["curves"], b["curves"])
print({k: b[k] for k in ("full", "spec", "tag", "snap")})
print({k: round(b["numbers"]["head"][k], 4) for k in ("topo_F1", "dup_frac", "junction_F1")})
