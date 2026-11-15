import json
for r in ("e316_p_s0", "e312_p_s0"):
    h = json.load(open(f"runs/{r}/ends_head.json"))
    print(r, {k: (v if not isinstance(v, (dict, list)) else type(v).__name__) for k, v in h.items()})
c = json.load(open("runs/e316_p_s0/ends_card_snap_thr0.1.json"))
print(list(c.keys())[:40])
import os
print(sorted(os.listdir("runs/e316_p_s0_maps")))
