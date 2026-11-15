import json, sys, os
for r in sys.argv[1:]:
    p = f"../../runs/{r}/ends_head.json"
    if not os.path.exists(p):
        print(r, "pending")
        continue
    h = json.load(open(p))
    print(f"{r:12s} ent/ext recall {h['ent_recall']:.3f}/{h['ext_recall']:.3f} pair R/P {h['pair_recall']:.3f}/{h['pair_precision']:.3f}")
