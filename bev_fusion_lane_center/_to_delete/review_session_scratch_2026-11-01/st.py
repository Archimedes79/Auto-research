import sys, os, json
sys.path.insert(0, os.environ["BFLC_ROOT"] + "/tools/prototype")
import lane_prototype as LP, evaluate as EV
reg = EV.registry()
for n in ("recall_dp", "recall", "champion", "champion_dp", "fast_recall", "e24_j1rc_s3", "ens_j1rc_23", "recall_ring"):
    e = reg[n]; base = reg[e["members"][0]] if e["kind"] == "union" else e
    t = LP.stored_timing(e, EV.builder_cfg(base).temporal)
    print(n, json.dumps({k: v for k, v in t.items() if k != "files"}))
