"""E320 read: one table per seed from the stored cards (scratch). usage: read_result.py <seed>"""
import json, os, sys
S = sys.argv[1]
KEYS = ("topo_F1", "topo_P", "topo_R", "lanes_found", "junction_F1", "junction_R", "dup_frac", "F1@1.0", "vert100_dp", "Q")
ROWS = [
    ("c  = E312 snapped", f"e312_p_s{S}", "ends_card_snap_thr0.1.json"),
    ("u' = E312 snap-nogeom", f"e312_p_s{S}", "ends_card_snap_nogeom_thr0.1.json"),
    ("u  = E312 unsnapped", f"e312_p_s{S}", "ends_card_thr0.1.json"),
    ("E312 snap-all", f"e312_p_s{S}", "ends_card_snap_all_thr0.1.json"),
    ("E320 PRIMARY attach+snap-nogeom", f"e320_a_s{S}", "ends_card_snap_nogeom_attach_thr0.1.json"),
    ("E320 own keypoints snapped", f"e320_a_s{S}", "ends_card_snap_thr0.1.json"),
    ("E320 own keypoints snap-nogeom", f"e320_a_s{S}", "ends_card_snap_nogeom_thr0.1.json"),
    ("E320 attach, proximity (secondary)", f"e320_a_s{S}", "ends_card_attach_thr0.1.json"),
    ("E320 attach r1.5", f"e320_a_s{S}", "ends_card_attach_r1.5_thr0.1.json"),
    ("E320 attach r1.0", f"e320_a_s{S}", "ends_card_attach_r1.0_thr0.1.json"),
    ("E320 own keypoints not snapped", f"e320_a_s{S}", "ends_card_thr0.1.json"),
]
print("seed", S, " ".join(f"{k:>11}" for k in KEYS), "  curves  wired_both")
for name, run, f in ROWS:
    p = f"runs/{run}/{f}"
    if not os.path.exists(p):
        print(f"{name:36s} (missing)"); continue
    c = json.load(open(p)); n = c["numbers"]["head"]
    print(f"{name:36s}", " ".join(f"{n[k]:11.4f}" for k in KEYS), f"  {c.get('curves')}  {c.get('wired_both_ends')}")
