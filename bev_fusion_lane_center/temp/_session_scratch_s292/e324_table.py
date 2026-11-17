"""E324 result table from the stored cards (numbers come from the files, none typed)"""
import json, os
R = os.environ["BFLC_ROOT"] + "/runs/"
K = ("topo_F1", "dup_frac", "F1@1.0", "lanes_found", "junction_F1", "vert100_dp", "topo_P", "topo_R")
def head(run, name):
    return json.load(open(f"{R}{run}/{name}"))["numbers"]["head"]
rows = {
 "c  E312 snapped": ("e312_p_s%d", "ends_card_snap_thr0.1.json"),
 "u' E312 nogeom": ("e312_p_s%d", "ends_card_snap_nogeom_thr0.1.json"),
 "l  E316 snapped": ("e316_p_s%d", "ends_card_snap_thr0.1.json"),
 "l' E316 nogeom": ("e316_p_s%d", "ends_card_snap_nogeom_thr0.1.json"),
 "s  PRIMARY attach+nogeom": ("e324_s_s%d", "ends_card_snap_nogeom_attach_thr0.1.json"),
 "s  own kp snapped": ("e324_s_s%d", "ends_card_snap_thr0.1.json"),
 "s  own kp nogeom": ("e324_s_s%d", "ends_card_snap_nogeom_thr0.1.json"),
}
out = {}
for s in (0, 1):
    print(f"== seed {s}")
    print(f"{'':28s}" + "".join(f"{k:>12s}" for k in K))
    for n, (r, f) in rows.items():
        h = head(r % s, f)
        out[(s, n)] = h
        print(f"{n:28s}" + "".join(f"{h[k]:12.4f}" for k in K))
json.dump({f"{s}|{n}": h for (s, n), h in out.items()}, open(os.environ["BFLC_ROOT"] + "/temp/_session_scratch_s292/e324_table.json", "w"), indent=1)
