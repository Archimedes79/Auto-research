import json, os, sys
ROOT = os.environ.get("BFLC_ROOT", ".")
KEYS = ("Q", "F1@1.0", "lanes_found", "topo_F1", "topo_P", "topo_R", "junction_F1", "junction_P", "junction_R", "dup_frac", "vert100_dp")
STORED = {"snap": "ends_card_snap_thr0.1.json", "primary": "ends_card_snap_nogeom_attach_thr0.1.json", "attach": "ends_card_attach_thr0.1.json"}
NEW = {"snap": "ends_card_null_snap_thr0.1.json", "primary": "ends_card_null_snap_nogeom_attach_thr0.1.json", "attach": "ends_card_null_attach_thr0.1.json"}
mx = {k: [0.0, None] for k in KEYS}
for S in (0, 1):
    for k in STORED:
        pa = f"runs/e324_s_s{S}/{STORED[k]}"; pb = f"temp/e352/diag_null_s{S}/{NEW[k]}"
        if not os.path.exists(pb):
            print("missing", pb); continue
        a = json.load(open(pa))["numbers"]; b = json.load(open(pb))["numbers"]
        for w in ("base", "head"):
            row = []
            for key in KEYS:
                d = b[w][key] - a[w][key]
                row.append(f"{key}:{d:+.1e}")
                if abs(d) > mx[key][0]: mx[key] = [abs(d), (S, k, w, a[w][key])]
            print(S, k, w, " ".join(row))
print("max |diff| per number (and where, with the stored value):")
for key, v in mx.items():
    print(f"  {key:12s} {v[0]:.2e}  {v[1]}")
