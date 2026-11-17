import json, os
T = json.load(open(os.environ["BFLC_ROOT"] + "/temp/_session_scratch_s292/e324_table.json"))
for s in (0, 1):
    k, kn, a = (T[f"{s}|{n}"]["topo_F1"] for n in ("s  own kp snapped", "s  own kp nogeom", "s  PRIMARY attach+nogeom"))
    print(f"seed {s}: k-kn {k-kn:.6f}  a-kn {a-kn:.6f}  k-a {k-a:.6f}  share {(a-kn)/(k-kn):.3f}")
