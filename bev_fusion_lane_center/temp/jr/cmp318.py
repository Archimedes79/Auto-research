import json
for s in (0, 1):
    for f in ("ends_card_snap_thr0.1.json", "ends_card_thr0.1.json", "ends_card_snap_nogeom_thr0.1.json"):
        row = []
        for r in (f"e317_g_s{s}", f"e318_u_s{s}"):
            try:
                h = json.load(open(f"runs/{r}/{f}"))["numbers"]["head"]
                row.append((r, round(h["topo_F1"], 4), round(h["dup_frac"], 4), round(h["junction_F1"], 4)))
            except Exception as e:
                row.append((r, "ERR"))
        d = round(row[1][1] - row[0][1], 4) if isinstance(row[1][1], float) and isinstance(row[0][1], float) else None
        print(s, f, row, "diff", d)
    try:
        j = json.load(open(f"temp/e318/sigma_check_e318_u_s{s}.json"))
        print("  sigma_check", {k: j[k] for k in list(j)[:12]})
    except Exception as e:
        print("  sigma_check ERR", e)
