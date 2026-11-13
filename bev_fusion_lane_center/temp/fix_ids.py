p = "training/data/lane_gt_data.py"
s = open(p, encoding="utf-8").read()
s = s.replace("    out.update(map_pos=pos, map_tan=tan,", "    out.update(map_ids=[o[\"mp\"][\"ids\"] for o in b], map_roads=[o[\"mp\"][\"roads\"] for o in b], map_pos=pos, map_tan=tan,")
open(p, "w", encoding="utf-8").write(s)
