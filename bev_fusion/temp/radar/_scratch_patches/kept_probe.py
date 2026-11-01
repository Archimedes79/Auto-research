import json
for line in open("runs/r2_rad_slots64_s0/log.jsonl", encoding="utf-8"):
    d = json.loads(line)
    if "val" in d:
        v = d["val"]
        print(d["step"], "val total", round(v.get("total", 0), 3), "radar_kept", round(v.get("radar_kept", float("nan")), 3), "groups", {k[6:]: round(x, 3) for k, x in v.items() if k.startswith("group.")})
