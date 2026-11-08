import re
dec = open("bev_fusion_lane_center/experiment_execution/DECISIONS.md", encoding="utf-8").read()
ids = sorted({int(m) for m in re.findall(r"^\|\s*(?:\d{4}-\d{2}-\d{2}\s*\|\s*)?D(\d+)\b", dec, re.M)})
open_sec = dec.split("## Open")[1].split("## Decided")[0]
rows = [l for l in open_sec.splitlines() if re.match(r"^\| D\d+", l)]
open_real = [re.match(r"^\| (D\d+)", l).group(1) for l in rows if "(decided" not in l.split("|")[1]]
print(len(ids), ids[0], ids[-1], "missing:", [i for i in range(1, ids[-1] + 1) if i not in ids])
print("open:", open_real)
