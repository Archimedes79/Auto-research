import json, os
p = os.path.join(os.environ["BFLC_ROOT"], "results", "E2E_Champion_Report", "report.json")
r = json.load(open(p, encoding="utf-8"))
u = [x for x in r["utilities"] if "render_q1115" not in x["path"]]
u.append({"path": "../../temp/qual_report_1117/select_champion_q1117.py", "usage": "wsl bash tools/env/_py14.sh select_champion_q1117.py   (writes spec + stats of the picture frames)"})
u.append({"path": "../../temp/qual_report_1117/render_champion_q1117.py", "usage": "wsl bash tools/env/_py14.sh render_champion_q1117.py <spec.json> <out dir>   (CPU; scratch derived from render_q1115.py, to be promoted)"})
r["utilities"] = u
json.dump(r, open(p, "w", encoding="utf-8"), indent=1)
print(len(u))
