import json, pickle, os
R = os.environ["BFLC_ROOT"]
H = R + "/temp/qual_report_1117"
rows = json.load(open(H + "/perframe_table.json"))
for n, v in rows["sys"].items():
    print(n, {k: (type(x).__name__, len(x) if hasattr(x, "__len__") else x) for k, x in v.items() if k not in ("dump",)})
P = pickle.load(open(H + "/pass1_with_champion.pkl", "rb"))
print(list(P.keys()))
f = P["frames"][0]
print(sorted(f.keys())); print({k: sorted(v.keys()) for k, v in f["sys"].items()})
print({k: f["sys"]["champion_S1_s0"][k] for k in ("man_ok", "man_n")}, len(f["sys"]["champion_S1_s0"]["lines"]))
