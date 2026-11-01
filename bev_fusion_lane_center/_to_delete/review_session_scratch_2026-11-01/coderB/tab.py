import json
d = json.load(open("l1_impact.json"))
keys = ["S", "Q", "F1@1.0", "J", "topo_F1", "lanes_found", "R@1_far", "vert100", "F1o2o@1.0", "dup_frac", "dir_acc"]
print("name | " + " | ".join(keys))
for n, r in d.items():
    print(n + " | " + " | ".join(f"{r['diff'][k][0]:.4f}->{r['diff'][k][1]:.4f}" if k in r["diff"] else "=" for k in keys))
    big = {k: round(v[1]-v[0], 4) for k, v in r["diff"].items() if abs(v[1]-v[0]) >= 5e-4}
    print("   |d| >= 5e-4:", big)
