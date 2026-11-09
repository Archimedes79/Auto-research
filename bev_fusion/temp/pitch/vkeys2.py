import json, sys
m = json.load(open(sys.argv[1]))
d = m.get("metrics", m)
for k in sorted(d):
    if k.startswith("det/dt+0.00/mean") or k.startswith("info/") or (k.startswith("seg_fov/seg/dt+0.00/lane_divider/all/all/")) or (k.startswith("seg_fov/seg/dt+0.00/road_divider/all/all/")):
        print(k, d[k])
print([k for k in d if "front" in k.lower()][:10])
print(m.get("checkpoint"), m.get("split"), m.get("weights"), m.get("gt_signature"))
