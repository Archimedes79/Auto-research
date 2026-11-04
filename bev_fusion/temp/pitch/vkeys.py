import json, sys
p = sys.argv[1]
m = json.load(open(p))
d = m.get("metrics", m)
ks = sorted(d)
print(len(ks), "keys; top-level:", list(m)[:8])
for k in ks:
    kl = k.lower()
    if ("skill" in kl or "ave" in kl or "vel" in kl or "static" in kl or "moving" in kl) and "seg_fov" not in kl:
        v = d[k]
        print(k, v if not isinstance(v, (dict, list)) else type(v).__name__)
