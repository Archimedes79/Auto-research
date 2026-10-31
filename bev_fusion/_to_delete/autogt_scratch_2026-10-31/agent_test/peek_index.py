import json, os
root = os.path.expanduser("~/dev_none")
base = "data/autogt"
print(os.listdir(base))
d = json.load(open(os.path.join(base, "7fe1149566", "index.json")))
print(type(d).__name__, len(d))
if isinstance(d, dict):
    for k in d:
        print(k, type(d[k]).__name__, str(d[k])[:200])
else:
    print(str(d[:2])[:400])
