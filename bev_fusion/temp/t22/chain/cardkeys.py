import json
c = json.load(open("runs/t21_pair4_s0/eval/card.json"))
print([k for k in c if k != "metrics"])
print(c.get("protocol"))
def find(d, key, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key: print(path + "/" + k, v)
            find(v, key, path + "/" + k)
find(c, "frame_hash"); find(c, "gt_signature"); find(c, "code")
