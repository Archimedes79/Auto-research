import json
d = json.load(open("splits/small.json"))
print("val:", d["val"])
print("train n:", len(d["train"]))
