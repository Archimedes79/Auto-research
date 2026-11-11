import json, sys
c = json.load(open(sys.argv[1]))
print("graph:", sorted(c["p"]["graph"].keys()))
print("nodes:", sorted(c["p"]["nodes"].keys()))
print("keys of p:", list(c["p"].keys()), "top:", list(c.keys()))
print("g keys:", {k: list(v.keys()) for k, v in c["g"].items()})
print("diff:", c["diff"])
print("readout:", c["readout"])
