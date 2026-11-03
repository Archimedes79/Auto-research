import json, glob
for f in sorted(glob.glob("runs/*/eval/card.json")):
    c = json.load(open(f))
    print(f.split("/")[1], "provenance" in c)
for r in ("t10_dual_s0_gen",):
    c = json.load(open(f"runs/{r}/eval/card.json"))
    print(json.dumps(c.get("provenance"), indent=1)[:2400])
