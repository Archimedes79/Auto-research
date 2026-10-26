import json, os
from nuscenes.map_expansion.map_api import NuScenesMap
root = os.path.expanduser("~/data/nuscenes")
d = json.load(open(os.path.join(root, "maps/expansion/singapore-hollandvillage.json")))
print("drivable_area records:", len(d["drivable_area"]), "with None token:", [r["token"][:8] for r in d["drivable_area"] if not all(r["polygon_tokens"])])
print("polygons without a token:", sum(1 for p in d["polygon"] if not p.get("token")), "keys of first polygon:", list(d["polygon"][0].keys()))
nm = NuScenesMap(dataroot=root, map_name="singapore-hollandvillage")
print("None in token2ind:", None in nm._token2ind["polygon"], "-> index", nm._token2ind["polygon"].get(None))
rec = nm.get("polygon", None)
print({k: (v if not isinstance(v, list) else f"list[{len(v)}]") for k, v in rec.items()})
# how many drivable records have a None token in the other locations?
for loc in ("boston-seaport", "singapore-onenorth", "singapore-queenstown", "singapore-hollandvillage"):
    dd = json.load(open(os.path.join(root, f"maps/expansion/{loc}.json")))
    print(loc, "drivable_area None-token records:", sum(1 for r in dd["drivable_area"] if not all(r["polygon_tokens"])), "of", len(dd["drivable_area"]))
