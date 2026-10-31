"""scratch: add the night sequences of nuScenes mini (stored at the final signature) to picks.json; writes picks_night.json for render.py"""
import json
import sys

sys.path.insert(0, ".")
import autogt  # noqa: E402

scenes = {s["name"]: s for s in json.load(open("/home/ml/data/nuscenes_full/v1.0-trainval/scene.json"))}
logs = {r["token"]: r["location"] for r in json.load(open("/home/ml/data/nuscenes_full/v1.0-trainval/log.json"))}
new = []
for name, k in (("scene-1077", 8), ("scene-1094", 10), ("scene-1100", 8)):
    tok = autogt.scene_tokens(name, "v1.0-trainval")[k]
    autogt.load(tok)                                                      # must be stored
    sc = scenes[name]
    new.append(dict(scene=name, k=k, tok=tok, title="night", note="night: the GT does not use the image", loc=logs[sc["log_token"]], desc=sc["description"]))
json.dump(dict(examples=new, thumbs=[]), open("temp/agt_gallery/picks_night.json", "w"), indent=1)
picks = json.load(open("temp/agt_gallery/picks.json"))
at = max(i for i, p in enumerate(picks["examples"]) if p["title"].startswith("location")) + 1
picks["examples"][at:at] = new
json.dump(picks, open("temp/agt_gallery/picks.json", "w"), indent=1)
print([(p["scene"], p["loc"], p["desc"][:50]) for p in new])
