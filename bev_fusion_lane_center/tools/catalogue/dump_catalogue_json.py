"""Writes scripts/e2e_catalogue.json from scripts/e2e_catalogue.py (read by the PowerPoint builder for the appendix slides)."""
import json
import sys

sys.path.insert(0, "/mnt/c/ML-Experiment/bev_fusion_lane_center/training")
import e2e_catalogue as CAT  # noqa: E402

data = {"themes": [{"id": a, "name": b, "loops": c} for a, b, c in CAT.THEMES], "exps": [{"theme": a, "id": b, "what": c, "result": d, "tag": e} for a, b, c, d, e in CAT.EXPS], "counts": dict(CAT.counts())}
json.dump(data, open("/mnt/c/ML-Experiment/bev_fusion_lane_center/tools/catalogue/e2e_catalogue.json", "w", encoding="ascii"), indent=1)
print("json written", len(data["exps"]), data["counts"])
