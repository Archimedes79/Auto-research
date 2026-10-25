"""Writes e2e_catalogue.json (next to this file) from e2e_catalogue.py (read by the PowerPoint builder for the appendix slides)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e2e_catalogue as CAT  # noqa: E402

data = {"themes": [{"id": a, "name": b, "loops": c} for a, b, c in CAT.THEMES], "exps": [{"theme": a, "id": b, "what": c, "result": d, "tag": e} for a, b, c, d, e in CAT.EXPS], "counts": dict(CAT.counts())}
json.dump(data, open(os.path.join(HERE, "e2e_catalogue.json"), "w", encoding="ascii"), indent=1)
print("json written", len(data["exps"]), data["counts"])
