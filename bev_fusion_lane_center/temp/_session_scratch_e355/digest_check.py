"""throw-away: sha256 of the files of the E343 code digest against the stored card (a digest file that changed is listed)"""
import hashlib
import json
import os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
card = json.load(open(os.path.join(root, "bev_fusion_lane_center/runs/eval/e324s_val/card_s0.json")))
bad = []
for rel, h in card["code_sha256"].items():
    p = os.path.join(root, rel)
    now = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else "MISSING"
    if now != h:
        bad.append(rel)
print(len(card["code_sha256"]), "files in the digest;", len(bad), "differ from the stored hash:", bad)
