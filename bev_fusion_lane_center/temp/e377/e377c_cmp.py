"""E377 throw-away: compare e377c_identity_head.json with e377c_identity_scratch.json"""
import json
import os

H = os.path.dirname(os.path.abspath(__file__))
h, s = (json.load(open(os.path.join(H, f"e377c_identity_{n}.json"))) for n in ("head", "scratch"))
for n in h:
    print(n, "losses equal" if h[n]["losses"] == s[n]["losses"] else "LOSSES DIFFER", "weights equal" if h[n]["sha"] == s[n]["sha"] else "WEIGHTS DIFFER", len(h[n]["losses"]), "steps")
