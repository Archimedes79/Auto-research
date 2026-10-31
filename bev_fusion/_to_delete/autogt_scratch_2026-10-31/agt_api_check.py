"""scratch: the new top-level API (current_signature, load) and the AGENTS.md key list against a real stored keyframe."""
import glob
import os
import sys

sys.path.insert(0, ".")
import autogt  # noqa: E402

print("current_signature", autogt.current_signature(), "| annotation", autogt.current_signature(boxes_at="annotation"))
tok = os.path.basename(sorted(glob.glob("data/autogt/15efbde2d1/*.npz"))[100])[:-4]
g = autogt.load(tok)
guide = open("autogt/AGENTS.md", encoding="utf-8").read()
keys = [k for k in g if k != "meta"] + list(g["meta"]) + list(g["meta"]["boxes"][0]) + ["dt", "steps"]
print("keys", len(keys), "missing in guide:", [k for k in keys if f"`{k}`" not in guide])
try:
    autogt.load("0" * 32)
except FileNotFoundError as e:
    print("error text ok:", str(e)[:200])
