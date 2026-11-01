"""one-off: the model bundle of a real bev_fusion run, nothing written (write=False)"""
import json
import os
import sys

sys.path.insert(0, "autoresearch/tools")
os.environ["AR_PROJECT"] = "bev_fusion"
import provenance as V  # noqa: E402

b = V.manifest(sys.argv[1], write=False)
print("bundle:", len(b["bundle"]["files"]), "files, hash", b["bundle"]["code_hash"])
print(b["bundle"]["files"])
print("gt:", b["gt"].get("version"), b["gt"].get("signature"), "| ckpt:", b["subject"].get("ckpt"), b["subject"].get("step"), "| seed:", b["seed"], "| code:", b["code"]["tag"])
print("data:", {k: b["data"].get(k) for k in ("split", "manifest_hash", "frame_hash")})
