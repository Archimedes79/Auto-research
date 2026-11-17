import json
import os
import sys

W = os.path.dirname(os.path.abspath(__file__))
R = os.path.normpath(os.path.join(W, "..", "..", ".."))
fx = json.load(open(os.path.join(W, "fx20.json")))
pot = json.load(open(os.path.join(W, "pot60.json")))
chk = json.load(open(os.path.join(W, "chk60.json")))
pins = dict(fixture={}, potential={}, check={})
for k in ("shift_min", "shift_max", "yaw_std_deg", "exact_samples", "own_share_error", "own_share_S", "connector_share_cut", "connector_share_uncut", "connector_samples"):
    pins["fixture"][k] = fx[k]
for h in ("undo", "identity", "shift_sign", "rotation_transposed"):
    for kk in ("n", "median", "p90", "p99"):
        pins["fixture"][f"undo.{h}.{kk}"] = fx["undo"][h][kk]
pins["potential"]["samples_per_frame"] = pot["samples_per_frame"]
for c, v in pot["classes"].items():
    for kk, vv in v.items():
        pins["potential"][f"classes.{c}.{kk}"] = vv
for i, b in enumerate(pot["by_bin"]):
    for kk in ("n", "true_share", "supported_if_true", "supported_if_false", "novel_share_supported", "novel_share_unsupported"):
        pins["potential"][f"by_bin.{i}.{kk}"] = b[kk]
for s in ("val", "train"):
    pins["check"][f"{s}.frames"] = chk[s]["frames"]
    pins["check"][f"{s}.samples_with_count_above_max"] = chk[s]["samples_with_count_above_max"]
    for m in ("exact", "error"):
        for kk in ("frames", "mean", "max", "truncated"):
            pins["check"][f"{s}.sizes.{m}.{kk}"] = chk[s]["sizes"][m][kk]
for k in ("train_frames", "val_test_frames"):
    pins["check"][f"train_guard.{k}"] = chk["train_guard"][k]
txt = json.dumps(pins, indent=1)
p = os.path.join(R, "training", "grid", "test_lane_map.py")
s = open(p, encoding="utf-8").read()
if "PINS_JSON" in s:
    s = s.replace("PINS_JSON", txt)
else:
    a = s.index('PINS = json.loads("""') + len('PINS = json.loads("""')
    b = s.index('""")', a)
    s = s[:a] + txt + s[b:]
open(p, "w", encoding="utf-8").write(s)
print(len(pins["fixture"]), len(pins["potential"]), len(pins["check"]))
