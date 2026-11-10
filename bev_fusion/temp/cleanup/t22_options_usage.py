"""scratch (Image Encoder, clean-base round 2026-11-10): which non-default values of the ModelConfig fields are used by a stored config (training/configs/*.json),
by the config of a kept / baseline run (runs/<name>/config.json), by a test or a pre-registered arm. Prints, per field, the default, the non-default values and the configs that set them.
Run from bev_fusion/ with the gcar python."""
import dataclasses
import glob
import json
import os
import sys

sys.path.insert(0, "training")
from model import ModelConfig  # noqa: E402

defaults = {f.name: (f.default if f.default is not dataclasses.MISSING else None) for f in dataclasses.fields(ModelConfig)}
srcs = {}
for p in sorted(glob.glob("training/configs/*.json")):
    srcs[os.path.basename(p)] = p
KEPT = ["t11_vel_hurdle_acc2", "t10_dual_s0", "t10_dual_s1", "t8_ctl_s0", "t8_ctl_s1", "t20_base_s0", "t21_pair4_s0", "t21_ctl_s0", "t22_B_s0", "t23_C_s0", "t23_C_s1", "t21_ctl_s1"]
for n in KEPT:
    p = f"runs/{n}/config.json"
    if os.path.exists(p):
        srcs["runs/" + n] = p
    else:
        print("no stored config for", n)
use = {}
for name, p in srcs.items():
    c = json.load(open(p))
    m = c.get("model", c)
    for k, v in m.items():
        if k.startswith("_"):
            continue
        if k not in defaults:
            use.setdefault("UNKNOWN:" + k, []).append(name)
            continue
        if v != defaults[k]:
            use.setdefault(k, {}).setdefault(json.dumps(v)[:60], []).append(name)
print("fields never set to a non-default value in any of", len(srcs), "configs:")
print("  ", sorted(k for k in defaults if k not in use))
print()
for k in sorted(k for k in use if not k.startswith("UNKNOWN")):
    print(k, "(default", json.dumps(defaults[k])[:40] + "):")
    for v, names in sorted(use[k].items()):
        short = [n.replace(".json", "") for n in names]
        print("    ", v, "<-", len(names), ":", ", ".join(short[:6]) + (" ..." if len(short) > 6 else ""))
for k in sorted(k for k in use if k.startswith("UNKNOWN")):
    print(k, use[k])
