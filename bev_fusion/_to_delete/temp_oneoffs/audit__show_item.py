"""One-off: keys, shapes and meta fields of one cached training item (for the flip augmentation)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "training")
from train import TrainConfig, make_datasets  # noqa: E402

cfg = TrainConfig(**json.load(open("runs/t1_r3_combined/config.json")))
cfg.workers = 0
tr, va = make_datasets(cfg)
it = va[100]
print("item keys:", list(it))
print("ego:", it["ego"])
print("radar keys:", {k: tuple(v.shape) for k, v in it["radar"].items()})
print("frames[0] keys:", {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in it["frames"][0].items()})
print("frame steps lens:", [len(f["steps"]) for f in it["frames"]], it["frames"][1]["steps"][:2])
for key in ("cur", "prev", "next"):
    g = it["gt"][key]
    print("== gt", key)
    for k, v in g.items():
        if k == "meta":
            continue
        print(f"  {k:18s} {type(v).__name__:8s} {tuple(np.shape(v))} {getattr(v, 'dtype', '')}")
    m = g["meta"]
    print("  meta keys:", list(m))
    print("  meta steps[:2]:", m.get("steps", [])[:2] if "steps" in m else None)
    if m["boxes"]:
        print("  box0:", {k: (np.round(v, 2).tolist() if isinstance(v, (list, tuple, np.ndarray)) else v) for k, v in m["boxes"][0].items()})
    break
