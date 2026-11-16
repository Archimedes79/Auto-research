"""Throw-away: do the stored route-token pickles hold items for frames with an empty graph (None)?  Read-only."""
import glob
import os
import pickle
import sys

HOME = os.path.expanduser("~")
for f in sorted(glob.glob(f"{HOME}/exp_gcar_b123/p10/*.pkl"))[:12]:
    try:
        z = pickle.load(open(f, "rb"))
    except Exception as e:  # noqa: BLE001
        print(os.path.basename(f), "unreadable", type(e).__name__)
        continue
    items = z.get("items") if isinstance(z, dict) else None
    if items is None:
        print(os.path.basename(f), "keys", list(z)[:6] if isinstance(z, dict) else type(z))
        continue
    none = sum(1 for it in items if (it["item"] if isinstance(it, dict) and "item" in it else it) is None)
    print(os.path.basename(f), "items", len(items), "with item None:", none, "part", z.get("part"))
sys.stdout.flush()
