"""throw-away, read-only, one thread: the share of GT boxes with an unknown velocity (finding #2 of records/open_findings_2026-11-02.md). Samples ~300 keyframes spread over the newest GT store."""
import glob
import os
import sys

import numpy as np

store = sorted(glob.glob("/mnt/c/Auto-research/data/autogt/*/"), key=os.path.getmtime)[-1]
files = sorted(glob.glob(store + "*.npz"))
step = max(1, len(files) // 300)
sel = files[::step][:300]
print("store", store, "files", len(files), "sampled", len(sel))
tot = obs = known = stat = unk_obs = unk_obs_moving_flag = 0
groups = {}
for f in sel:
    z = np.load(f, allow_pickle=True)
    if "meta" not in z.files:
        print("no meta key; keys:", z.files[:10])
        sys.exit(1)
    meta = z["meta"].item() if z["meta"].shape == () else z["meta"]
    for b in meta.get("boxes", []):
        tot += 1
        o = bool(b.get("observed", True))
        k = bool(b.get("vel_known", True))
        s = bool(b.get("stationary", False))
        obs += o
        if o and not s and not k:
            unk_obs += 1
            groups[b.get("group")] = groups.get(b.get("group"), 0) + 1
        stat += s
        known += k
print(f"boxes {tot}, observed {obs}, stationary {stat}, vel_known {known}")
print(f"observed, non-stationary, velocity UNKNOWN: {unk_obs} = {100 * unk_obs / max(1, obs):.1f} % of the observed boxes; by group {groups}")
