import json, os, glob
from collections import Counter
for f in sorted(glob.glob(os.path.expanduser("~/data/nuscenes_full/maps/expansion/*.json"))):
    m = json.load(open(f)); print(os.path.basename(f), Counter(r.get("stop_line_type") for r in m["stop_line"]), len(m["ped_crossing"]))
