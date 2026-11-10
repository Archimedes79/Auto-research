"""scratch: structure of runs/<run>/eval/by_n/metrics_n1000.json (top keys, and the keys that hold lane_divider / stop_line numbers)"""
import json
import sys

d = json.load(open(f"bev_fusion/runs/{sys.argv[1]}/eval/by_n/metrics_n1000.json"))
print(type(d).__name__, list(d)[:12] if isinstance(d, dict) else len(d))
flat = d.get("metrics", d) if isinstance(d, dict) else {}
ks = [k for k in flat if "lane_divider/all/all/iou@0" in k or "stop_line" in k][:12]
for k in ks:
    print(k, flat[k] if not isinstance(flat[k], (dict, list)) else "...")
print(len(flat))
