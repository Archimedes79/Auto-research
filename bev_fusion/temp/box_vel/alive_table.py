"""t11 vel cause: which runs have a live velocity output? Joins runs/<run>/config.json (effective batch, steps, velocity weight, width) with the
stored-detection scan (temp/box_vel/out/stored_dets.txt: max predicted vehicle speed at dt 0). Alive = the detections reach > 1 m/s.
Run (from bev_fusion/): python3 temp/box_vel/alive_table.py
"""
import json
import os
import re

rows = {}
for line in open("temp/box_vel/out/stored_dets.txt"):
    m = re.match(r"(\S+?)/eval/scene_records\.pkl\s+0\.0\s+(\d+)\s+([\d.]+)\s+(\d+)", line)
    if m:
        rows[m.group(1)] = (int(m.group(2)), float(m.group(3)), int(m.group(4)))
for line in open("temp/box_vel/out/stored_dets_t8ctl.txt"):
    m = re.match(r"(\S+?)/eval/scene_records_n1000\.pkl\s+0\.0\s+(\d+)\s+([\d.]+)\s+(\d+)", line)
    if m:
        rows[m.group(1)] = (int(m.group(2)), float(m.group(3)), int(m.group(4)))
print(f"{'run':20s} {'batch x accum x seq':>20s} {'steps':>6s} {'w_vel':>5s} {'width':>5s} {'max |v| m/s':>12s}  verdict")
for run in sorted(rows):
    p = f"runs/{run}/config.json"
    if not os.path.exists(p):
        continue
    c = json.load(open(p))
    n, mx, nd = rows[run]
    seq = c.get("seq_len", 1)
    bat = c.get("batch_size", 1) * c.get("accum", 1)
    wv = c.get("loss", {}).get("w_obj_vel", 0.2)
    width = c.get("model", {}).get("trunk_width", "-")
    verdict = "alive" if mx > 1.0 else ("tiny" if mx > 0.1 else "dead")
    print(f"{run:20s} {c.get('batch_size', 1):>6d} x {c.get('accum', 1)} x {seq:<3d}      {c.get('steps', 0):6d} {wv:5.1f} {width!s:>5s} {mx:12.3f}  {verdict}")
