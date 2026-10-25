"""Group temp/analysis and temp/launchers by loop so that temp/ can be cleared loop by loop.   Run: _run.sh temp_group.py [--go]"""
import os
import re
import shutil
import sys
import time
from collections import Counter

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
GO = "--go" in sys.argv


def loop_of(name):
    m = re.match(r"^_?(?:fig_)?p(\d+)_", name) or re.match(r"^_run_p(\d+)", name) or re.match(r"^_run_p(\d+)", name.replace("_p", "_p"))
    if m:
        n = int(m.group(1))
        return f"loop{n:02d}"
    if re.match(r"^_run_(e13|e14|final17|master3|build17)", name) or re.search(r"17", name.split(".")[0]) and name.startswith(("_run", "_cmp")):
        return "loop17"
    if re.match(r"^(_run_(r3|resample|timing|zero_tok|insample|test16b|e122|test_eval)|_smoke|_reeval|_table_r|_r3tab|_q_|_final16b|_state_loop16|_phase16b|_forks|_fork_|_route_|_recut|_regen_rung|_tok_sub|_zero_tok|_make_chain16)", name):
        return "loop16"
    return "misc"


cnt = Counter()
for sub in ("analysis", "launchers"):
    d = f"{R}/temp/{sub}"
    for f in sorted(os.listdir(d)):
        p = f"{d}/{f}"
        if not os.path.isfile(p):
            continue
        g = loop_of(f)
        cnt[(sub, g)] += 1
        if GO:
            os.makedirs(f"{d}/{g}", exist_ok=True)
            shutil.move(p, f"{d}/{g}/{f}")
            open("/mnt/c/ML-Experiment/restructure/MIGRATION_LOG.csv", "a").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},tempgroup,{p},{d}/{g}/{f}\n")
print(dict(cnt))
