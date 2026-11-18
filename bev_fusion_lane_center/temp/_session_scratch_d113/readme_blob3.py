import io
import subprocess

JOBS = {
    "evaluation/README.md": ["| `jpast_eval.py` |"],
    "training/grid/README.md": ["| `jpast.py` |", "| `test_jpast.py` |", "| `test_lane_past.py` |", "| `test_jpast_train.py` |"],
}
for p, prefixes in JOBS.items():
    cur = io.open(p, encoding="utf-8").read().split("\n")
    head = subprocess.run(["git", "show", "HEAD:bev_fusion_lane_center/" + p], capture_output=True, text=True, encoding="utf-8").stdout.split("\n")
    for pre in prefixes:
        mine = [l for l in cur if l.startswith(pre)]
        idx = [k for k, l in enumerate(head) if l.startswith(pre)]
        assert len(mine) == 1 and len(idx) == 1, (p, pre, len(mine), len(idx))
        head[idx[0]] = mine[0]
    out = "temp/_session_scratch_d113/" + p.replace("/", "_") + ".mine3"
    io.open(out, "w", encoding="utf-8", newline="\n").write("\n".join(head))
    print(p, "->", out)
