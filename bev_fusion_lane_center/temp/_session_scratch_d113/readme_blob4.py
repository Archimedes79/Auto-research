import io
import subprocess

JOBS = {
    "evaluation/README.md": (["| `jmap_eval.py` |"], "| `jpast_eval.py` |"),
    "training/README.md": (["| `data/lane_map_data.py` |", "| `train/jmap_train.py` |"], "| `train/jpast_train.py` |"),
    "training/grid/README.md": (["| `jmap.py` |", "| `test_jmap.py` |", "| `test_lane_map.py` |", "| `test_jmap_train.py` |"], "| `test_jpast_train.py` |"),
}
for p, (prefixes, after) in JOBS.items():
    cur = io.open(p, encoding="utf-8").read().split("\n")
    head = subprocess.run(["git", "show", "HEAD:bev_fusion_lane_center/" + p], capture_output=True, text=True, encoding="utf-8").stdout.split("\n")
    mine = []
    for pre in prefixes:
        m = [l for l in cur if l.startswith(pre)]
        assert len(m) == 1, (p, pre, len(m))
        mine.append(m[0])
    i = [k for k, l in enumerate(head) if l.startswith(after)]
    assert len(i) == 1, (p, after, len(i))
    out = head[:i[0] + 1] + mine + head[i[0] + 1:]
    dst = "temp/_session_scratch_d113/" + p.replace("/", "_") + ".mine4"
    io.open(dst, "w", encoding="utf-8", newline="\n").write("\n".join(out))
    print(p, "->", dst, len(mine), "rows")
