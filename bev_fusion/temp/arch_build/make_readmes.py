"""one-off: HEAD version of a README with only my rows replaced by / added from the working-tree rows (the other hunks of the working file
belong to other chats). A row in ROWS is taken from the working file whole (HEAD's row must exist, or it is added before `before`).
usage: make_readmes.py <readme> <out>"""
import subprocess
import sys

readme, out = sys.argv[1], sys.argv[2]
ROWS = {
    "bev_fusion/training/README.md": [("| `precompute.py` |", None), ("| `nuscenes_dataset.py` |", None)],
    "bev_fusion/tests/README.md": [("| `test_slots.py` |", None), ("| `test_bundle.py` |", "| `test_velocity_hurdle.py` |"), ("| `test_runner.py` |", "| `test_velocity_hurdle.py` |")],
    "bev_fusion/README.md": [("| `runner/` |", "| `tests/` |")],
}[readme]
head = subprocess.run(["git", "show", f"HEAD:{readme}"], capture_output=True, text=True, encoding="utf-8", check=True).stdout.split("\n")
work = open(readme, encoding="utf-8").read().split("\n")
for prefix, before in ROWS:
    wi = [l for l in work if l.startswith(prefix)]
    assert len(wi) == 1, (prefix, len(wi))
    hi = [i for i, l in enumerate(head) if l.startswith(prefix)]
    if before is None:                                         # replace HEAD's row
        assert len(hi) == 1, (prefix, hi)
        head[hi[0]] = wi[0]
    else:                                                      # a new row before another
        assert not hi, (prefix, "already in HEAD")
        at = [i for i, l in enumerate(head) if l.startswith(before)]
        assert len(at) == 1, (before, at)
        head.insert(at[0], wi[0])
open(out, "w", encoding="utf-8", newline="\n").write("\n".join(head))
print("ok", readme, len(ROWS), "rows")
