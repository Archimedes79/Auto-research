"""Scratch: add the t13 row to experiments/README.md in the working tree and in a HEAD-based blob (temporary-index commit).
Usage: python3 readme_row.py <repo root>  -> temp/t13/blob_experiments_README.md"""
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
here = Path(__file__).resolve().parent
rel = "bev_fusion/experiments/README.md"
ROW = ("| `t13_complete_network/` | does adding the blocks that passed their own rules to the incumbent chain (`t10_dual` + the joint motion block of t15) help? "
       "a ladder, one comparison per rung; first candidate rung: the radar slot `delta_slots` (arm MR against the t15 reference, 2 seeds, 3000 steps, GT v5 pinned); "
       "numbers, guards and margins from the suite; the long run is a successor card | DRAFT revision 2 (2026-11-01 night): critic pass 1 (REVISE) answered, "
       "inputs missing (t15 result, r2 verdict, a target of the radar slot above its margin); nothing run |\n")
ANCHOR = "| `t12_motion_block/` |"
cur = (root / rel).read_bytes().decode("utf-8")
head = subprocess.run(["git", "show", "HEAD:" + rel], cwd=root, capture_output=True, check=True).stdout.decode("utf-8")
for name, text in (("working tree", cur), ("HEAD", head)):
    assert text.count(ANCHOR) == 1, (name, text.count(ANCHOR))
    assert "t13_complete_network" not in text, name
cur = cur.replace(ANCHOR, ROW + ANCHOR)
head = head.replace(ANCHOR, ROW + ANCHOR)
(root / rel).write_bytes(cur.encode("utf-8"))
(here / "blob_experiments_README.md").write_bytes(head.encode("utf-8"))
print("ok")
