"""rebuilds temp/radar/tests_readme_mine.md = the CURRENT HEAD tests/README.md + the test_radar_memory_ghost row (the working copy already has it)."""
import re
import subprocess

src = open("temp/radar/readme_ghost.py", encoding="utf-8").read()
row = re.search(r'row = \((.*?)\)\nanchor', src, re.S).group(1)
row = eval("(" + row + ")")
anchor = "| `test_object_head.py`"
head = subprocess.run(["git", "-C", "..", "show", "HEAD:bev_fusion/tests/README.md"], capture_output=True, check=True).stdout.decode("utf-8")
assert head.count(anchor) == 1 and "test_radar_memory_ghost" not in head
open("temp/radar/tests_readme_mine.md", "w", encoding="utf-8", newline="").write(head.replace(anchor, row + anchor, 1))
print("rebuilt from HEAD", subprocess.run(["git", "-C", "..", "rev-parse", "--short", "HEAD"], capture_output=True).stdout.decode().strip())
