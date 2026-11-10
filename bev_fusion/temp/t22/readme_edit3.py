"""scratch: third round of my tests/README.md row (autocast case, corrected planted-bug count): applied to the working file and to a copy of the CURRENT HEAD file (temp/t22/blob_tests_README.md). Run from the repository root."""
import subprocess

path = "bev_fusion/tests/README.md"
old = "15 more planted bugs killed (scratch `temp/t22/mut_K2.py`); the test removes"
new = ("'all' under autocast (bf16 on the CPU: the current-frame student call has to come before the older-frame no_grad call, the swapped order is caught); 16 more planted bugs killed "
       "(scratch `temp/t22/mut_K2.py`, `mut_K5.py`, `mut_K4.sh`; the camera-control and marker-key mutants were re-run after a first run died of a missing package in the private copy); the test removes")
work = open(path, encoding="utf-8", newline="").read()
head = subprocess.run(["git", "show", "HEAD:" + path], capture_output=True, check=True).stdout.decode("utf-8")
for label, text in (("working", work), ("HEAD", head)):
    assert text.count(old) == 1, (label, text.count(old))
open(path, "w", encoding="utf-8", newline="").write(work.replace(old, new))
open("bev_fusion/temp/t22/blob_tests_README.md", "w", encoding="utf-8", newline="").write(head.replace(old, new))
print("edited", path, "and the HEAD-based copy")
