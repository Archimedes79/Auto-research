import os
os.chdir("/mnt/c/Auto-research")
p = "autogt/cli.py"
s = open(p, encoding="utf-8").read()
old = "    Memory-light and resumable: the sequences are read in groups of GROUP_SCENES with the light devkit (the GT is identical to the"
assert old in s
s = s.replace(old, "    Memory-light and resumable: the sequences are read in groups of GROUP_SCENES (fewer with several workers: every worker takes a group\n    of its own, so the devkit of a group is loaded once) with the light devkit (the GT is identical to the", 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("doc ok")
