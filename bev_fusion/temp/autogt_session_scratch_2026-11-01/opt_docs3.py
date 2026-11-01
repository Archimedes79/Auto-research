import os
os.chdir("/mnt/c/Auto-research")
R = lambda p: open(p, encoding="utf-8").read()
W = lambda p, s: open(p, "w", encoding="utf-8", newline="\n").write(s)
r = R("autogt/README.md")
old = "python -m autogt sets                                       # the scene sets small / full and how much of each is stored; `versions` lists the GT versions"
assert old in r
r = r.replace(old, old + "\npython -m autogt have [--set small|full | --scenes .. | --scenes-file F] [--stride N]   # is the stored GT of the current generator there for these sequences? exit 0 / 1 (the check before a training); `generate --scenes-file F` fills what is missing", 1)
W("autogt/README.md", r)
a = R("autogt/AGENTS.md")
old = "| a small or a full run |"
assert old in a
a = a.replace(old, "| to know before a training whether the GT of its sequences is stored | `python -m autogt have --scenes-file names.txt` (or `--set small|full`, `--scenes ...`, `--stride N`): exit 0 when every keyframe has a file for the current generator, else 1 and the missing count; `python -m autogt generate '~/data/nuscenes_full' --scenes-file names.txt` fills the rest (resumable); in code `autogt.sets.stored_for(names)` gives the missing sample tokens |\n" + old, 1)
W("autogt/AGENTS.md", a)
print("docs ok")
