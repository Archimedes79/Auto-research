p = "splits/README.md"
s = open(p, encoding="utf-8").read()
old = "**Use.** `python training/splits.py names small`"
assert s.count(old) == 1
new = ("**The evaluation reads these files too.** A set of the evaluation is `<split>.<part>` (`small.val`, `large.train`, ...) = a file of this folder and its `train` or `val` list "
       "(`python evaluation/bev_sets.py` lists them; `bev_suite.py RUN --split small.val ...`, `evaluation/README.md` \"The sets\"). It has no list of its own: a new set for the evaluation is a new file here.\n\n") + old
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("ok")
