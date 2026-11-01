import os
os.chdir("/mnt/c/Auto-research")
R = lambda p: open(p, encoding="utf-8").read()


def W(p, s):
    open(p, "w", encoding="utf-8", newline="\n").write(s)


r = R("autogt/README.md")
r = r.replace("five radars; `rawdata.py`) -> `generate`", "five radars; `sets.check_raw`) -> `generate`", 1)
r = r.replace("| `rawdata.py` | `check(dataroot, ...)`: are the raw files the generator reads on disk; `python -m autogt data` |\n", "", 1)
r = r.replace("| `sets.py`, `sets/small.txt`, `sets/full.txt` | the scene sets (one scene name per line): `names`, `scenes`, `tokens`, `status`, `complete_in`; `generate(set_name=...)`, `python -m autogt sets` |",
              "| `sets.py`, `sets/small.txt`, `sets/full.txt` | the scene sets (one scene name per line): `names`, `scenes`, `tokens`, `status`, `complete_in`; `generate(set_name=...)`, `python -m autogt sets`; and `check_raw(dataroot, ...)`: are the raw files the generator reads on disk, `python -m autogt data` |", 1)
r = r.replace("| `checks.py`, `view.py`, `video.py`, `report.py`, `clean.py`, `encode.py`, `selftest.py` | the tools behind the commands (`encode.py`: H.264 video with a read-back check; mp4v does not play in browsers or the Claude app) |",
              "| `checks.py`, `view.py`, `video.py`, `report.py`, `clean.py`, `selftest.py` | the tools behind the commands (`video.py` also holds `encode`: H.264 video with a read-back check; mp4v does not play in browsers or the Claude app) |", 1)
W("autogt/README.md", r)
a = R("autogt/AGENTS.md")
a = a.replace("`checks.py view.py video.py report.py cli.py suite.py clean.py encode.py selftest.py sets.py` do not change it.", "`checks.py view.py video.py report.py cli.py suite.py clean.py selftest.py sets.py` do not change it.", 1)
a = a.replace("`sets.py` + `sets/` (scene sets), `rawdata.py` (raw-data check),", "`sets.py` + `sets/` (scene sets, and the raw-data check), ", 1)
a = a.replace("`video.py`, `encode.py` (H.264 with a read-back check),", "`video.py` (with `encode`: H.264 with a read-back check),", 1)
W("autogt/AGENTS.md", a)
print("rawdata" in r + a, "encode.py" in r + a)
