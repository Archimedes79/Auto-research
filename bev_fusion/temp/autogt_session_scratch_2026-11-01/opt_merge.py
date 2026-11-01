"""scratch: merge autogt/encode.py into video.py and autogt/rawdata.py into sets.py (fewer small files), keeping every function"""
import os
import re

os.chdir("/mnt/c/Auto-research")
R = lambda p: open(p, encoding="utf-8").read()


def W(p, s):
    open(p, "w", encoding="utf-8", newline="\n").write(s)


# ---- encode.py -> video.py
enc = R("autogt/encode.py")
body = enc[enc.index("CANDIDATES = "):]
vid = R("autogt/video.py")
vid = vid.replace("H.264 (autogt/encode.py; read back after encoding), 12 fps.", "H.264 (`encode` below: ffmpeg with libx264, read back after encoding), 12 fps.")
vid = vid.replace("import math\nimport os\nimport tempfile\n", "import math\nimport os\nimport shutil\nimport subprocess\nimport tempfile\n", 1)
vid = vid.replace("from .encode import encode\n", "", 1)
head_doc = enc[:enc.index('"""', 3) + 3]
section = ("# ---- frames -> a video that plays everywhere (H.264, yuv420p, mp4 with the index up front), checked by reading it back\n"
           "# cv2.VideoWriter('mp4v') makes MPEG-4 part 2, which browsers, the Claude app and most players refuse (2026-10-30: the first GT videos\n"
           "# did not play). H.264 needs an ffmpeg with libx264: looked up in $FFMPEG, the PATH and the ffmpeg.exe builds installed on the Windows\n"
           "# side (called through WSL interop). Without one an animated GIF is written and the result says so. The file is decoded again after\n"
           "# encoding (frame count, first frame not black); a video that does not play is never reported as done. (Same method as\n"
           "# autoresearch/evalsuite/render/video.py, kept here so that autogt stays standalone.)\n")
anchor = "COL = {"
assert anchor in vid
vid = vid.replace(anchor, section + body.rstrip("\n") + "\n\n\n" + anchor, 1)
W("autogt/video.py", vid)
os.remove("autogt/encode.py")

# ---- rawdata.py -> sets.py
raw = R("autogt/rawdata.py")
fn = raw[raw.index("def check("):]
fn = fn.replace("def check(dataroot, version=None, set_name=None, scenes=None, examples=10):\n    from . import sets\n    from .cli import detect_version\n",
                "def check_raw(dataroot, version=None, set_name=None, scenes=None, examples=10):\n    from .cli import detect_version\n    from .light_nusc import _KEEP, _records, _value\n    import json\n", 1)
fn = fn.replace("sets.scenes(set_name)", "scenes_of(set_name)").replace("def report(r):", "def report_raw(r):")
assert "scenes_of" in fn
st = R("autogt/sets.py")
st = st.replace("def scenes(name):", "def scenes(name):", 1)
st = st.replace('''    sets.status(name, ...)        {"set", "version", "have", "total", "complete"} for the current generator signature
"""''', '''    sets.status(name, ...)        {"set", "version", "have", "total", "complete"} for the current generator signature
    sets.check_raw(dataroot, ...) are the raw files the generator reads on disk (tables, map expansion, CAM_FRONT, the five radars)?
                                  `python -m autogt data`; sets.report_raw(result) prints it. The tools that fetch, verify and link the raw
                                  data are `datasets/NuScenes/tools/` (its README describes the layout on D:)
"""''', 1)
st = st.rstrip("\n") + "\n\n\n# ---- the raw files a run needs\n" + (
    "# autogt reads the json tables, the map expansion of the cities of the sequences, and per keyframe and radar sweep the files of CAM_FRONT\n"
    "# and the five radars (the image is needed to look at the GT, the radar files for the velocity correction). One pass over the tables (no\n"
    "# devkit) and one listing per sensor folder; a file counts as present by its name, its content is not read.\n\n\n") + fn.replace("scenes_of(set_name)", "scenes(set_name)")
W("autogt/sets.py", st)
os.remove("autogt/rawdata.py")

# ---- users
c = R("autogt/cli.py")
c = c.replace("    from . import rawdata\n    r = rawdata.check(a.dataroot, a.version, a.set_name, a.scenes)\n    print(rawdata.report(r))",
              "    from . import sets\n    r = sets.check_raw(a.dataroot, a.version, a.set_name, a.scenes)\n    print(sets.report_raw(r))", 1)
W("autogt/cli.py", c)
t = R("autogt/selftest.py")
t = t.replace("    from . import rawdata\n    ok_ = rawdata.check(", "    ok_ = sets.check_raw(", 1).replace("bad_ = rawdata.check(", "bad_ = sets.check_raw(", 1)
t = t.replace('assert "missing" in rawdata.report(bad_) and "all on disk" in rawdata.report(ok_)', 'assert "missing" in sets.report_raw(bad_) and "all on disk" in sets.report_raw(ok_)', 1)
W("autogt/selftest.py", t)
v = R("autogt/versions.py")
v = v.replace('"encode.py", ', "").replace('"rawdata.py", ', "")
W("autogt/versions.py", v)
print("merged; rawdata mentions left in selftest:", t.count("rawdata"), "| encode mentions in video:", vid.count("encode"))
