"""scratch (Image Encoder, clean-base round): README edits applied to BOTH the working file and a copy of its HEAD version (temp/cleanup/headcopy/<path>), which becomes the committed blob
(HEAD + my hunk only; the working files of tests/README.md, training/README.md and training/configs/README.md carry other chats' uncommitted hunks).
Each edit is (path, regex, replacement); a regex must match exactly once in each of the two texts.
usage: python3 temp/cleanup/edit_readmes.py EDITS.py   (EDITS.py defines EDITS = [(path, pattern, replacement), ...] relative to the repository root)"""
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
HEAD = os.path.join(ROOT, "bev_fusion", "temp", "cleanup", "headcopy")
ns = {}
exec(open(sys.argv[1], encoding="utf-8").read(), ns)
for path, pat, rep in ns["EDITS"]:
    work = os.path.join(ROOT, path)
    hc = os.path.join(HEAD, path)
    if not os.path.exists(hc):
        os.makedirs(os.path.dirname(hc), exist_ok=True)
        open(hc, "wb").write(subprocess.run(["git", "show", "HEAD:" + path], cwd=ROOT, capture_output=True, check=True).stdout)
    for p in ((hc,) if "--head-only" in sys.argv else (work, hc)):
        raw = open(p, "rb").read().decode("utf-8")
        crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n")
        found = re.findall(pat, text, flags=re.M | re.S)
        assert len(found) == 1, f"{path} ({'working' if p == work else 'HEAD copy'}): pattern matched {len(found)} times: {pat[:70]}"
        new = re.sub(pat, lambda m: m.expand(rep), text, count=1, flags=re.M | re.S)       # expand: \1 is the group, not a literal (the critic found the literal in the first run)
        if crlf:
            new = new.replace("\n", "\r\n")
        open(p, "wb").write(new.encode("utf-8"))
    print("edited", path)
