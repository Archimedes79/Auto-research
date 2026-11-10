"""scratch (Image Encoder, clean-base round): README rows against the files of a folder: every `name` in the first column of a table row must exist in the folder, every file of the folder must have a row.
usage: python3 temp/cleanup/readme_rows_check.py FOLDER [FOLDER ...]   (from bev_fusion/; the README of the folder is FOLDER/README.md)"""
import os
import re
import sys

for folder in sys.argv[1:]:
    text = open(os.path.join(folder, "README.md"), encoding="utf-8").read()
    named = set()
    for line in text.splitlines():
        m = re.match(r"\|\s*((?:`[^`]+`(?:\s*,\s*)?)+)\s*\|", line)
        if m:
            named.update(re.findall(r"`([^`]+)`", m.group(1)))
    files = {f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f != "README.md" and not f.endswith((".pyc", ".orig", ".rej"))}
    dirs = {f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and f != "__pycache__"}
    missing = sorted(n for n in named if n.split("/")[0].split(" ")[0] not in files | dirs and "." in n and "/" not in n and " " not in n)
    undocumented = sorted(f for f in files if f not in named and not any(f in n for n in named))
    print(f"{folder}: {len(named)} names in rows, {len(files)} files; row without file: {missing or '-'}; file without row: {undocumented or '-'}")
