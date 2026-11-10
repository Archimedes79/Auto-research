"""Scratch: put the row I committed as a blob into the WORKING copy too (other chats have uncommitted hunks in these files; whoever commits the working copy must not drop my row)."""
import os

REPO = "/mnt/c/Auto-research"
BLOBS = os.path.join(REPO, "bev_fusion/temp/cleanup/blobs")
JOBS = {"bev_fusion/evaluation/README.md": "| `bev_eval.py` + `evalsuite.json` |", "bev_fusion/tests/README.md": "| `test_evaluation_api.py` |"}
for path, anchor in JOBS.items():
    blob = open(os.path.join(BLOBS, path.replace("/", "_")), encoding="utf-8", newline="").read().split("\n")
    row = blob[next(k for k, l in enumerate(blob) if l.startswith(anchor)) + 1]
    full = os.path.join(REPO, path)
    text = open(full, encoding="utf-8", newline="").read()
    if row in text:
        print(path, "has the row already")
        continue
    lines = text.split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith(anchor))
    lines.insert(i + 1, row)
    with open(full, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines))
    print(path, "row inserted after line", i + 1)
