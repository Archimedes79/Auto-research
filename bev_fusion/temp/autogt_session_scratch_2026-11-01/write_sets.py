"""scratch: write autogt/sets/small.txt (the 10 sequences of nuScenes mini) and full.txt (the 850 sequences of trainval) from the nuScenes json tables.
Run from C:\\Auto-research in WSL:  wsl bash autogt/run_nice.sh bev_fusion/temp/autogt_session_scratch_2026-11-01/write_sets.py"""
import os
import sys

sys.path.insert(0, ".")
from autogt.suite import default_root, json_tables

os.makedirs("autogt/sets", exist_ok=True)
for name, version, head in (
        ("small", "v1.0-mini", ["small: the 10 sequences of nuScenes mini (404 keyframes; day, night, rain, Boston and Singapore):",
                                "small experiments; about 25 minutes to generate. One scene name per line, # starts a comment."]),
        ("full", "v1.0-trainval", ["full: all 850 sequences of nuScenes trainval (34,149 keyframes): releases; 5 - 8 hours to generate.",
                                   "One scene name per line, # starts a comment."])):
    scenes, _ = json_tables(default_root(version), version)
    names = sorted(s["name"] for s in scenes)
    with open(f"autogt/sets/{name}.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(f"# {h}\n" for h in head) + "\n".join(names) + "\n")
    print(name, version, len(names), names[:2], names[-1])
