"""scratch: for each candidate deletion group, which OTHER tracked files name it (path prefix or file name), outside the group itself
    wsl python3 bev_fusion/temp/t16/pointers.py > bev_fusion/temp/t16/pointers.txt   (from the repository root)"""
import os
import subprocess

H = "bev_fusion_lane_center/experiment_execution/history/"
GROUPS = {
    "lane history/launchers": [H + "launchers/"],
    "lane history/migration_tools": [H + "migration_tools/"],
    "lane history/literature_sources": [H + "literature_sources/"],
    "lane history/indexes": [H + "indexes/"],
    "lane history/top-level files": [H + "PATH_MAP.md", H + "README_2026-10-13_loop1.md", H + "SYSTEM_2026-10-25.md", H + "schedule_loops19-22_2026-11-02.csv"],
    "lane history/old prompts+plans": [H + "prompts/Loop16_Prompt_EN.md", H + "prompts/Loop_Prompt_Next_EN.md", H + "plans/"],
    "autoresearch/history": ["autoresearch/history/"],
    "autoresearch/fewer_files.md": ["autoresearch/fewer_files.md"],
    "autoresearch/architecture.*": ["autoresearch/architecture.json", "autoresearch/architecture.svg", "autoresearch/architecture.md"],
}
files = [f for f in subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split("\n") if f]
texts = {}
for f in files:
    if f.endswith((".md", ".py", ".sh", ".json", ".csv", ".ps1", ".txt", ".yml")) and os.path.exists(f) and os.path.getsize(f) < 2_000_000:
        try:
            texts[f] = open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            pass
for name, prefixes in GROUPS.items():
    members = [f for f in files if any(f == p or f.startswith(p) for p in prefixes)]
    bases = set()
    needles = []
    for p in prefixes:
        needles.append(p)
        if not p.endswith("/"):
            needles.append(os.path.basename(p))
    # sub-folder names under a folder prefix are also pointers ('history/indexes', 'literature_sources')
    for p in prefixes:
        if p.endswith("/"):
            needles.append(p.rstrip("/").split("/")[-1] + "/")
    hits = {}
    for f, t in texts.items():
        if f in members:
            continue
        for n in needles:
            if n and n in t:
                hits.setdefault(f, set()).add(n)
    print(f"== {name}: {len(members)} tracked files; named by {len(hits)} other files")
    for f, ns in sorted(hits.items())[:14]:
        print(f"   {f}  <- {sorted(ns)[:2]}")
    if len(hits) > 14:
        print(f"   ... {len(hits) - 14} more")
