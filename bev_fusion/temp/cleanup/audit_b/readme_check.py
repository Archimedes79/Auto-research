"""Read-only: for a folder's README, (a) tracked files of the folder (non-recursive and recursive) not named in the README, (b) file names the README
names in backticks that exist nowhere under the folder (by basename search over git ls-files of the whole bev_fusion tree plus autoresearch).
Usage: python readme_check.py <folder relative to repo root> ..."""
import os, re, subprocess, sys

REPO = "/mnt/c/Auto-research"
allfiles = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout.split("\n")
bases = {}
for f in allfiles:
    bases.setdefault(os.path.basename(f), []).append(f)
for folder in sys.argv[1:]:
    rd = os.path.join(REPO, folder, "README.md")
    txt = open(rd, encoding="utf-8").read()
    tracked = [f for f in allfiles if f.startswith(folder + "/") and f != folder + "/README.md"]
    print(f"=== {folder}/README.md ({len(txt)} chars), {len(tracked)} tracked files below")
    for f in tracked:
        rel = f[len(folder) + 1:]
        b = os.path.basename(f)
        if rel in txt or b in txt:
            continue
        if b in ("__init__.py", "README.md", ".gitkeep"):
            continue
        # directories mentioned?
        top = rel.split("/")[0]
        if "/" in rel and (top + "/") in txt:
            continue
        print(f"  not named: {rel}")
    names = set(re.findall(r"`([A-Za-z0-9_./-]+\.(?:py|sh|json|md|csv|txt|ps1))`", txt))
    for nm in sorted(names):
        b = os.path.basename(nm)
        full = nm if nm in allfiles else None
        if nm in allfiles or (folder + "/" + nm) in allfiles or any(f.endswith("/" + nm) for f in allfiles):
            continue
        if b in bases:
            continue
        print(f"  names a file that is not tracked anywhere: {nm}")
