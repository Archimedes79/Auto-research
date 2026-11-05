"""Commit whole files that are mine alone through a temporary index (HEAD + these files). usage (repo root, WSL):
python3 commit_one.py <message file> <path> [<path> ...]    - prints the diff stat first; update-ref refuses when HEAD moved meanwhile"""
import os
import subprocess
import sys
import tempfile

REPO = "/mnt/c/Auto-research"


def git(*a, env=None, inp=None):
    r = subprocess.run(["git"] + list(a), cwd=REPO, env=env, input=inp, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"git {' '.join(a)} failed: {r.stderr}")
    return r.stdout


msg, paths = open(sys.argv[1], encoding="utf-8").read(), sys.argv[2:]
index = tempfile.mktemp(prefix="idx_", dir=os.path.join(REPO, "bev_fusion", "temp", "head_session_2026-11-01"))
env = dict(os.environ, GIT_INDEX_FILE=index)
old_head = git("rev-parse", "HEAD").strip()
git("read-tree", "HEAD", env=env)
for p in paths:
    blob = git("hash-object", "-w", "--stdin", env=env, inp=open(os.path.join(REPO, p), encoding="utf-8").read()).strip()
    git("update-index", "--add", "--cacheinfo", f"100644,{blob},{p}", env=env)
tree = git("write-tree", env=env).strip()
os.remove(index)
print(git("diff", "--stat", "HEAD", tree))
c = git("commit-tree", tree, "-p", old_head, inp=msg).strip()
git("update-ref", "-m", "head design gaps", "HEAD", c, old_head)
git("reset", "-q", "HEAD", "--", *paths)
print("commit", c)
