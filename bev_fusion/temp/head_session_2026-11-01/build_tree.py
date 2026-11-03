"""Build the tree of my commit through a temporary index: HEAD + my own files + my own rows in the shared READMEs.
Prints the tree id; writes nothing to the real index or HEAD. Run from the repository root in the WSL: python3 <this> [--commit "message file"]

Whole files that are mine alone (new, or the working copy is HEAD + my hunks only - checked by the diff the caller reads) are taken
from the working tree; the shared READMEs get my rows by exact-string replacement on the HEAD blob (each anchor must match once)."""
import os
import subprocess
import sys
import tempfile

REPO = "/mnt/c/Auto-research"
B = "bev_fusion/"


def git(*a, env=None, inp=None, text=True):
    r = subprocess.run(["git"] + list(a), cwd=REPO, env=env, input=inp, capture_output=True, text=text)
    if r.returncode:
        raise SystemExit(f"git {' '.join(a)} failed: {r.stderr}")
    return r.stdout


def head(path):
    return git("show", "HEAD:" + path)


def work(path):
    return open(os.path.join(REPO, path), encoding="utf-8").read()


def line_of(path, marker):
    ls = [x for x in work(path).splitlines() if x.startswith(marker)]
    assert len(ls) == 1, (path, marker, len(ls))
    return ls[0]


def once(text, s, what):
    assert text.count(s) == 1, f"{what}: anchor found {text.count(s)} times in HEAD"


whole = [B + "tests/test_fusion_area_real.py", B + "tests/run_gates.sh", B + "experiments/t14_dontcare/results/fusion_area_real_gt.txt", B + "plan/head_design.md", B + "plan/plan.md", B + "plan/README.md", B + "training/losses.py", B + "training/startable.py",
         B + "training/configs/t14_dc_ctl_s0.json", B + "training/configs/t14_dc_fusion_s0.json", B + "experiments/t14_dontcare/experiment.md"]
files = {p: work(p) for p in whole}

# my rows in the shared READMEs: row text from the working copy, placed by an anchor of the HEAD blob
for path, marker, anchor in (
        (B + "experiments/README.md", "| `t14_dontcare/` |", "| `t11_vel_cause/` |"),
        (B + "training/configs/README.md", "| `t14_dc_ctl_s0.json`", "| `t11_vel_hurdle_acc2.json` |"),
        (B + "training/README.md", "| `startable.py` |", "| `scratch.py` |"),
        (B + "tests/README.md", "| `test_fusion_area_real.py` |", "| `test_slots.py` |")):
    h = head(path)
    once(h, anchor, path)
    files[path] = h.replace(anchor, line_of(path, marker) + "\n" + anchor, 1)
tp = B + "training/README.md"
old = "unobserved boxes (no lidar / radar point) ignored in every object target)"
once(files[tp], old, tp)
files[tp] = files[tp].replace(old, old[:-1] + "; `LossConfig.fusion_area_only`, default off: D38 (b), the fused weights x camera x radar, tested)", 1)

index = tempfile.mktemp(prefix="idx_", dir=os.path.join(REPO, B, "temp", "head_session_2026-11-01"))
env = dict(os.environ, GIT_INDEX_FILE=index)
git("read-tree", "HEAD", env=env)
for path, text in files.items():
    blob = git("hash-object", "-w", "--stdin", env=env, inp=text).strip()
    mode = git("ls-tree", "HEAD", path).split()[0] if git("ls-tree", "HEAD", path).strip() else "100644"
    git("update-index", "--add", "--cacheinfo", f"{mode},{blob},{path}", env=env)
tree = git("write-tree", env=env).strip()
os.remove(index)
print("tree", tree)
print(git("diff-tree", "-r", "--name-status", "HEAD", tree))
print(git("diff", "--stat", "HEAD", tree))
if "--commit" in sys.argv:
    msg = open(sys.argv[sys.argv.index("--commit") + 1], encoding="utf-8").read()
    old_head = git("rev-parse", "HEAD").strip()
    c = git("commit-tree", tree, "-p", old_head, inp=msg).strip()
    git("update-ref", "-m", "head design, fusion-area option, startable check", "HEAD", c, old_head)     # refuses if HEAD moved meanwhile
    git("reset", "-q", "HEAD", "--", *files.keys())
    print("commit", c)
