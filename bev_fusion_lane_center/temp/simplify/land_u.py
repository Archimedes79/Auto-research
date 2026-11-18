"""Landing scratch for D117 units U1-U5 (+ the critic's should-fixes): from the worktree C:/Auto-research_wt_simplify (base 3f33a319) into the live tree.
usage (from the problem folder, in WSL): python3 temp/simplify/land_u.py [--apply]
Checks, then (with --apply) copies the 7 code files into the live tree and writes README blobs (HEAD + only my changed rows) to temp/simplify/blob/;
prints the commit_paths command. Refuses when: the lane chat's landing is not in HEAD (jtiming.timed missing / jpast.py still tracked), a job imports
the token code, a code file of mine changed on main since the base or in the live tree, or a README row of mine was changed by someone else (CONFLICT)."""
import difflib
import os
import subprocess
import sys

BASE = "3f33a319"
PROB = "bev_fusion_lane_center"
WT = "/mnt/c/Auto-research_wt_simplify/" + PROB
CODE = ["training/grid/jtoken.py", "training/train/jtoken_train.py", "training/train/jlive_train.py", "evaluation/jtoken_eval.py",
        "training/grid/test_jtoken.py", "training/grid/test_jlive_train.py", "evaluation/tests/test_jtoken_eval.py"]
READMES = ["evaluation/README.md", "training/README.md", "training/grid/README.md", "runs/README.md"]
apply = "--apply" in sys.argv


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd="..").stdout


def show(rev, p):
    r = subprocess.run(["git", "show", f"{rev}:{PROB}/{p}"], capture_output=True, text=True, cwd="..")
    return r.stdout if r.returncode == 0 else None


bad = []
# (1) the lane chat's landing is in
if "def timed" not in (show("HEAD", "evaluation/jtiming.py") or ""):
    bad.append("HEAD's evaluation/jtiming.py has no `def timed` (the lane chat's landing is not in)")
if show("HEAD", "training/grid/jpast.py") is not None:
    bad.append("training/grid/jpast.py is still tracked in HEAD (the lane chat's landing is not in)")
# (2) no job imports the token code
ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
busy = [l for l in ps if any(k in l for k in ("jtoken_train.py", "jlive_train.py", "jtoken_eval.py", "jarm.py", "jtoken_fast", "bench_token_step")) and "land_u.py" not in l]
if busy:
    bad.append("jobs use the token code: " + " | ".join(b[:120] for b in busy))
# (3) code files: main and the live tree unchanged since the base
for p in CODE:
    b, h = show(BASE, p), show("HEAD", p)
    live = open(p, encoding="utf-8").read()
    if b != h:
        bad.append(f"{p}: changed on main since {BASE}")
    if live != h:
        bad.append(f"{p}: the live working file differs from HEAD")
# (4) README blobs: my changed rows on top of HEAD
blobs = {}
for p in READMES:
    b, h, w = show(BASE, p).splitlines(True), show("HEAD", p).splitlines(True), open(os.path.join(WT, p), encoding="utf-8").read().splitlines(True)
    out = list(h)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, b, w, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag in ("replace", "delete"):
            for k, old in enumerate(b[i1:i2]):
                if old not in out:
                    bad.append(f"CONFLICT {p}: my base row changed on main: {old[:100]!r}")
                    continue
                idx = out.index(old)
                new = w[j1 + k] if tag == "replace" and j1 + k < j2 else None
                out[idx:idx + 1] = [new] if new is not None else []
            if tag == "replace" and j2 - j1 > i2 - i1:
                anchor = b[i2 - 1]
                if anchor in out:
                    pos = out.index(w[j1 + (i2 - i1) - 1]) + 1
                    out[pos:pos] = w[j1 + (i2 - i1):j2]
                else:
                    bad.append(f"CONFLICT {p}: no anchor for added rows")
        elif tag == "insert":
            anchor = b[i1 - 1]
            if anchor not in out:
                bad.append(f"CONFLICT {p}: the anchor row of my added row changed on main: {anchor[:100]!r}")
                continue
            pos = out.index(anchor) + 1
            out[pos:pos] = w[j1:j2]
    blobs[p] = "".join(out)
    print(f"{p}: HEAD {len(h)} lines -> blob {len(out)} lines")

if bad:
    print("REFUSED:\n  " + "\n  ".join(bad))
    sys.exit(1)
os.makedirs("temp/simplify/blob", exist_ok=True)
args = []
for p, txt in blobs.items():
    dst = "temp/simplify/blob/" + p.replace("/", "__")
    open(dst, "w", encoding="utf-8").write(txt)
    args.append(f"--blob {p}={dst}")
if apply:
    for p in CODE:
        open(p, "w", encoding="utf-8", newline="").write(open(os.path.join(WT, p), encoding="utf-8", newline="").read())
    print("copied the 7 code files into the live tree")
print("commit: python3 ../autoresearch/tools/commit_paths.py -F <msg> " + " ".join(CODE) + " " + " ".join(args))
