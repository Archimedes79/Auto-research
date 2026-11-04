"""Control of what goes into git and what gets pushed (rules: the root CLAUDE.md, section "Git: what goes in, when to commit, when to push").

    python autoresearch/tools/check_git.py                  # push check + pending changes + scratch that something points at (exit 1 on an error)
    python autoresearch/tools/check_git.py --paths A B ...  # before `git commit -- A B ...`: is every path (a folder: its files) fit for git? (exit 1 if not)
    python autoresearch/tools/check_git.py --used           # the full list of scratch files that living code or records point at
    python autoresearch/tools/check_git.py --selftest

What it checks
  1. push check: no commit that is not yet on the upstream branch (without one: on any remote branch) may ADD a scratch or forbidden file, in any
     of its commits (a file added and deleted again is still published by a push), or add / change a file over 1 MB
  2. pending changes: the staged files (and, with --paths, the named ones) must be fit for git; deletions are never an error. Scratch = a file inside a
     `temp/` or `_to_delete/` folder (any case; except its README.md): it is not committed by hand, a run's tag keeps what ran. Forbidden = data,
     checkpoints, caches, binaries, logs and anything over 1 MB. Untracked scratch is only counted: `git add -A` would sweep it in.
  3. scratch that is used: a file in a `temp/` folder named by a living file (code, launcher, run record, experiment card) or a folder of scratch put on
     `sys.path` by living code. Such a file is not temporary any more; it belongs in a code folder (training/, evaluation/, tools/ ...) with a test, or the
     card should carry the command instead. Listed, never an error: it is the work list for promoting them. "run or imported" means the referrer is code, a
     launcher or a run record (a docstring mention counts, so look at the list); folders and globs (`temp/t10/*.sh`) are not followed.
Folders named history/ are archives and are neither scanned for references nor reported. Merge commits are not inspected (their count is shown).
Exit codes: 0 fine, 1 an error found, 2 wrong use or nothing to check.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

MAX_BYTES = 1_000_000
SCRATCH_DIRS = ("temp", "_to_delete")
FORBIDDEN_EXT = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".pkl", ".pickle", ".h5", ".png", ".jpg", ".jpeg", ".svg", ".mp4", ".gif", ".pdf", ".pptx", ".docx", ".xlsx",
                 ".html", ".zip", ".tgz", ".gz", ".tar", ".7z", ".log", ".tmp", ".lock", ".done", ".bin", ".onnx", ".safetensors", ".parquet"}   # the "never" block of .gitignore, and more
ALLOWED_NAMES = {"architecture.svg"}                                                    # the generated architecture drawings are text
TEXT_EXT = {".md", ".py", ".sh", ".json", ".csv", ".txt", ".ps1", ".yml", ".yaml"}
# a path through a temp/ folder with a file extension, optionally after $VAR/ or ${VAR}/; not followed by more name (temp/x.pyc, temp/a.json.gz), a sentence dot is fine
REF = re.compile(r"(?<![\w.])((?:\$\{?\w+\}?/|(?:[\w.\-]+/)*)temp/(?:[\w.\-]+/)*[\w.\-]+\.(?:py|sh|jsonl|json|csv|md|txt|ps1|yaml|yml))(?![\w\-])(?!\.\w)")
SYSPATH = re.compile(r"sys\.path\.(?:insert|append)\(.*?[\"']((?:[\w.\-]+/)*temp(?:/[\w.\-]+)*)[\"']")


def git(top, *args, check=True, inp=None):
    r = subprocess.run(["git", "-C", top, "-c", "core.quotepath=off", *args], capture_output=True, input=inp, encoding=None if inp is None else "utf-8")
    out = r.stdout.decode("utf-8", errors="replace") if inp is None else r.stdout
    if check and r.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {(r.stderr.decode('utf-8', 'replace') if inp is None else r.stderr).strip()[:200]}")
    return out


def toplevel(start="."):
    top = subprocess.run(["git", "-C", start, "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
    if not top:
        raise SystemExit("not in a git repository")
    return top


def classify(path, size=None):
    """'scratch' | 'forbidden' | 'ok' for a repository-relative path (size in bytes when known)"""
    parts = path.replace("\\", "/").split("/")
    name = parts[-1]
    low = [p.lower() for p in parts]
    if any(p in SCRATCH_DIRS for p in low[:-1]) and not (name == "README.md" and low[-2] in SCRATCH_DIRS):
        return "scratch"
    if "__pycache__" in parts or (os.path.splitext(name)[1].lower() in FORBIDDEN_EXT and name not in ALLOWED_NAMES):
        return "forbidden"
    return "forbidden" if size is not None and size > MAX_BYTES else "ok"


def verdict(top, path, size=None):
    """classify() for a path in this repository: a file of a forbidden TYPE that the repository's .gitignore whitelists on purpose (`!/autoresearch/architecture.svg`)
    is fine, because git would take it without `-f`; the size limit and scratch folders have no exception"""
    kind = classify(path, size)
    if kind == "forbidden" and not (size or 0) > MAX_BYTES and "__pycache__" not in path:
        if subprocess.run(["git", "-C", top, "check-ignore", "--no-index", "-q", "--", path], capture_output=True).returncode == 1:      # 1 = not ignored
            return "ok"
    return kind


def names(top, *args):
    """a git listing command as a list of paths (-z: names with spaces or non-ASCII letters arrive intact; `-z` goes first, after a `--` it would be a path)"""
    return [x for x in git(top, args[0], "-z", *args[1:]).split("\0") if x]


def pushed_commits(top):
    """(rev-list arguments of the commits a push would send, text about the comparison base) or (None, text) when nothing can be compared"""
    up = git(top, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", check=False).strip()
    if up:
        ahead, behind = (int(x) for x in git(top, "rev-list", "--left-right", "--count", f"HEAD...{up}").split())
        return [f"{up}..HEAD"], f"{up}: {ahead} commit(s) ahead, {behind} behind" + (" (`git fetch` first for the true remote state)" if not ahead and not behind else "")
    if not git(top, "remote").strip():
        return None, "no remote: nothing to push"
    return ["HEAD", "--not", "--remotes"], "no upstream branch: every commit that is on no remote branch counts as unpushed"


def push_check(top):
    """(errors, info) for the commits a push would send: a scratch or forbidden file they ADD (also when a later commit deletes it again), a file over 1 MB
    they add or change"""
    rng, base = pushed_commits(top)
    if rng is None:
        return [], [base]
    info = [base]
    commits = git(top, "rev-list", "--no-merges", *rng).split()
    merges = len(git(top, "rev-list", "--merges", *rng).split())
    for c in git(top, "log", "--no-merges", "--format=%h %an %s", *rng).splitlines():
        info.append("  to push: " + c[:140])
    if merges:
        info.append(f"  {merges} merge commit(s) are not inspected")
    entries = []                                              # (status, blob, path, commit)
    for c in commits:
        tok = git(top, "diff-tree", "--root", "--no-commit-id", "-r", "-z", "--raw", "--no-abbrev", "--no-renames", "--diff-filter=AM", c).split("\0")
        entries += [(tok[i].split()[-1], tok[i].split()[3], tok[i + 1], c[:8]) for i in range(0, len(tok) - 1, 2)]
    sizes = {}
    if entries:
        for line in git(top, "cat-file", "--batch-check", inp="\n".join(sorted({e[1] for e in entries})) + "\n").splitlines():
            f = line.split()
            if len(f) == 3 and f[1] == "blob":
                sizes[f[0]] = int(f[2])
    errors = []
    for status, blob, path, c in entries:
        kind = verdict(top, path, sizes.get(blob)) if status == "A" else ("forbidden" if sizes.get(blob, 0) > MAX_BYTES else "ok")      # a change of an old scratch file is legacy
        if kind != "ok":
            why = f" ({sizes[blob] // 1000} kB)" if sizes.get(blob, 0) > MAX_BYTES else ""
            errors.append(f"a push would publish {kind} file {path}{why} (commit {c})")
    return errors, info


def pending(top):
    """(staged paths without deletions, untracked paths): what the index would commit now and what `git add -A` would add on top"""
    return names(top, "diff", "--cached", "--name-only", "--diff-filter=d"), names(top, "ls-files", "--others", "--exclude-standard")


def path_errors(top, paths):
    errors = []
    for p in paths:
        try:
            size = os.path.getsize(os.path.join(top, p))
        except OSError:
            continue                                                 # deleted: nothing to commit
        kind = verdict(top, p, size)
        if kind != "ok":
            why = "inside a temp/ or _to_delete/ folder (scratch is not committed; promote it to a code folder or put the command in the card)" if kind == "scratch" \
                else f"a data / binary / log type or over 1 MB ({size // 1000} kB)"
            errors.append(f"{kind}: {p} - {why}")
    return errors


def used_scratch(top):
    """({scratch file: {'tracked': bool, 'executed': [referrers that are code, launchers or run records], 'cited': [documents that name it]}},
    [(referrer, scratch folder put on sys.path)]) for the scratch that a living file (not scratch itself, not in a history/ folder) points at. A path is resolved
    from the referrer's folder, its project folder (the first folder with a temp/ below the top), then the top; $VAR/ prefixes are dropped."""
    tracked = set(names(top, "ls-files"))
    files = sorted(tracked | set(names(top, "ls-files", "--others", "--exclude-standard")))
    out, imports = {}, []
    for f in files:
        parts = f.split("/")
        if classify(f) != "ok" or "history" in parts or os.path.splitext(f)[1].lower() not in TEXT_EXT:
            continue
        try:
            with open(os.path.join(top, f), encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        proj = parts[0] if os.path.isdir(os.path.join(top, parts[0], "temp")) else ""
        bases = [os.path.dirname(f)] + ([proj] if proj else []) + [""]
        for m in set(REF.findall(text)):
            rel = re.sub(r"^\$\{?\w+\}?/", "", m)
            for base in bases:
                t = os.path.normpath(os.path.join(base, rel)).replace("\\", "/")
                if classify(t) == "scratch" and os.path.isfile(os.path.join(top, t)):
                    kind = "executed" if (f.endswith((".py", ".sh", ".ps1")) or os.path.basename(f) == "run.json") else "cited"
                    d = out.setdefault(t, {"tracked": t in tracked, "executed": [], "cited": []})
                    d[kind].append(f)
                    break
        if f.endswith(".py"):
            for m in set(SYSPATH.findall(text)):
                for base in bases:
                    t = os.path.normpath(os.path.join(base, m)).replace("\\", "/")
                    if classify(t + "/x") == "scratch" and os.path.isdir(os.path.join(top, t)):
                        imports.append((f, t))
                        break
    return out, imports


def report(top, show_all=False):
    errors, lines = [], []
    e, info = push_check(top)
    errors += e
    lines += ["push check:"] + ["  " + i for i in info] + ["  " + x for x in e]
    staged, untracked = pending(top)
    se = path_errors(top, staged)
    errors += se
    scr = [p for p in untracked if classify(p) == "scratch"]
    lines += ["", f"pending: {len(staged)} staged file(s), {len(untracked)} untracked ({len(scr)} of them scratch that `git add -A` would sweep in)"]
    lines += ["  " + x for x in se] + [f"  untracked scratch: {p}" for p in scr[:10]] + ([f"  ... and {len(scr) - 10} more"] if len(scr) > 10 else [])
    tracked_scratch = [p for p in names(top, "ls-files") if classify(p) == "scratch"]
    used, imports = used_scratch(top)
    ex = sorted((t for t, d in used.items() if d["executed"]), key=lambda t: -len(used[t]["executed"]))
    ci = sorted((t for t, d in used.items() if not d["executed"]), key=lambda t: -len(used[t]["cited"]))
    lines += ["", f"scratch in git: {len(tracked_scratch)} tracked file(s) in temp/ or _to_delete/ folders (legacy; new ones are refused above)",
              f"scratch that is used: {len(ex)} file(s) run or imported by living files, {len(ci)} only cited in documents, {len(set(t for _, t in imports))} scratch folder(s) put on sys.path"
              + ("" if show_all else "  (--used lists them)")]
    if show_all:
        for title, group in (("run or imported (the first to promote):", ex), ("only cited:", ci)):
            if group:
                lines.append("  " + title)
                for t in group:
                    d = used[t]
                    refs = d["executed"] + d["cited"]
                    lines.append(f"    {t} [{'tracked' if d['tracked'] else 'UNTRACKED'}] {len(refs)} ref(s), e.g. {', '.join(refs[:2])}")
        if imports:
            lines.append("  scratch folders on sys.path (their modules are imported by living code):")
            lines += [f"    {t}  <- {f}" for f, t in sorted(set(imports), key=lambda x: x[1])]
    return errors, "\n".join(lines)


def _selftest():
    assert classify("bev_fusion/temp/t8/x.sh") == "scratch" and classify("bev_fusion/temp/README.md") == "ok"
    assert classify("bev_fusion/temp/sub/README.md") == "scratch" and classify("x/_to_delete/a.py") == "scratch"
    assert classify("a\\temp\\x.py") == "scratch" and classify("a/Temp/y.py") == "scratch" and classify("A/TEMP/z.py") == "scratch"      # separators, case
    assert classify("a/templates/x.py") == "ok" and classify("a/mytemp/x.py") == "ok" and classify("a/temp.py") == "ok"                    # only a folder named temp
    assert classify("bev_fusion/runs/a/model.pt") == "forbidden" and classify("a/b.png") == "forbidden" and classify("a/architecture.svg") == "ok"
    assert classify("a/x.pkl") == "forbidden" and classify("a/w.safetensors") == "forbidden" and classify("a/f.lock") == "forbidden"
    assert classify("a/big.json", 2_000_000) == "forbidden" and classify("a/small.json", 10) == "ok" and classify("a/__pycache__/x.py") == "forbidden"
    assert REF.findall("see temp/x.py. And $REPO/temp/y.sh, ${BFLC}/temp/a/b.json; ../bev_fusion/temp/z.md") == ["temp/x.py", "$REPO/temp/y.sh", "${BFLC}/temp/a/b.json", "../bev_fusion/temp/z.md"]
    assert REF.findall("temp/x.pyc temp/data.json.gz temp/ok.jsonl") == ["temp/ok.jsonl"], REF.findall("temp/x.pyc temp/data.json.gz temp/ok.jsonl")
    with tempfile.TemporaryDirectory() as top:
        def sh(*a, repo=None):
            subprocess.run(["git", "-C", repo or top, "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True)

        def put(rel, text="x", binary=0):
            p = os.path.join(top, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(text.encode("utf-8") if not binary else b"0" * binary)
        sh("init", "-q", "-b", "main")
        put(".gitignore", "*.ign\n*.npz\n*.svg\n!/proj/drawing.svg\n")      # types that are ignored, with one deliberate exception
        put("proj/temp/README.md")
        put("proj/training/run.py", "import sys\nsys.path.insert(0, 'temp/lib')\nprint(1)")
        put("proj/temp/tool.sh", "echo hi")                      # scratch, used by the code below
        put("proj/temp/lonely.py", "pass")                       # scratch, named by nobody
        put("proj/temp/lib/mod.py", "x = 1")                     # a scratch folder that code puts on sys.path
        put("proj/experiments/e1/experiment.md", "made with `temp/tool.sh` and `temp/gone.sh`, see temp/lonely.py.")
        put("proj/training/launch.sh", "bash $REPO/temp/tool.sh")
        put("proj/history/old.md", "temp/lonely.py")             # archives are not scanned
        sh("add", "-A", "-f")
        sh("commit", "-qm", "base")
        sh("branch", "-q", "up")                                 # the "upstream": a local branch is enough for the arithmetic
        sh("branch", "-q", "--set-upstream-to=up")
        used, imports = used_scratch(top)
        assert set(used) == {"proj/temp/tool.sh", "proj/temp/lonely.py"}, used                  # temp/gone.sh does not exist; history/ not scanned
        assert used["proj/temp/tool.sh"]["executed"] == ["proj/training/launch.sh"] and used["proj/temp/tool.sh"]["cited"] == ["proj/experiments/e1/experiment.md"]
        assert used["proj/temp/lonely.py"]["executed"] == [] and used["proj/temp/lonely.py"]["cited"] == ["proj/experiments/e1/experiment.md"]
        assert imports == [("proj/training/run.py", "proj/temp/lib")], imports
        put("proj/temp/new.py", "pass")
        put("proj/rec/a b.json", "{}")                           # names with a space; two untracked files in one folder
        put("proj/rec/c.json", "{}")
        assert sorted(names(top, "ls-files", "--cached", "--others", "--exclude-standard", "--", os.path.join(top, "proj", "rec"))) == ["proj/rec/a b.json", "proj/rec/c.json"]
        put("proj/data.npz", binary=10)
        put("proj/evaluation/good.py", "pass")
        put("proj/temp/loose.sh", "echo")                        # untracked scratch: counted, not an error
        put("proj/ign/a.ign")                                    # a folder with ignored files only
        sh("add", "-f", "proj/temp/new.py", "proj/evaluation/good.py")
        errors, text = report(top)
        assert errors == ["scratch: proj/temp/new.py - inside a temp/ or _to_delete/ folder (scratch is not committed; promote it to a code folder or put the command in the card)"], errors
        assert "pending: 2 staged file(s), 3 untracked (1 of them scratch" in text and "untracked scratch: proj/temp/loose.sh" in text, text
        assert path_errors(top, ["proj/evaluation/good.py"]) == [] and len(path_errors(top, ["proj/data.npz", "proj/temp/x.py"])) == 1      # x.py does not exist: skipped
        put("proj/drawing.svg", "<svg/>")                        # whitelisted by the .gitignore of the repository: fine; the same type elsewhere is not
        put("proj/other.svg", "<svg/>")
        assert path_errors(top, ["proj/drawing.svg"]) == [] and len(path_errors(top, ["proj/other.svg"])) == 1
        sh("rm", "-q", "-f", "proj/temp/lonely.py")                # a staged deletion of scratch is no error (the legacy clean-up must stay possible)
        assert path_errors(top, ["proj/temp/lonely.py"]) == [] and report(top)[0] == errors
        sh("commit", "-qm", "scratch by mistake")                # now a push would publish it
        errors, _ = report(top)
        assert any("a push would publish scratch file proj/temp/new.py" in e and "(commit " in e for e in errors), errors
        assert push_check(top)[1][0].startswith("up: 1 commit(s) ahead, 0 behind"), push_check(top)[1]
        put("proj/code/big.json", "0" * 3_000_000)               # a 3 MB file in a later commit
        sh("add", "proj/code/big.json")
        put("proj/code/hidden.npz", binary=5)                    # added in one commit, deleted in the next: the push still publishes the blob
        sh("add", "-f", "proj/code/hidden.npz")
        put("proj/training/run.py", "#" * 1_500_000)             # an old file that grows past 1 MB
        sh("add", "proj/training/run.py")
        sh("commit", "-qm", "big and hidden")
        sh("rm", "-q", "-f", "proj/code/hidden.npz")
        sh("commit", "-qm", "delete the hidden file again")
        errors, _ = report(top)
        assert any("proj/code/big.json (3000 kB)" in e for e in errors) and any("forbidden file proj/code/hidden.npz" in e for e in errors), errors
        assert any("forbidden file proj/training/run.py (1500 kB)" in e for e in errors), errors
        # the command line: exit codes
        here = os.getcwd()
        os.chdir(top)
        try:
            assert main(["--paths", "proj/evaluation/good.py"]) == 0 and main(["--paths", "proj/temp/new.py"]) == 1 and main([]) == 1
            for bad in (["--paths"], ["--path", "proj/x.py"], ["--paths", "proj/ign"]):                # no path; a typo; a folder of ignored files only
                try:
                    code = main(bad)
                except SystemExit as e:
                    code = e.code
                assert code == 2, (bad, code)
        finally:
            os.chdir(here)
    with tempfile.TemporaryDirectory() as lone:                  # no remote at all, and a remote without an upstream branch
        subprocess.run(["git", "-C", lone, "init", "-q", "-b", "main"], check=True)
        open(os.path.join(lone, "a.py"), "w").write("x")
        subprocess.run(["git", "-C", lone, "add", "-A"], check=True)
        subprocess.run(["git", "-C", lone, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "m"], check=True)
        assert push_check(lone) == ([], ["no remote: nothing to push"])
        os.makedirs(os.path.join(lone, "temp"))
        open(os.path.join(lone, "temp", "s.py"), "w").write("x")
        subprocess.run(["git", "-C", lone, "add", "-A"], check=True)
        subprocess.run(["git", "-C", lone, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "scratch"], check=True)
        subprocess.run(["git", "-C", lone, "remote", "add", "origin", lone], check=True)               # a remote, but no branch pushed: all commits are unpushed
        errors, info = push_check(lone)
        assert errors and "no upstream branch" in info[0], (errors, info)
    print("check_git selftest ok: classes (separators, case, substrings, extensions, size), the reference regex, references resolved ($VAR, project folder, missing target, "
          "history/), sys.path folders, staged scratch refused and a staged deletion not, pushes that add scratch / big / deleted-again files refused, no remote / no upstream, exit codes")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(prog="check_git", description=__doc__.split("\n")[0], allow_abbrev=False)     # no abbreviations: `--path A` must not pass for `--paths A`
    ap.add_argument("--paths", nargs="+", metavar="PATH", help="check these paths (a folder: its files) as the content of the next commit")
    ap.add_argument("--used", action="store_true", help="list the scratch that living files point at")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    top = toplevel()
    if a.paths:
        paths = []
        for p in a.paths:
            if os.path.isdir(p):                                       # a folder: the files git would take from it (tracked, or untracked and not ignored)
                paths += [os.path.join(top, f) for f in names(top, "ls-files", "--cached", "--others", "--exclude-standard", "--", os.path.abspath(p))]
            else:
                paths.append(p)
        paths = [os.path.relpath(os.path.abspath(p), top).replace("\\", "/") for p in paths]
        if not paths:
            print("nothing to check: the named folders hold no file that git would take (all ignored?)")
            return 2
        errors = path_errors(top, paths)
        print("\n".join(errors) if errors else f"ok: {len(paths)} path(s) fit for git")
        return 1 if errors else 0
    errors, text = report(top, show_all=a.used)
    print(text)
    print("\n" + (f"{len(errors)} ERROR(S)" if errors else "no errors"))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
