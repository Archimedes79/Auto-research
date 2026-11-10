"""Commit exactly the paths you name, whatever else is staged or modified in the shared checkout (one tool instead of a commit script per chat).
Standard library only.

    python autoresearch/tools/commit_paths.py -F msg.txt PATH [PATH ...]                    working-tree state of each path (a folder: all of it; a deleted file: its deletion)
    python autoresearch/tools/commit_paths.py -m "subject" --trailer "Co-Authored-By: ..." PATH ...
    python autoresearch/tools/commit_paths.py -F msg.txt PATH ... --blob autoresearch/README.md=/tmp/readme_mine.md
                                                         a file that other chats also edit: commit YOUR version (the committed file + only your hunks, built beforehand)
    python autoresearch/tools/commit_paths.py ... --dry-run                                  print what would be committed, change nothing
    python autoresearch/tools/commit_paths.py --selftest

Paths are relative to the folder you run it in (or absolute); they are literal names, no globs. How it works (the method of the git rule, "own paths only"):
a temporary index is filled from HEAD, the paths are added to it, a tree and a commit are written with git plumbing, and HEAD moves with a compare-and-swap
(`update-ref HEAD new old`: when another chat committed in the meantime, nothing happens and you run it again). Then the shared index is brought in line for
the files you committed, and only for those whose staged content is the old or the new committed version: a file that another chat staged with other
content stays staged (the tool says so; their next commit would revert your part of it, so tell them). Working files are never touched.
Refused (exit 1, nothing committed): `.` or the top folder, a path outside the repository, a path that does not exist and is not in HEAD, an ignored path,
an embedded repository (a gitlink: temp/gt_v8 is a worktree), scratch or forbidden files (`check_git.py` classes: temp/, _to_delete/, data, binaries,
over 1 MB, measured on the committed content; --allow-scratch lifts the temp/ rule only, for a record that must have it: a checkpoint or a file over 1 MB
in temp/ is still refused), an empty result (the tree would not
change), a missing git identity. The author is the repository's git identity; the message ends with the --trailer lines you give (the session's attribution).
Hooks are not run (plumbing commit): run the checks yourself (`check_git.py --paths`, the folder's tests).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_git as CG  # noqa: E402


def run(top, *args, env=None, inp=None, check=True, literal=True):
    env = dict(env if env is not None else os.environ)
    env.pop("GIT_LITERAL_PATHSPECS", None)
    if literal:
        env["GIT_LITERAL_PATHSPECS"] = "1"                                  # names are literal: no globs, no `:(top)` magic (check-ignore does not support it)
    r = subprocess.run(["git", "-C", top, *args], capture_output=True, env=env, input=inp.encode("utf-8") if inp is not None else None)   # bytes in and out: a text-mode pipe turns LF into CRLF on Windows
    r.stdout, r.stderr = (x.decode("utf-8", errors="replace") for x in (r.stdout, r.stderr))
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} failed: {(r.stderr or r.stdout).strip()}")
    return r


def to_local(p):
    """a Windows drive path under a POSIX python (WSL) becomes /mnt/<drive>/..."""
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    return f"/mnt/{m.group(1).lower()}/{m.group(2).replace(chr(92), '/')}" if m and os.name != "nt" else p


def rel_to_top(top, p):
    full = os.path.abspath(to_local(p))
    rel = os.path.relpath(full, os.path.abspath(top)).replace("\\", "/")
    if rel in (".", ""):
        raise RuntimeError(f"{p}: the whole repository is not a path (name the files or folders you made)")
    if rel == ".." or rel.startswith("../"):
        raise RuntimeError(f"{p} is outside the repository")
    return rel


def diff_raw(top, a, b):
    """[(old mode, new mode, old sha, new sha, status, path)] of `git diff-tree -r` between two trees (NUL separated: names are exact)"""
    out = run(top, "diff-tree", "-r", "-z", "--no-renames", a, b).stdout.split("\0")
    rows, i = [], 0
    while i + 1 < len(out) and out[i].startswith(":"):
        om, nm, osha, nsha, st = out[i][1:].split()
        rows.append((om, nm, osha, nsha, st[0], out[i + 1]))
        i += 2
    return rows


def sync_shared_index(top, rows):
    """bring the shared index in line for the committed files: only an entry that equals the old or the new committed content is updated; -> (updated, left staged by others)"""
    names = [r[5] for r in rows]
    staged = {}
    for chunk in (names[i:i + 200] for i in range(0, len(names), 200)):
        for rec in run(top, "ls-files", "-s", "-z", "--", *chunk).stdout.split("\0"):
            if rec:
                meta, path = rec.split("\t", 1)
                staged[path] = meta.split()[1]
    left, todo = [], []
    zero = "0" * 40
    for om, nm, osha, nsha, st, path in rows:
        have = staged.get(path)                                       # None: not in the shared index
        old = None if osha == zero else osha
        new = None if nsha == zero else nsha
        if have in (old, new):
            todo.append((nm, new, path))
        else:
            left.append(path)
    if todo:
        lines = "".join((f"{nm} {new}\t{path}\0" if new else f"0 {zero}\t{path}\0") for nm, new, path in todo)
        for attempt in range(6):                                     # a concurrent git command holds index.lock for a moment
            r = run(top, "update-index", "-z", "--index-info", inp=lines, check=False)
            if r.returncode == 0:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("the commit is made, but the shared index could not be updated (" + (r.stderr or "").strip().splitlines()[-1]
                               + "): repair with `git reset -q HEAD -- " + " ".join(f"'{t[2]}'" for t in todo[:5]) + (" ..." if len(todo) > 5 else "") + "` or `git update-index --refresh`")
    return len(todo), left


def commit_paths(top, paths, message, blobs=None, allow_scratch=False, dry_run=False):
    """-> (commit id or None or "dry-run", [changed files], [files left staged by others]); RuntimeError with the reason when it refuses"""
    blobs = {rel_to_top(top, k): os.path.abspath(v) for k, v in (blobs or {}).items()}
    rels = [rel_to_top(top, p) for p in paths]
    clash = sorted(set(rels) & set(blobs))
    if clash:
        raise RuntimeError(f"named as a path and as a blob: {clash[0]} (give it once)")
    for var in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT"):
        if run(top, "var", var, check=False).returncode != 0:
            raise RuntimeError("no git identity (git config user.name / user.email): a commit cannot be made")
    old = run(top, "rev-parse", "HEAD").stdout.strip()
    old_tree = run(top, "rev-parse", f"{old}^{{tree}}").stdout.strip()
    tmpd = tempfile.mkdtemp(prefix="commit_paths_")
    env = dict(os.environ, GIT_INDEX_FILE=os.path.join(tmpd, "index"))
    try:
        run(top, "read-tree", old, env=env)
        for rel in rels:
            tracked = run(top, "ls-files", "--error-unmatch", "--", rel, env=env, check=False).returncode == 0
            if not tracked and not os.path.lexists(os.path.join(top, rel)):
                raise RuntimeError(f"{rel}: neither on disk nor in HEAD")
            r = run(top, "add", "-A", *(["-f"] if allow_scratch else []), "--", rel, env=env, check=False)
            if r.returncode != 0:
                raise RuntimeError(f"{rel}: git add refused: {(r.stderr or r.stdout).strip().splitlines()[0] if (r.stderr or r.stdout).strip() else 'ignored?'}  (an ignored path is not committed)")
        for rel, src in blobs.items():
            if run(top, "check-ignore", "-q", "--no-index", "--", rel, check=False, literal=False).returncode == 0 and not allow_scratch:
                raise RuntimeError(f"{rel}: an ignored path is not committed")
            if not os.path.isfile(src):
                raise RuntimeError(f"{rel}: the blob source {src} does not exist")
            h = run(top, "hash-object", "-w", "--path", rel, src).stdout.strip()
            mode = next((x.split()[0] for x in run(top, "ls-tree", old, "--", rel).stdout.splitlines() if x), "100644")
            run(top, "update-index", "--add", "--cacheinfo", f"{mode},{h},{rel}", env=env)
        tree = run(top, "write-tree", env=env).stdout.strip()
        if tree == old_tree:
            return None, [], []
        rows = diff_raw(top, old_tree, tree)
        changed = [r[5] for r in rows]                                   # deletions too
        links = [r[5] for r in rows if r[1] == "160000"]
        if links:
            raise RuntimeError(f"refused, nothing committed: {links[0]} is an embedded repository (a gitlink): commit the files inside it from there")
        errors = []
        for om, nm, osha, nsha, st, path in rows:
            if st == "D":
                continue
            size = int(run(top, "cat-file", "-s", nsha).stdout.strip())      # the committed content, not the working file
            kind = CG.verdict(top, path, size)
            if kind == "scratch" and allow_scratch:                          # lifts the folder rule only: the type and size rules still hold for the file itself
                kind = "forbidden" if "__pycache__" in path.split("/") else CG.classify(os.path.basename(path), size)   # by name: no .gitignore whitelist applies inside temp/
            if kind != "ok":
                errors.append(f"{kind}: {path} - " + ("inside a temp/ or _to_delete/ folder (promote it to a code folder or put the command in the card)" if kind == "scratch"
                                                       else f"a data / binary / log type or over 1 MB ({size // 1000} kB)"))
        if errors:
            raise RuntimeError("refused, nothing committed:\n  " + "\n  ".join(errors[:12]))
        if dry_run:
            return "dry-run", changed, []
        new = run(top, "commit-tree", tree, "-p", old, inp=message).stdout.strip()
        r = run(top, "update-ref", "-m", "commit_paths", "HEAD", new, old, check=False)      # compare-and-swap: fails when HEAD moved meanwhile
        if r.returncode != 0:
            raise RuntimeError("HEAD moved or the ref is locked (another chat committed): run it again" + (f" [{r.stderr.strip()}]" if r.stderr.strip() else ""))
        _, left = sync_shared_index(top, rows)
        return new, changed, left
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


def selftest():
    tmp = tempfile.mkdtemp()
    saved, cwd0 = dict(os.environ), os.getcwd()
    try:
        for v in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX"):
            os.environ.pop(v, None)
        os.environ.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
        top = os.path.join(tmp, "r")
        os.makedirs(top)
        os.chdir(top)                                              # paths are relative to the folder you run it in

        def g(*a):
            return run(top, *a).stdout

        def w(rel, text, mode=None):
            os.makedirs(os.path.dirname(os.path.join(top, rel)) or top, exist_ok=True)
            open(os.path.join(top, rel), "w", encoding="utf-8").write(text)
            if mode:
                os.chmod(os.path.join(top, rel), mode)
        g("init", "-q")
        w("a.py", "a = 1\n")
        w("shared.md", "line1\nline2\n")
        w("old.txt", "x\n")
        w("run.sh", "#!/bin/sh\n", 0o755)
        w("sub/in.md", "in\n")
        w(".gitignore", "ign.txt\ntemp/\n*.log\n")
        g("add", "-A")
        g("commit", "-q", "-m", "base")
        # my paths and another chat's staged / modified work
        w("mine/new.py", "n = 1\n")
        w("a.py", "a = 2\n")
        w("other.py", "theirs\n")
        g("add", "other.py")                                       # another chat's staged file
        os.remove(os.path.join(top, "old.txt"))                    # a deletion of mine
        w("shared.md", "line1 theirs\nline2 mine\n")               # a shared file: the other chat's hunk is in the working file
        open(os.path.join(tmp, "shared_mine.md"), "w").write("line1\nline2 mine\n")   # my version: the committed file + only my hunk
        w("run.sh", "#!/bin/sh\necho x\n", 0o755)
        open(os.path.join(tmp, "run_mine.sh"), "w").write("#!/bin/sh\necho x\n")
        new, changed, left = commit_paths(top, ["mine", "a.py", "old.txt"], "msg\n\nTrailer: x\n", blobs={"shared.md": os.path.join(tmp, "shared_mine.md"), "run.sh": os.path.join(tmp, "run_mine.sh")})
        assert new and sorted(changed) == ["a.py", "mine/new.py", "old.txt", "run.sh", "shared.md"] and left == [], (changed, left)      # old.txt = my deletion, listed too
        assert g("show", "-s", "--format=%B", "HEAD").strip().endswith("Trailer: x")
        assert g("show", "HEAD:shared.md") == "line1\nline2 mine\n" and g("show", "HEAD:a.py") == "a = 2\n", "my blob, my file"
        assert "100755" in g("ls-tree", "HEAD", "run.sh"), "the exec bit of a blob path is kept"
        assert "old.txt" not in g("ls-tree", "-r", "--name-only", "HEAD") and "other.py" not in g("ls-tree", "-r", "--name-only", "HEAD"), "my deletion yes, their file no"
        assert open(os.path.join(top, "shared.md")).read() == "line1 theirs\nline2 mine\n", "the working file keeps their hunk"
        assert g("diff", "--cached", "--name-only").split() == ["other.py"], "their staged file is still staged, mine is not"
        assert not [d for d in os.listdir(tempfile.gettempdir()) if d.startswith("commit_paths_")], "the temporary index is removed"
        # another chat staged a DIFFERENT version of a file I commit: their staging stays, and I am told
        w("b.py", "b = 1\n")
        g("add", "b.py")
        w("b.py", "b = 1  # mine\n")
        g("add", "b.py")                                           # staged: "b = 1 # mine" ...
        w("b.py", "b = 2  # theirs, staged later\n")
        g("add", "b.py")
        open(os.path.join(tmp, "b_mine.py"), "w").write("b = 1  # mine\n")
        _, ch, left = commit_paths(top, [], "m\n", blobs={"b.py": os.path.join(tmp, "b_mine.py")})
        assert ch == ["b.py"] and left == ["b.py"] and "b.py" in g("diff", "--cached", "--name-only").split(), (ch, left)
        g("reset", "-q", "HEAD", "--", "b.py")
        os.remove(os.path.join(top, "b.py"))
        g("rm", "-q", "--cached", "--ignore-unmatch", "b.py")
        # refusals
        w("ign.txt", "i\n")
        w("temp/x.py", "t\n")
        w("big.txt", "x" * 1_300_000)
        w("ä.py", "u\n")
        os.makedirs(os.path.join(top, "temp", "x"), exist_ok=True)
        w("_to_delete/ä.py", "u\n")
        sub = os.path.join(top, "sub")
        os.makedirs(os.path.join(top, "emb"))
        subprocess.run(["git", "-C", os.path.join(top, "emb"), "init", "-q"], check=True, env=dict(os.environ))
        w("emb/f.txt", "f\n")
        subprocess.run(["git", "-C", os.path.join(top, "emb"), "add", "f.txt"], check=True)
        subprocess.run(["git", "-C", os.path.join(top, "emb"), "commit", "-q", "-m", "e"], check=True)
        w("c.py", "c\n")
        w("other_dir/g.py", "g\n")
        for bad, word in ((["a.py"], None), (["ign.txt"], "git add refused"), (["nope.py"], "neither on disk"), (["temp/x.py"], "git add refused"), (["big.txt"], "forbidden"),
                          (["_to_delete/ä.py"], "scratch"), (["."], "whole repository"), ([top], "whole repository"), (["../elsewhere.py"], "outside"), (["*.py"], "neither on disk"),
                          (["emb"], "embedded repository")):
            try:
                got = commit_paths(top, bad, "m\n")
                assert word is None and got == (None, [], []), (bad, got)
            except RuntimeError as e:
                assert word and word in str(e), (bad, str(e))
        assert commit_paths(top, ["temp/x.py"], "m\n", allow_scratch=True)[0], "scratch only when asked (-f for ignored paths)"
        # review 2026-11-02: --allow-scratch lifts the temp/ rule only (a checkpoint or a file over 1 MB there is still refused); a deletion-only commit lists the file;
        # the message goes to git as bytes (a text-mode pipe gives CRLF on Windows)
        w("temp/ck.pt", "weights")
        w("temp/big.txt", "x" * 1_300_000)
        for bad in ("temp/ck.pt", "temp/big.txt"):
            try:
                commit_paths(top, [bad], "m\n", allow_scratch=True)
                raise AssertionError(f"{bad}: --allow-scratch must not lift the type and size rules")
            except RuntimeError as e:
                assert "forbidden: " + bad in str(e), (bad, str(e))
        os.remove(os.path.join(top, "temp", "x.py"))
        dele, chd, _ = commit_paths(top, ["temp/x.py"], "m\n", allow_scratch=True)
        assert dele and chd == ["temp/x.py"], "a deletion-only commit names the file"
        seen, real_sub = [], subprocess.run
        subprocess.run = lambda cmd, *a, **kw: (seen.append(kw.get("input")) if "commit-tree" in cmd else None) or real_sub(cmd, *a, **kw)
        try:
            assert commit_paths(top, ["other_dir"], "ä msg\n")[0]
        finally:
            subprocess.run = real_sub
        assert len(seen) == 1 and isinstance(seen[0], bytes) and g("log", "-1", "--format=%s").strip() == "ä msg", seen
        try:
            commit_paths(top, ["c.py"], "m\n", blobs={"c.py": os.path.join(tmp, "b_mine.py")})
            raise AssertionError("a path named twice must be refused")
        except RuntimeError as e:
            assert "named as a path and as a blob" in str(e)
        try:
            commit_paths(top, [], "m\n", blobs={"x.log": os.path.join(tmp, "b_mine.py")})
            raise AssertionError("an ignored blob path must be refused")
        except RuntimeError as e:
            assert "ignored" in str(e), e
        # relative to the folder you run it in
        old_cwd = os.getcwd()
        os.chdir(sub)
        try:
            w("sub/new_in_sub.md", "n\n")
            w("new_in_sub.md", "top-level one\n")
            n2, ch2, _ = commit_paths(top, ["new_in_sub.md"], "m\n")
            assert ch2 == ["sub/new_in_sub.md"], ch2
        finally:
            os.chdir(old_cwd)
        # a dry run changes nothing; an unknown identity refuses
        head = g("rev-parse", "HEAD").strip()
        dry, ch, _ = commit_paths(top, ["c.py"], "m\n", dry_run=True)
        assert dry == "dry-run" and ch == ["c.py"] and g("rev-parse", "HEAD").strip() == head, "a dry run changes nothing"
        for v in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
            os.environ.pop(v)
        home = os.environ.get("HOME")
        os.environ.update(HOME=tmp, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        run(top, "config", "user.useConfigOnly", "true")
        try:
            commit_paths(top, ["c.py"], "m\n", dry_run=True)
            raise AssertionError("no identity: refuse also in a dry run")
        except RuntimeError as e:
            assert "no git identity" in str(e), e
        run(top, "config", "--unset", "user.useConfigOnly")
        os.environ.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
        if home:
            os.environ["HOME"] = home
        # the compare-and-swap, and a locked index after the commit
        real = run
        calls = {"n": 0}

        def moving(top_, *a, **kw):
            if a and a[0] == "update-ref" and calls["n"] == 0:      # another chat commits between the tree and the ref update
                calls["n"] += 1
                subprocess.run(["git", "-C", top_, "commit", "-q", "--allow-empty", "-m", "someone else"], check=True)
            return real(top_, *a, **kw)
        globals()["run"] = moving
        try:
            commit_paths(top, ["c.py"], "m\n")
            raise AssertionError("a moved HEAD must stop the commit")
        except RuntimeError as e:
            assert "HEAD moved" in str(e), e
        finally:
            globals()["run"] = real
        w("d.py", "d\n")
        g("add", "d.py")
        lock = os.path.join(top, ".git", "index.lock")
        open(lock, "w").write("")
        real_sleep = time.sleep
        time.sleep = lambda s: None
        try:
            commit_paths(top, ["d.py"], "m\n")
            raise AssertionError("a locked shared index must be reported")
        except RuntimeError as e:
            assert "shared index could not be updated" in str(e) and "git reset -q HEAD" in str(e), e
        finally:
            time.sleep = real_sleep
            os.remove(lock)
        assert g("log", "-1", "--format=%s").strip() == "m", "the commit itself was made"
        g("reset", "-q", "HEAD", "--", "d.py")
        assert commit_paths(top, ["c.py"], "m\n")[0], "and works on the next try"
        print("commit_paths selftest passed")
    finally:
        os.chdir(cwd0)
        os.environ.clear()
        os.environ.update(saved)
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("-m", "--message")
    ap.add_argument("-F", "--message-file")
    ap.add_argument("--trailer", action="append", default=[], help="a line appended to the message (repeatable), e.g. the Co-Authored-By of the session")
    ap.add_argument("--blob", action="append", default=[], metavar="PATH=SOURCE", help="commit SOURCE's content as PATH (a file other chats also edit)")
    ap.add_argument("--allow-scratch", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.paths or a.blob) or not (a.message or a.message_file):
        ap.error("name the paths and a message (-m or -F)")
    msg = a.message if a.message else open(a.message_file, encoding="utf-8-sig").read()
    msg = msg.rstrip("\n") + "\n" + ("\n" + "\n".join(a.trailer) + "\n" if a.trailer else "")
    top = CG.toplevel(".")
    blobs = dict(b.split("=", 1) for b in a.blob)
    try:
        new, changed, left = commit_paths(top, a.paths, msg, blobs, a.allow_scratch, a.dry_run)
    except RuntimeError as e:
        sys.exit(str(e))
    if new is None:
        print("nothing to commit: the named paths equal HEAD")
    else:
        print(("would commit " if new == "dry-run" else f"committed {new[:10]}: ") + f"{len(changed)} file(s)" + "".join(f"\n  {c}" for c in changed[:15]) + ("\n  ..." if len(changed) > 15 else ""))
        if left:
            print("LEFT STAGED (another chat staged other content; their next commit would revert your part): " + ", ".join(left))
    return 0


if __name__ == "__main__":
    sys.exit(main())
