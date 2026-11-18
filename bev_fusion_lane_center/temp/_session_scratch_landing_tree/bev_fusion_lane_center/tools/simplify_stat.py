"""simplify_stat.py - what a simplification removed, measured from git: files and lines of named path groups at a BEFORE ref (an archive tag or a commit) and at an AFTER ref, written as a json the reporter reads (`values` of report.json),
so the "simplification" table of a report fills itself.

    wsl python3 tools/simplify_stat.py --before archive/s2-s3-blocks-2026-11-18~1 --after HEAD --group token=training/grid/jtoken.py,training/train/jtoken_train.py --group tests=training/grid/test_jtoken*.py --out temp/simplify.json
    wsl python3 tools/simplify_stat.py selftest

A group is `NAME=pathspec,pathspec,...` (git pathspecs, `*` matches inside a name, relative to the repository root; NAME without dots). Only text files that git tracks are counted (`*.py` and what the pathspec names). Per group the json holds
`files_before`, `files_after`, `lines_before`, `lines_after`, `lines_removed` (before - after, negative = grew), `files_removed` / `files_added` (counts) and `removed` / `added` (the file names, sorted); the top level holds `before` and `after` (the refs resolved to commit ids).
Counts come from `git grep -c ''` at the ref (no checkout, the working tree is not read). Not measured here: parameters and milliseconds (they come from the cards of the compared systems).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile


def git(repo, *args):
    r = subprocess.run(["git", "-C", repo, "-c", "safe.directory=*", *args], capture_output=True, text=True)
    if r.returncode not in (0, 1):                                                      # git grep exits 1 when nothing matches
        raise SystemExit(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def counts(repo, ref, specs):
    """{path: lines} of the tracked files that match the pathspecs at `ref`"""
    out = git(repo, "grep", "-c", "", ref, "--", *specs)
    res = {}
    for line in out.splitlines():
        path, _, n = line.rpartition(":")
        path = path[len(ref) + 1:] if path.startswith(ref + ":") else path
        res[path] = int(n)
    return res


def measure(repo, before, after, groups):
    out = {"before": git(repo, "rev-parse", before).strip(), "after": git(repo, "rev-parse", after).strip()}
    for name, specs in groups.items():
        b, a = counts(repo, before, specs), counts(repo, after, specs)
        out[name] = dict(files_before=len(b), files_after=len(a), lines_before=sum(b.values()), lines_after=sum(a.values()), lines_removed=sum(b.values()) - sum(a.values()),
                         files_removed=len(set(b) - set(a)), files_added=len(set(a) - set(b)), removed=sorted(set(b) - set(a)), added=sorted(set(a) - set(b)))
    return out


def parse_groups(items):
    groups = {}
    for it in items:
        name, _, specs = it.partition("=")
        assert name and "." not in name and specs, f"--group NAME=pathspec[,pathspec]: {it!r}"
        groups[name] = [s for s in specs.split(",") if s]
    return groups


def selftest():
    with tempfile.TemporaryDirectory() as d:
        run = lambda *a: subprocess.run(["git", "-C", d, "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True)  # noqa: E731
        run("init", "-q")
        os.makedirs(os.path.join(d, "m"))
        w = lambda p, n: open(os.path.join(d, p), "w").write("x\n" * n)  # noqa: E731
        w("m/a.py", 10)
        w("m/b.py", 5)
        w("m/t_a.py", 3)
        w("other.txt", 7)
        run("add", ".")
        run("commit", "-qm", "one")
        run("tag", "before")
        os.remove(os.path.join(d, "m/b.py"))                                            # a file removed, one shrunk, one added, one outside the groups changed
        w("m/a.py", 4)
        w("m/c.py", 2)
        w("other.txt", 1)
        run("add", "-A")
        run("commit", "-qm", "two")
        r = measure(d, "before", "HEAD", {"code": ["m/a.py", "m/b.py", "m/c.py"], "tests": ["m/t_*.py"], "none": ["nothing/*.py"]})
        c = r["code"]
        assert (c["files_before"], c["files_after"], c["lines_before"], c["lines_after"], c["lines_removed"]) == (2, 2, 15, 6, 9), c
        assert (c["files_removed"], c["files_added"], c["removed"], c["added"]) == (1, 1, ["m/b.py"], ["m/c.py"]), c
        assert r["tests"]["lines_before"] == r["tests"]["lines_after"] == 3 and r["tests"]["files_removed"] == 0, r["tests"]
        assert r["none"]["files_before"] == 0 and r["none"]["lines_removed"] == 0, r["none"]
        assert r["before"] != r["after"] and len(r["before"]) == 40
        try:
            measure(d, "no-such-ref", "HEAD", {"code": ["m/a.py"]})
            raise AssertionError("an unknown ref must stop")
        except SystemExit:
            pass
        assert parse_groups(["a=x/*.py,y.py"]) == {"a": ["x/*.py", "y.py"]}
    print("simplify_stat selftest ok")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", default="HEAD")
    ap.add_argument("--group", action="append", required=True)
    ap.add_argument("--out")
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    a = ap.parse_args()
    res = measure(os.path.abspath(a.repo), a.before, a.after, parse_groups(a.group))
    text = json.dumps(res, indent=1)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
