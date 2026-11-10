"""The frame of the BEV system is frozen (../FROZEN.md): every file listed in ../frozen.json has the recorded SHA-256. CPU, milliseconds, no data.

    python tests/test_frozen.py             check; exit 1 and the list of changed / missing files otherwise
    python tests/test_frozen.py --update    rewrite the hashes of the listed files (only for a bug fix, with a line in FROZEN.md's changelog)
    python tests/test_frozen.py --selftest  the check on a toy folder: a clean copy passes, an edited file and a missing file fail

The list is the table of FROZEN.md written as data (frozen.json: {"files": [paths relative to bev_fusion/]}); a new frozen file is added to the list and to the table together."""
import hashlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()       # line endings do not count


def check(root, spec):
    """(changed, missing): the listed files whose hash differs from the recorded one, and the ones that do not exist"""
    changed, missing = [], []
    for rel, want in spec["files"].items():
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            missing.append(rel)
        elif sha(p) != want:
            changed.append(rel)
    return changed, missing


def selftest():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "training"))
        for n in ("a.py", "b.py"):
            with open(os.path.join(d, "training", n), "w") as f:
                f.write(f"x = '{n}'\n")
        spec = {"files": {f"training/{n}": sha(os.path.join(d, "training", n)) for n in ("a.py", "b.py")}}
        assert check(d, spec) == ([], []), "a clean copy passes"
        with open(os.path.join(d, "training", "a.py"), "a") as f:
            f.write("y = 1\n")
        assert check(d, spec) == (["training/a.py"], []), "an edited file fails"
        os.remove(os.path.join(d, "training", "b.py"))
        assert check(d, spec) == (["training/a.py"], ["training/b.py"]), "a missing file fails"
        with open(os.path.join(d, "training", "a.py"), "w", newline="") as f:
            f.write("x = 'a.py'\r\n")
        assert check(d, {"files": {"training/a.py": hashlib.sha256(b"x = 'a.py'\n").hexdigest()}}) == ([], []), "line endings do not count"
    print("test_frozen selftest passed")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    path = os.path.join(ROOT, "frozen.json")
    spec = json.load(open(path, encoding="utf-8"))
    if "--update" in sys.argv:
        spec["files"] = {rel: sha(os.path.join(ROOT, rel)) for rel in spec["files"]}
        json.dump(spec, open(path, "w", encoding="utf-8"), indent=1)
        open(path, "a").write("\n")
        print(f"frozen.json updated: {len(spec['files'])} files (add the reason to the changelog of FROZEN.md)")
        return
    changed, missing = check(ROOT, spec)
    if changed or missing:
        print("FROZEN FRAME GATE: FAIL")
        for rel in changed:
            print(f"  changed: {rel}")
        for rel in missing:
            print(f"  missing: {rel}")
        print("  a frozen file changes only for a bug (FROZEN.md: a test, a critic pass, a changelog line, `python tests/test_frozen.py --update`)")
        sys.exit(1)
    print(f"FROZEN FRAME GATE: PASS ({len(spec['files'])} files unchanged)")


if __name__ == "__main__":
    main()
