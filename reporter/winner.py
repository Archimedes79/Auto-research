#!/usr/bin/env python3
"""winner: export the code of a winning system as a small, self-describing folder (source + README + MANIFEST, never weights),
so that it can be handed over, put into git, and checked later for drift.

    python winner.py export <name> --src <file-or-glob> [--src ...] [--strip <base dir>] --out <project>/winner [--limit 1000000]
                             [--title T] [--numbers "VAL topo F1 0.6931 / 0.6964 (s0 / s1)"] [--command "python ... "] [--weights "runs/x (not in git)"]
                             [--allow-dirty]
    python winner.py verify <project>/winner/<name>        # re-hash the files against MANIFEST.json; if the sources still exist, report drift
    python winner.py selftest

What it writes into <out>/<name>/: code/ (the files, relative to --strip), README.md (what it is, the command, the numbers to reproduce, where the weights
are, the source commit), MANIFEST.json (file, size, sha256; source paths; git commit and whether the sources had uncommitted changes; date).
Refuses: weight-like files (.pt .pth .ckpt .npy .npz .pkl .bin .safetensors), a single file over 300 KB, a folder over --limit bytes, sources with uncommitted
changes (unless --allow-dirty, which is written into the manifest). Standalone: python 3.8+, git optional."""
import io
import contextlib
import argparse, datetime, glob, hashlib, json, os, shutil, subprocess, sys, tempfile

WEIGHTS = (".pt", ".pth", ".ckpt", ".npy", ".npz", ".pkl", ".bin", ".safetensors", ".onnx", ".h5")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(cwd, *a):
    try:
        return subprocess.run(("git",) + a, cwd=cwd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


def export(a):
    files = []
    for g in a.src:
        files += [f for f in sorted(glob.glob(g, recursive=True)) if os.path.isfile(f) and "__pycache__" not in f]
    files = sorted(set(os.path.abspath(f) for f in files))
    if not files:
        sys.exit("no source files matched")
    strip = os.path.abspath(a.strip) if a.strip else os.path.commonpath([os.path.dirname(f) for f in files])
    bad = [f for f in files if f.lower().endswith(WEIGHTS)]
    if bad:
        sys.exit("refused: weight-like files (they never go into a winner folder): " + ", ".join(os.path.basename(b) for b in bad[:5]))
    big = [f for f in files if os.path.getsize(f) > 300_000]
    if big:
        sys.exit("refused: file over 300 KB: " + ", ".join(os.path.basename(b) for b in big[:5]))
    total = sum(os.path.getsize(f) for f in files)
    if total > a.limit:
        sys.exit("refused: %d bytes of source, limit %d" % (total, a.limit))
    root = os.path.dirname(files[0])
    dirty = [f for f in files if git(os.path.dirname(f), "status", "--porcelain", "--", f)]
    if dirty and not a.allow_dirty:
        sys.exit("refused: uncommitted changes in " + ", ".join(os.path.relpath(f, strip) for f in dirty[:8]) + " (commit first, or --allow-dirty: it is written into the manifest)")
    dest = os.path.join(a.out, a.name)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    entries = []
    for f in files:
        rel = os.path.relpath(f, strip).replace("\\", "/")
        os.makedirs(os.path.dirname(os.path.join(dest, "code", rel)), exist_ok=True)
        shutil.copy(f, os.path.join(dest, "code", rel))
        entries.append({"file": "code/" + rel, "bytes": os.path.getsize(f), "sha256": sha(f), "source": f.replace("\\", "/")})
    commit = git(root, "rev-parse", "HEAD")
    manifest = {"name": a.name, "title": a.title or a.name, "date": datetime.date.today().isoformat(), "source_commit": commit,
                "sources_uncommitted": [os.path.relpath(f, strip) for f in dirty], "files": entries, "total_bytes": total,
                "numbers": a.numbers, "command": a.command, "weights": a.weights}
    json.dump(manifest, open(os.path.join(dest, "MANIFEST.json"), "w", encoding="utf-8"), indent=1)
    lines = ["# %s" % (a.title or a.name), "",
             "A winner folder: the source of the system, small enough to hand over; **no weights**. Written by `winner.py` on %s from commit `%s`%s." % (
                 manifest["date"], commit[:10] or "unknown", " (with uncommitted changes in %d file(s): see MANIFEST.json)" % len(dirty) if dirty else ""), "",
             "## What is here", "", "`code/` holds %d file(s), %d bytes. `MANIFEST.json` lists every file with its sha256; `python winner.py verify <this folder>` re-checks them." % (len(entries), total), "",
             "## Numbers to reproduce", "", a.numbers or "PENDING", "", "## How to run", "", "```", a.command or "PENDING", "```", "",
             "## Weights", "", a.weights or "PENDING: where the checkpoints are (they are not in git because of their size)", ""]
    open(os.path.join(dest, "README.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("%s: %d file(s), %d bytes, commit %s%s" % (dest, len(entries), total, commit[:10] or "none", ", DIRTY sources" if dirty else ""))
    return dest


def verify(folder):
    m = json.load(open(os.path.join(folder, "MANIFEST.json"), encoding="utf-8"))
    bad, drift, gone = [], [], []
    for e in m["files"]:
        p = os.path.join(folder, e["file"])
        if not os.path.exists(p) or sha(p) != e["sha256"]:
            bad.append(e["file"])
        s = e.get("source")
        if s and os.path.exists(s) and sha(s) != e["sha256"]:
            drift.append(e["file"])
        elif s and not os.path.exists(s):
            gone.append(e["file"])
    print("%s: %d file(s); %d changed inside the folder; %d differ from their source now (drift); %d whose source is gone (moved or removed)"
          % (folder, len(m["files"]), len(bad), len(drift), len(gone)))
    for f in bad:
        print("  changed:", f)
    for f in drift:
        print("  source has moved on:", f)
    for f in gone:
        print("  source is gone:", f)
    return len(bad)


def selftest():
    d = tempfile.mkdtemp(prefix="winner_selftest_")
    try:
        os.makedirs(os.path.join(d, "src", "sub"))
        open(os.path.join(d, "src", "a.py"), "w").write("print('a')\n")
        open(os.path.join(d, "src", "sub", "b.py"), "w").write("print('b')\n")
        ns = argparse.Namespace(name="w1", src=[os.path.join(d, "src", "**", "*.py")], strip=os.path.join(d, "src"), out=os.path.join(d, "out"), limit=1_000_000,
                                title="Test winner", numbers="0.69 / 0.70", command="python a.py", weights="runs/x", allow_dirty=True)
        dest = export(ns)
        assert os.path.exists(os.path.join(dest, "code", "sub", "b.py")) and os.path.exists(os.path.join(dest, "README.md"))
        assert verify(dest) == 0
        open(os.path.join(d, "src", "a.py"), "w").write("print('changed')\n")          # drift is reported, the folder itself stays valid
        assert verify(dest) == 0
        os.rename(os.path.join(d, "src", "sub", "b.py"), os.path.join(d, "src", "b2.py"))   # a moved source is named, not passed over in silence
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert verify(dest) == 0
        assert "source is gone: code/sub/b.py" in out.getvalue(), out.getvalue()
        os.rename(os.path.join(d, "src", "b2.py"), os.path.join(d, "src", "sub", "b.py"))
        open(os.path.join(d, "src", "w.pt"), "wb").write(b"0")
        ns.src = [os.path.join(d, "src", "*")]
        try:
            export(ns)
            raise AssertionError("a weight file must be refused")
        except SystemExit:
            pass
        print("selftest OK")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("export")
    e.add_argument("name"); e.add_argument("--src", action="append", required=True); e.add_argument("--strip"); e.add_argument("--out", required=True, help="the project's winner folder, e.g. bev_fusion_lane_center/winner (D139 / D140)")
    e.add_argument("--limit", type=int, default=1_000_000); e.add_argument("--title"); e.add_argument("--numbers"); e.add_argument("--command"); e.add_argument("--weights")
    e.add_argument("--allow-dirty", action="store_true")
    v = sp.add_parser("verify"); v.add_argument("folder")
    sp.add_parser("selftest")
    x = ap.parse_args()
    if x.cmd == "export":
        export(x)
    elif x.cmd == "verify":
        sys.exit(1 if verify(x.folder) else 0)
    else:
        selftest()
