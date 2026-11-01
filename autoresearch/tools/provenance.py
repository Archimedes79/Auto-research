"""Provenance of a result package (design: autoresearch/reproducibility.md): which autoGT (version, signature, mode), which evaluator
code, which frames, which model and environment made a card, so that the experiment can be reproduced later. Standard library only.

    python3 provenance.py collect --gt-key 0.5_4_15efbde2d1 --split val --frames tokens.txt --evaluator bev_fusion/evaluation
            --ckpt runs/x/ckpt.pt --config runs/x/config.json --run x --seed 0 > provenance.json
        the `provenance` block; the writer of a card puts it under card["provenance"] (and the launcher into run.json). A card writer calls
        collect() from its own interpreter (the environment block describes the interpreter that runs collect: torch and CUDA are filled
        when torch is already imported there); the command line describes the shell's python
    python3 provenance.py repro <run id> [--card card.json]     what to check out and run to reproduce a run (prints, changes nothing)
    python3 provenance.py compare a.json b.json                 exit 0 comparable, 1 they differ, 2 not shown comparable (a value is unknown)
    python3 provenance.py --selftest

A missing value reads "unknown", never a guess: no frames, no git, a path outside the repository all give unknown, and a dirty value that is
unknown blocks like a dirty one. Two cards are comparable when GT version, GT signature, evaluator code hash and frame hash agree; a value
that is unknown on one side is listed as "cannot tell"; a card whose evaluator code differs from its commit is a blocker. The evaluator hash
covers the source files (.py .sh .ps1 .json .yaml .yml) under the given paths, sorted by posix path, line endings normalised, so it agrees
between Windows and WSL. A checkpoint over 64 MB is hashed by its first and last 4 MB plus its size (marked p): a change in the middle is not
seen. `ckpt_hash` is deliberately not a comparison key: cards of different models are the point of a comparison.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import project as P  # noqa: E402

SCHEMA = 1
CODE_SUFFIX = (".py", ".sh", ".ps1", ".json", ".yaml", ".yml")
COMPARE_KEYS = (("gt", "version"), ("gt", "signature"), ("evaluator", "code_hash"), ("data", "frame_hash"), ("data", "manifest_hash"))
UNKNOWN = "unknown"


def git(repo, *args):
    """stdout of a git call, None when git fails (missing, timeout, not a repository, a path outside it): a failure never reads as clean"""
    try:
        r = subprocess.run(["git", "--no-optional-locks", "-C", repo, *args], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def repo_root(start=None):
    out = git(start or HERE, "rev-parse", "--show-toplevel")
    return os.path.normpath(out.strip()) if out and out.strip() else os.path.dirname(os.path.dirname(HERE))   # tools/ -> autoresearch/ -> repository


def file_hash(path, partial_over=64 << 20):
    """12 hex of the content; a file over 64 MB (a checkpoint) is hashed by its first and last 4 MB plus its size and marked with a p"""
    size, h = os.path.getsize(path), hashlib.sha256()
    with open(path, "rb") as f:
        if size <= partial_over:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(f.read(4 << 20))
            f.seek(-(4 << 20), 2)
            h.update(f.read(4 << 20))
            h.update(str(size).encode())
    return h.hexdigest()[:12] + ("p" if size > partial_over else "")


def source_files(paths):
    out = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"evaluator path not found: {p}")
        if os.path.isfile(p):
            out.append(p)
            continue
        for dp, dn, fn in os.walk(p):
            dn[:] = sorted(x for x in dn if x not in ("__pycache__", ".git"))
            out += [os.path.join(dp, f) for f in sorted(fn) if f.endswith(CODE_SUFFIX)]
    if not out:
        raise ValueError(f"no source file ({', '.join(CODE_SUFFIX)}) under the evaluator paths {list(paths)}")
    return sorted(set(out))


def tree_hash(paths, base):
    """12 hex over the posix-relative names (sorted as such) and the bytes, CRLF read as LF, of the source files under `paths`"""
    h = hashlib.sha256()
    for rel, f in sorted((os.path.relpath(f, base).replace("\\", "/"), f) for f in source_files(paths)):
        with open(f, "rb") as fh:
            h.update(rel.encode() + b"\0" + fh.read().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:12]


def dirty_files(repo, paths):
    """the source files under `paths` that differ from HEAD (modified, untracked, renamed, or ignored but part of the hash); UNKNOWN when it cannot be told"""
    rel = [os.path.relpath(p, repo).replace("\\", "/") for p in paths]
    if any(r == ".." or r.startswith("../") for r in rel):
        return UNKNOWN
    out = git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", *rel)
    ign = git(repo, "ls-files", "-o", "-i", "--exclude-standard", "-z", "--", *rel)
    if out is None or ign is None:
        return UNKNOWN
    names, parts, i = [], out.split("\0"), 0
    while i < len(parts):
        e = parts[i]
        i += 1
        if len(e) < 4:
            continue
        names.append(e[3:])
        if e[0] in "RC" and i < len(parts):          # a rename or copy carries the second name in the next entry
            names.append(parts[i])
            i += 1
    names += [n for n in ign.split("\0") if n]
    return sorted(n for n in set(names) if n.endswith(CODE_SUFFIX))


def evaluator_block(paths, repo, settings=None):
    paths = [os.path.abspath(p) for p in paths]
    tree = tree_hash(paths, repo)                    # raises when a path is missing or holds no source file
    commit = (git(repo, "rev-parse", "--short=10", "HEAD") or "").strip() or UNKNOWN
    return {"commit": commit, "dirty": dirty_files(repo, paths), "code_hash": tree,
            "paths": [os.path.relpath(p, repo).replace("\\", "/") for p in paths], "settings": settings or {}}


def gt_block(key, repo, data_root=None):
    """key = the GT cache key of the run (e.g. 0.5_4_15efbde2d1): the last part is the generator signature; the store's index.json names the version"""
    sig = key.split("_")[-1] if key else ""
    out = {"key": key or UNKNOWN, "signature": sig or UNKNOWN, "version": UNKNOWN}
    p = os.path.join(data_root or os.path.join(repo, "data", "autogt"), sig, "index.json") if sig else ""
    if p and os.path.isfile(p):
        try:
            d = json.load(open(p, encoding="utf-8"))
            out.update(version=d.get("gt_version", UNKNOWN), dataset=d.get("version", UNKNOWN), res=d.get("res"), ss=d.get("ss"),
                       options=d.get("options", {}), index_hash=file_hash(p))
        except (OSError, ValueError, AttributeError):
            out["error"] = f"unreadable index {p}"
    return out


def frames_block(split, tokens):
    toks = sorted(set(tokens))
    return {"split": split or UNKNOWN, "n": len(toks),
            "frame_hash": hashlib.sha256("\n".join(toks).encode()).hexdigest()[:12] if toks else UNKNOWN}


def dataset_block(version, repo):
    """the identity of the raw data: the hash of datasets/NuScenes/manifest.json (archives, tables, sensor-folder fingerprints), the version key and the counts"""
    out = {"dataset": version or UNKNOWN, "manifest_hash": UNKNOWN}
    p = os.path.join(repo, "datasets", "NuScenes", "manifest.json")
    if os.path.isfile(p):
        out["manifest_hash"] = file_hash(p)
        try:
            d = json.load(open(p, encoding="utf-8")).get("datasets", {}).get(version or "", {})
            out.update(scenes=d.get("scenes"), samples=d.get("samples"))
        except (OSError, ValueError, AttributeError):
            out["manifest_error"] = "unreadable manifest"
    return out


def subject_block(run=None, ckpt=None, config=None, step=None, run_dir=None, repo=None):
    out = {"run": run or UNKNOWN}
    if ckpt and os.path.isfile(ckpt):
        out.update(ckpt=os.path.basename(ckpt), ckpt_hash=file_hash(ckpt))
    if config and os.path.isfile(config):
        out["config_hash"] = file_hash(config)
    if step is not None:
        out["step"] = step
    g = os.path.join(run_dir, "gt.json") if run_dir else ""
    if g and os.path.isfile(g):
        out["train_gt"] = gt_block(json.load(open(g, encoding="utf-8")).get("gt_signature"), repo or repo_root())
    return out


def env_block():
    from importlib import metadata

    def ver(name):
        try:
            return metadata.version(name)
        except Exception:
            return UNKNOWN
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=20).stdout.strip().splitlines()[0]
    except Exception:
        gpu = UNKNOWN
    tv = getattr(sys.modules.get("torch"), "version", None)          # only when the caller has torch loaded: no heavy import here
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": ver("torch"), "cuda": getattr(tv, "cuda", None) or UNKNOWN,
            "numpy": ver("numpy"), "gpu": gpu, "host": platform.node()}


def collect(gt_key=None, split=None, tokens=(), evaluator=(), run=None, ckpt=None, config=None, step=None, run_dir=None, seed=None,
            settings=None, repo=None, data_root=None, dataset=None):
    """dataset = the nuScenes version key (v1.0-trainval, v1.0-mini); default: the one the GT store names"""
    repo = repo or repo_root()
    gt = gt_block(gt_key, repo, data_root)
    data = dict(frames_block(split, tokens), **dataset_block(dataset or gt.get("dataset"), repo))
    return {"schema": SCHEMA, "made": dt.datetime.now().isoformat(timespec="seconds"), "gt": gt,
            "evaluator": evaluator_block(evaluator, repo, settings) if evaluator else {"commit": UNKNOWN, "dirty": UNKNOWN, "code_hash": UNKNOWN},
            "data": data, "subject": subject_block(run, ckpt, config, step, run_dir, repo), "environment": env_block(),
            "seed": seed if seed is not None else UNKNOWN}


def _get(d, keys):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _prov(x):
    p = (x or {}).get("provenance", x) if isinstance(x, dict) else None
    return p if isinstance(p, dict) else {}


def compare(a, b):
    """-> (differences, cannot_tell): a difference in a known value or a dirty / unchecked evaluator blocks a comparison, an unknown value cannot be told"""
    a, b = _prov(a), _prov(b)
    diff, unknown = [], []
    for k in COMPARE_KEYS:
        x, y = _get(a, k), _get(b, k)
        if x in (None, UNKNOWN) or y in (None, UNKNOWN):
            unknown.append(".".join(k))
        elif x != y:
            diff.append(f"{'.'.join(k)}: {x} != {y}")
    for side, d in (("first", a), ("second", b)):
        if d.get("error"):
            diff.append(f"the provenance of the {side} card could not be collected: {d['error']}")
            continue
        dirty = _get(d, ("evaluator", "dirty"))
        if dirty is None:
            unknown.append(f"evaluator.dirty ({side} card)")
        elif dirty == UNKNOWN:
            diff.append(f"the evaluator of the {side} card was not checked against its commit")
        elif dirty:
            diff.append(f"the evaluator of the {side} card differs from its commit ({len(dirty)} file(s), e.g. {dirty[0]})")
    return diff, unknown


def verdict(diff, unknown):
    """-> (text, exit code): 0 comparable, 1 they differ, 2 not shown comparable"""
    lines = ["differs: " + x for x in diff] + ["cannot tell: " + x for x in unknown]
    if diff:
        return "\n".join(lines), 1
    if unknown:
        return "\n".join(lines + ["not shown comparable"]), 2
    return "comparable", 0


def repro(run_id, card_path=None):
    rj = os.path.join(P.path("run_records"), run_id, "run.json")
    if not os.path.isfile(rj):
        sys.exit(f"no run record {rj}")
    r = json.load(open(rj, encoding="utf-8"))
    from importlib import metadata
    try:
        torch_here = metadata.version("torch")
    except Exception:
        torch_here = UNKNOWN
    L = [f"run {run_id}: status {r.get('status')}, exit {r.get('exit')}, ended {r.get('ended')}",
         f"code: tag {r.get('tag')} (commit {r.get('commit')}, code hash {r.get('code_hash')}); check it out with: git worktree add ../repro_{run_id} {r.get('tag')}"]
    drift = r.get("code_drift") or {}
    if drift.get("n"):
        L.append(f"WARNING: {drift['n']} file(s) differed from the commit when it ran: {', '.join(drift.get('paths', [])[:5])}")
    L += [f"command: {r.get('command')}   (cwd {r.get('cwd')})",
          f"recorded: python {r.get('python')}, {r.get('gpu')}; this machine: python {platform.python_version()}, torch {torch_here}"]
    prov = json.load(open(card_path, encoding="utf-8")).get("provenance") if card_path else None
    if card_path and not prov:
        L.append(f"the card {card_path} has no provenance block: GT version, evaluator hash and frames are unknown")
    elif prov:
        g, e, d, s = (prov.get(k) or {} for k in ("gt", "evaluator", "data", "subject"))
        v = str(g.get("version", ""))
        tag = "autogt/" + v if v.startswith("v") else "<autogt tag>"
        store = os.path.join(repo_root(), "data", "autogt", str(g.get("signature")))
        L += [f"GT: version {g.get('version')} (signature {g.get('signature')}, dataset {g.get('dataset')}, options {g.get('options')}), code: git show {tag}",
              (f"the GT store {store} is present" if os.path.isdir(store) else
               f"the GT store {store} is MISSING: rebuild it with the code of {tag} (git worktree add ../autogt_{v} {tag}) and the options above"),
              f"evaluator: commit {e.get('commit')}, code hash {e.get('code_hash')}" + (f", NOT CLEAN or unchecked: {e.get('dirty')}" if e.get("dirty") else ""),
              f"frames: split {d.get('split')}, {d.get('n')} frames, hash {d.get('frame_hash')}",
              f"model: run {s.get('run')}, checkpoint {s.get('ckpt')} ({s.get('ckpt_hash')}), step {s.get('step')}, config {s.get('config_hash')}; "
              f"trained on GT {_get(s, ('train_gt', 'version'))}",
              f"seed {prov.get('seed')}; recorded torch {_get(prov, ('environment', 'torch'))}, CUDA {_get(prov, ('environment', 'cuda'))}"]
    else:
        L.append("no card given: pass --card <card.json> for the GT version, the evaluator hash and the frames")
    print("\n".join(L))


def selftest():
    import contextlib
    import io
    tmp = tempfile.mkdtemp()
    old_env = os.environ.get("AR_PROJECT")
    try:
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")

        def run(*a):
            subprocess.run(["git", "-C", tmp, *a], check=True, capture_output=True, env=env)
        os.makedirs(os.path.join(tmp, "ev"))
        os.makedirs(os.path.join(tmp, "data", "autogt", "abc123"))
        open(os.path.join(tmp, "ev", "m.py"), "w").write("x = 1\n")
        json.dump({"signature": "abc123", "gt_version": "v5", "version": "v1.0-trainval", "res": 0.5, "ss": 4, "options": {"boxes_at": "camera"}},
                  open(os.path.join(tmp, "data", "autogt", "abc123", "index.json"), "w"))
        os.makedirs(os.path.join(tmp, "datasets", "NuScenes"))
        mf = os.path.join(tmp, "datasets", "NuScenes", "manifest.json")
        json.dump({"datasets": {"v1.0-trainval": {"scenes": 850, "samples": 34149}}}, open(mf, "w"))
        run("init", "-q")
        run("add", "ev")
        run("commit", "-q", "-m", "x")
        ck = os.path.join(tmp, "ck.pt")
        open(ck, "wb").write(b"abc")
        ev = [os.path.join(tmp, "ev")]
        a = collect("0.5_4_abc123", "val", ["t2", "t1"], ev, run="r", ckpt=ck, step=5, seed=1, repo=tmp)
        assert a["gt"]["version"] == "v5" and a["gt"]["signature"] == "abc123" and a["gt"]["options"] == {"boxes_at": "camera"}, a["gt"]
        assert a["evaluator"]["dirty"] == [] and a["data"]["n"] == 2 and a["subject"]["ckpt_hash"] and a["seed"] == 1
        assert a["data"]["frame_hash"] == frames_block("val", ["t1", "t2", "t1"])["frame_hash"], "the frame hash ignores order and duplicates"
        assert frames_block("val", [])["frame_hash"] == UNKNOWN and frames_block("val", [])["n"] == 0, "no frames is unknown, not the hash of nothing"
        assert a["data"]["dataset"] == "v1.0-trainval" and a["data"]["scenes"] == 850 and a["data"]["manifest_hash"] != UNKNOWN, a["data"]
        json.dump({"datasets": {"v1.0-trainval": {"scenes": 851, "samples": 34149}}}, open(mf, "w"))
        m2 = collect("0.5_4_abc123", "val", ["t2", "t1"], [os.path.join(tmp, "ev")], repo=tmp)
        assert any("data.manifest_hash" in x for x in compare(a, m2)[0]), "another manifest = other raw data = not comparable"
        json.dump({"datasets": {"v1.0-trainval": {"scenes": 850, "samples": 34149}}}, open(mf, "w"))
        assert dataset_block("v1.0-trainval", os.path.join(tmp, "nowhere"))["manifest_hash"] == UNKNOWN, "no manifest reads unknown"
        assert gt_block("0.5_4_zzz", tmp)["version"] == UNKNOWN, "an unknown store reads unknown"
        open(os.path.join(tmp, "data", "autogt", "abc123", "index.json"), "w").write("{torn")
        assert gt_block("0.5_4_abc123", tmp)["version"] == UNKNOWN and "error" in gt_block("0.5_4_abc123", tmp), "a torn index does not crash"
        # the evaluator hash: a changed file, CRLF, a path that does not exist
        h0 = tree_hash(ev, tmp)
        open(os.path.join(tmp, "ev", "m.py"), "wb").write(b"x = 1\r\n")
        assert tree_hash(ev, tmp) == h0, "CRLF and LF hash alike (Windows and WSL agree)"
        open(os.path.join(tmp, "ev", "m.py"), "w").write("x = 2\n")
        b = collect("0.5_4_abc123", "val", ["t1", "t2"], ev, repo=tmp)
        assert b["evaluator"]["dirty"] == ["ev/m.py"] and b["evaluator"]["code_hash"] != a["evaluator"]["code_hash"], "a changed evaluator file shows"
        os.makedirs(os.path.join(tmp, "docs"))
        open(os.path.join(tmp, "docs", "README.md"), "w").write("no source here\n")
        for bad in ([os.path.join(tmp, "ev_typo")], [os.path.join(tmp, "docs")]):
            try:
                evaluator_block(bad, tmp)
                raise AssertionError("a missing path or a folder without source files must raise")
            except (FileNotFoundError, ValueError):
                pass
        # dirty: a name with spaces, a rename, an untracked file, an ignored but hashed file; non-source files do not count
        run("checkout", "--", "ev/m.py")
        open(os.path.join(tmp, "ev", "a b.py"), "w").write("y = 1\n")
        run("add", "ev")
        run("commit", "-q", "-m", "space")
        assert dirty_files(tmp, ev) == [], dirty_files(tmp, ev)
        run("mv", "ev/a b.py", "ev/c d.py")
        assert dirty_files(tmp, ev) == ["ev/a b.py", "ev/c d.py"], dirty_files(tmp, ev)
        run("commit", "-q", "-m", "renamed")
        open(os.path.join(tmp, "ev", "README.md"), "w").write("doc\n")
        assert dirty_files(tmp, ev) == [], "a file that is not hashed is not dirty"
        open(os.path.join(tmp, ".gitignore"), "w").write("ev/ig.py\n")
        open(os.path.join(tmp, "ev", "ig.py"), "w").write("z = 1\n")
        assert dirty_files(tmp, ev) == ["ev/ig.py"], "an ignored file that is part of the hash is not in the commit"
        assert dirty_files(tmp, [os.path.join(os.path.dirname(tmp), "elsewhere")]) == UNKNOWN, "a path outside the repository is unknown"
        nogit = tempfile.mkdtemp()
        try:
            assert dirty_files(nogit, [os.path.join(nogit, "x")]) == UNKNOWN, "no repository: unknown, not clean"
        finally:
            shutil.rmtree(nogit, ignore_errors=True)
        assert collect(repo=tmp)["evaluator"]["dirty"] == UNKNOWN, "a card made without an evaluator is not clean"
        # compare and its exit codes
        diff, unk = compare(a, b)
        assert any("code_hash" in x for x in diff) and any("differs from its commit" in x for x in diff), diff
        diff, unk = compare(a, dict(a))
        assert diff == [] and unk == [] and verdict(diff, unk) == ("comparable", 0), (diff, unk)
        c = dict(a, gt=dict(a["gt"], version="v6"))
        assert any("gt.version" in x for x in compare(a, c)[0]) and verdict(*compare(a, c))[1] == 1
        d = dict(a, gt=gt_block("", tmp))
        diff, unk = compare(a, d)
        assert diff == [] and "gt.version" in unk and verdict(diff, unk)[1] == 2, "an unknown value cannot be told: exit 2, not 0"
        assert verdict(*compare({}, {}))[1] == 2 and verdict(*compare({"provenance": None}, {"provenance": None}))[1] == 2, "no provenance at all: not shown comparable"
        assert compare({"provenance": a}, {"provenance": a}) == ([], []), "a whole card is accepted"
        e = dict(a, evaluator=dict(a["evaluator"], dirty=UNKNOWN))
        assert any("not checked" in x for x in compare(a, e)[0]), "an unchecked evaluator blocks"
        err = {"schema": 1, "error": "collector failed"}
        assert any("could not be collected" in x for x in compare(a, err)[0]) and verdict(*compare(a, err))[1] == 1, "an error-only block blocks like decide.provenance_check"
        assert verdict(*compare(a, dict(a, evaluator={})))[1] == 2, "a block without the evaluator is not shown comparable"
        # checkpoints: the partial hash sees the head and the tail, marks itself, and the small one is whole
        big = os.path.join(tmp, "big.bin")
        with open(big, "wb") as f:
            f.seek((64 << 20) + 10)
            f.write(b"x")
        hb = file_hash(big)
        with open(big, "r+b") as f:
            f.seek(-1, 2)
            f.write(b"y")
        assert hb.endswith("p") and file_hash(big) != hb and not file_hash(ck).endswith("p"), "a change in the tail shows"
        # repro on a fake run record and a card
        proj = os.path.join(tmp, "proj")
        os.makedirs(os.path.join(proj, "rr", "r1"))
        json.dump({"run_records": "rr"}, open(os.path.join(proj, "project.json"), "w"))
        json.dump({"id": "r1", "status": "ok", "exit": 0, "tag": "tg/r1", "commit": "abc", "code_hash": "h", "command": "bash e.sh", "cwd": "/x",
                   "python": "3.10", "code_drift": {"n": 1, "paths": ["a.py"]}}, open(os.path.join(proj, "rr", "r1", "run.json"), "w"))
        cardf = os.path.join(tmp, "card.json")
        json.dump({"provenance": a}, open(cardf, "w"))
        os.environ["AR_PROJECT"] = proj
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            repro("r1", cardf)
            repro("r1")
        txt = buf.getvalue()
        assert "tag tg/r1" in txt and "WARNING: 1 file(s)" in txt and "autogt/v5" in txt and "frames: split val, 2 frames" in txt and "no card given" in txt, txt
        print("provenance selftest passed")
    finally:
        if old_env is None:
            os.environ.pop("AR_PROJECT", None)
        else:
            os.environ["AR_PROJECT"] = old_env
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["collect", "repro", "compare"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gt-key")
    ap.add_argument("--split")
    ap.add_argument("--frames", help="file with one frame token per line")
    ap.add_argument("--evaluator", nargs="*", default=[])
    ap.add_argument("--run")
    ap.add_argument("--ckpt")
    ap.add_argument("--config")
    ap.add_argument("--step", type=int)
    ap.add_argument("--seed")
    ap.add_argument("--run-dir")
    ap.add_argument("--card")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "collect":
        toks = open(a.frames, encoding="utf-8").read().split() if a.frames else []
        print(json.dumps(collect(a.gt_key, a.split, toks, a.evaluator, a.run, a.ckpt, a.config, a.step, a.run_dir, a.seed), indent=1))
    elif a.cmd == "repro" and a.args:
        repro(a.args[0], a.card)
    elif a.cmd == "compare" and len(a.args) == 2:
        text, code = verdict(*compare(*(json.load(open(x, encoding="utf-8")) for x in a.args)))
        print(text)
        sys.exit(code)
    else:
        ap.error("collect | repro <run id> | compare a.json b.json | --selftest")


if __name__ == "__main__":
    main()
