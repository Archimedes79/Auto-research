"""Provenance of a result package (design: autoresearch/reproducibility.md): which autoGT (version, signature, mode), which evaluator
code, which frames, which model and environment made a card, so that the experiment can be reproduced later. Standard library only.

    python3 provenance.py collect --gt-key 0.5_4_15efbde2d1 --split val --frames tokens.txt --evaluator bev_fusion/evaluation
            --ckpt runs/x/ckpt.pt --config runs/x/config.json --run x --seed 0 > provenance.json
        the `provenance` block; the writer of a card puts it under card["provenance"] (and the launcher into run.json). A card writer calls
        collect() from its own interpreter (the environment block describes the interpreter that runs collect: torch and CUDA are filled
        when torch is already imported there); the command line describes the shell's python
    python3 provenance.py manifest <run id> [--run-dir d]       the provenance of a TRAINING run from its run record and run folder (GT it trained on,
                                                                checkpoint, config, code tag, model bundle) -> <problem>/results/<id>/provenance.json
    python3 provenance.py repro <run id> [--card card.json] [--repeat [--dir d]]     what to check out and run to reproduce a run (prints, changes nothing; reads that manifest); --repeat checks the tag out in a worktree under temp/repro/ and prints the repeat command
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
import re
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
BUNDLE_ROOTS = ("model.py", "losses.py")   # the network and its losses; the bundle is every module of the same folder they import, transitively
BUNDLE_EXCLUDE = ("train.py", "nuscenes_dataset.py", "feature_cache.py")   # the training loop and the data loaders are not part of the network: not followed, but reported when the bundle imports them


def git(repo, *args, timeout=60):
    """stdout of a git call, None when git fails (missing, timeout, not a repository, a path outside it): a failure never reads as clean"""
    try:
        r = subprocess.run(["git", "--no-optional-locks", "-C", repo, *args], capture_output=True, text=True, timeout=timeout)
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
    pj = os.path.join(results_dir(run_id), "provenance.json")
    if not card_path and os.path.isfile(pj):  # the run's own manifest (`manifest <run id>`): GT, model and settings of the training
        prov = json.load(open(pj, encoding="utf-8"))
        L.append(f"manifest of the run: {pj} (made {prov.get('made')})")
        rs, bd = prov.get("run_status") or {}, prov.get("bundle") or {}
        if rs and not rs.get("final"):
            L.append(f"WARNING: the run is not final (status {rs.get('status')}, exit {rs.get('exit')}): its checkpoint is not a finished model")
        for key, what in (("missing_roots", "root modules missing"), ("unparsed", "files that do not parse (their imports are unknown)"), ("imports_excluded", "the network imports the loop or a loader"),
                          ("external_local", "repository code outside the bundle (its version is the run's GT version / tag)"), ("dynamic_imports", "files that import by name at run time")):
            if bd.get(key):
                L.append(f"bundle: {what}: {', '.join(bd[key][:6])}")
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
              (f"evaluator: commit {e.get('commit')}, code hash {e.get('code_hash')}" + (f", NOT CLEAN or unchecked: {e.get('dirty')}" if e.get("dirty") else "")) if e else
              f"training code: tag {_get(prov, ('code', 'tag'))}, commit {_get(prov, ('code', 'commit'))}",
              (f"model bundle (network + losses): {len(_get(prov, ('bundle', 'files')) or [])} files, hash {_get(prov, ('bundle', 'code_hash'))}"
               + (f", copied to {_get(prov, ('bundle', 'copied_to'))}/" if _get(prov, ("bundle", "copied_to")) else "")) if prov.get("bundle") else "",
              f"frames: split {d.get('split')}, {d.get('n')} frames, hash {d.get('frame_hash')}",
              f"model: run {s.get('run')}, checkpoint {s.get('ckpt')} ({s.get('ckpt_hash')}), step {s.get('step')}, config {s.get('config_hash')}; "
              f"trained on GT {_get(s, ('train_gt', 'version'))}",
              f"seed {prov.get('seed')}; recorded torch {_get(prov, ('environment', 'torch'))}, CUDA {_get(prov, ('environment', 'cuda'))}"]
    else:
        L.append("no card given: pass --card <card.json> for the GT version, the evaluator hash and the frames")
    print("\n".join(x for x in L if x))


def import_closure(code_dir, roots=BUNDLE_ROOTS, exclude=(), info=None):
    """the files (relative, sorted) of code_dir that the root modules import, transitively: the network and its losses. Only modules found under
    code_dir count; a parent package of an imported module counts too (python runs its __init__). A module in `exclude` (the training loop, the
    data loaders) is not followed and not in the result. `info` (a dict, optional) collects what the closure cannot say by itself: "skipped"
    ("importer -> excluded module"), "unparsed" (files that do not parse: their imports are unknown), "external" (top-level names of absolute
    imports that are neither in code_dir nor in the standard library), "dynamic" (files that import by name at run time)"""
    import ast
    info = info if info is not None else {}
    for k in ("skipped", "unparsed", "external", "dynamic"):
        info.setdefault(k, set())
    mods = {}
    for dp, dn, fn in os.walk(code_dir):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn:
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(dp, f), code_dir).replace("\\", "/")
                name = rel[:-3].replace("/", ".")
                mods[name[:-9] if name.endswith(".__init__") else name] = rel
    byfile = {v: k for k, v in mods.items()}
    todo, seen = [r for r in roots if r in byfile], set()
    std = getattr(sys, "stdlib_module_names", ())
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        try:
            tree = ast.parse(open(os.path.join(code_dir, rel), encoding="utf-8-sig").read())   # utf-8-sig: a BOM (Notepad) is no syntax error
        except (OSError, SyntaxError, ValueError):
            info["unparsed"].add(rel)
            continue
        name = byfile[rel]
        pkg = name if rel.endswith("__init__.py") else (name.rsplit(".", 1)[0] if "." in name else "")
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and ((isinstance(n.func, ast.Name) and n.func.id == "__import__") or (isinstance(n.func, ast.Attribute) and n.func.attr == "import_module")):
                info["dynamic"].add(rel)
            cands, absolute = [], False
            if isinstance(n, ast.Import):
                cands, absolute = [a.name for a in n.names], True
            elif isinstance(n, ast.ImportFrom):
                base = n.module or ""
                absolute = not n.level
                if n.level:
                    up = pkg.split(".") if pkg else []
                    up = up[:len(up) - (n.level - 1)] if n.level > 1 else up
                    base = ".".join(up + ([n.module] if n.module else []))
                cands = [base] + [f"{base}.{a.name}" if base else a.name for a in n.names]
            for c in cands:
                parts = c.split(".") if c else []
                prefixes = [".".join(parts[:i]) for i in range(1, len(parts) + 1)]
                hit = [p for p in prefixes if p in mods]
                if absolute and not hit and parts and parts[0] not in std:
                    info["external"].add(parts[0])
                for p in hit:
                    if mods[p] in exclude:
                        info["skipped"].add(f"{rel} -> {mods[p]}")
                    elif mods[p] not in seen:
                        todo.append(mods[p])
    return sorted(seen)


def _hash_files(root, files):
    h = hashlib.sha256()
    for rel in files:
        h.update(rel.encode() + b"\0" + open(os.path.join(root, rel), "rb").read().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:12]


def bundle(code_dir, roots=BUNDLE_ROOTS, into=None, config=None, exclude=BUNDLE_EXCLUDE):
    """The model bundle of a run: the network and loss modules (import_closure of `roots` under code_dir) with a hash over their text (line endings
    normalised), and the run's config. into = a folder to copy them to (results/<id>/bundle/), so the model can be rebuilt from that folder and the
    checkpoint alone; it is made next to the target, checked against the hash and moved into place, and an existing non-empty folder is left alone.
    What the files do NOT carry is listed, never hidden: missing_roots, unparsed (their imports are unknown), imports_excluded (the network imports
    the loop or a loader), external_local (repository folders such as autogt that the code imports: they are not in the bundle, their version is the
    GT version of the run), dynamic_imports. Nothing found reads "unknown", never an empty hash"""
    if not code_dir or not os.path.isdir(code_dir):
        return {"code_hash": UNKNOWN, "files": [], "source": UNKNOWN}
    info = {}
    files = import_closure(code_dir, roots, exclude, info)
    if not files:
        return {"code_hash": UNKNOWN, "files": [], "source": code_dir, "note": f"none of {list(roots)} found"}
    local = sorted(t for t in info["external"] if any(os.path.isdir(os.path.join(b, t)) for b in (repo_root(), P.root())))
    out = {"code_hash": _hash_files(code_dir, files), "files": files, "roots": list(roots), "missing_roots": [r for r in roots if r not in files],
           "excluded": list(exclude), "imports_excluded": sorted(info["skipped"]), "unparsed": sorted(info["unparsed"]), "external_local": local,
           "dynamic_imports": sorted(info["dynamic"]), "source": os.path.basename(os.path.dirname(code_dir.rstrip("/\\"))) + "/" + os.path.basename(code_dir.rstrip("/\\"))}
    if config and os.path.isfile(config):
        out["config_hash"] = file_hash(config)
    if into:
        if os.path.isdir(into) and os.listdir(into):
            out["copy_skipped"] = f"{os.path.basename(into)}/ exists and is not empty: left as it is"
        else:
            tmp = into.rstrip("/\\") + ".tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            for rel in files:
                os.makedirs(os.path.dirname(os.path.join(tmp, rel)) or tmp, exist_ok=True)
                shutil.copy2(os.path.join(code_dir, rel), os.path.join(tmp, rel))
            if config and os.path.isfile(config):
                shutil.copy2(config, os.path.join(tmp, "config.json"))
            if _hash_files(tmp, files) != out["code_hash"]:
                shutil.rmtree(tmp, ignore_errors=True)
                out["copy_skipped"] = "the copy did not read back equal"
            else:
                if os.path.isdir(into):
                    os.rmdir(into)   # empty (checked above)
                os.replace(tmp, into)
                out["copied_to"] = os.path.basename(into)
    return out


def results_dir(run_id):
    """the result folder of a run: <problem>/<project.json "results", default results>/<run id>. Text records go to git (package.json, provenance.json);
    the heavy files stay in the run folder and are only listed in the package"""
    return os.path.join(P.root(), P.cfg_of(P.root()).get("results", "results"), run_id)


def repro_prepare(run_id, dest=None):
    """Make a run repeatable without touching the current checkout: the run's code tag is checked out (detached) in a worktree under temp/repro/<id>
    (or `dest`), data/ and <problem>/data/ are linked to the real ones, and the command that repeats the run is printed, not started: it goes through
    run.py as `<id>.repro` with --no-card (a repetition feeds no verdict) and writes into the worktree's own runs/ folder, so the original run stays as
    it is. Remove it later with `git worktree remove --force <dir>`. -> (worktree dir, command line)"""
    rj = os.path.join(P.path("run_records"), run_id, "run.json")
    if not os.path.isfile(rj):
        sys.exit(f"no run record {rj}")
    r = json.load(open(rj, encoding="utf-8"))
    tag, repo = r.get("tag"), repo_root()
    if not tag or git(repo, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}") is None:
        sys.exit(f"the code tag {tag} of {run_id} does not exist here (git fetch --tags): nothing to check out")
    dest = os.path.abspath(dest or os.path.join(repo, "temp", "repro", run_id))
    if not os.path.isdir(dest):
        if git(repo, "worktree", "add", "--detach", dest, tag, timeout=3600) is None:   # a checkout on the Windows disk from WSL is slow
            git(repo, "worktree", "unlock", dest)                      # a failed or timed-out checkout leaves a locked, half-made worktree: take it away again
            git(repo, "worktree", "remove", "--force", dest)
            sys.exit(f"git worktree add {dest} {tag} failed (the half-made worktree was removed; check the leftovers in that folder; never run `git worktree prune` from WSL: it drops the registration of worktrees with Windows paths)")
    for rel in ("data", os.path.join(os.path.relpath(P.root(), repo), "data")):
        src, dst = os.path.join(repo, rel), os.path.join(dest, rel)
        if os.path.isdir(src) and not os.path.lexists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst)
    cwd = r.get("cwd") or P.root()
    rel = os.path.relpath(cwd, repo) if os.path.commonpath([os.path.abspath(cwd), repo]) == repo else os.path.relpath(P.root(), repo)
    work = os.path.normpath(os.path.join(dest, rel))
    cmd = (f"cd {work} && AR_PROJECT=. python3 ../autoresearch/tools/run.py --id {run_id}.repro --hyp \"repetition of {run_id}\" "
           f"--pred \"same code, same GT, same seed: the numbers agree within seed noise\" --rule \"by hand: a repetition, no verdict\" --no-card -- {r.get('command')}")
    return dest, cmd


def find_run_dir(run_id):
    """the folder a run writes into: <cards folder>/<id>, or the id with . as _ (bev_fusion: runs/t10_dual_s0); None when neither exists"""
    root = P.root()
    base = P.cfg_of(root).get("cards", "runs")
    for name in (run_id, run_id.replace(".", "_")):
        d = os.path.join(root, base, name)
        if os.path.isdir(d):
            return d
    return None


def final_ckpt(run_dir):
    """(path, step) of the last checkpoint of a run folder: the highest ckpt_<step>.pt, else model.pt / block.pt / ckpt.pt; (None, None) when there is none"""
    best = None
    for f in os.listdir(run_dir):
        m = re.fullmatch(r"ckpt_(\d+)\.pt", f)
        if m and (best is None or int(m.group(1)) > best[1]):
            best = (os.path.join(run_dir, f), int(m.group(1)))
    if best:
        return best
    for f in ("model.pt", "block.pt", "ckpt.pt"):
        if os.path.isfile(os.path.join(run_dir, f)):
            return os.path.join(run_dir, f), None
    return None, None


PROBE = ("import json, platform\nfrom importlib import metadata\n"
         "def v(n):\n    try:\n        return metadata.version(n)\n    except Exception:\n        return None\n"
         "print(json.dumps({'python': platform.python_version(), 'torch': v('torch'), 'numpy': v('numpy')}))\n")


def trainer_python(r):
    """the interpreter a run was started with: project.json "trainer_python", else the .../bin/python that the script named in the run's command calls
    (bev_fusion: temp/train/run_ar.sh -> ~/miniconda3/envs/gcar/bin/python); None when it is not found on this machine"""
    cands = [P.cfg_of(P.root()).get("trainer_python")]
    cwd = r.get("cwd") or P.root()
    for tok in (r.get("command") or "").split():
        f = os.path.join(cwd, tok)
        if tok.endswith(".sh") and os.path.isfile(f):
            try:
                cands += re.findall(r"(~?/[\w./-]*/bin/python[\d.]*)", open(f, encoding="utf-8", errors="replace").read())
            except OSError:
                pass
    for c in cands:
        c = os.path.expanduser(c) if c else c
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def trainer_env(r):
    """python, torch, CUDA and numpy versions of the interpreter that ran the training: asked through the package metadata (torch is not imported:
    about 0.1 s); the CUDA version is the +cuNNN part of the torch version; {} when the interpreter is not found or does not answer"""
    py = trainer_python(r)
    if not py:
        return {}
    try:
        out = subprocess.run([py, "-c", PROBE], capture_output=True, text=True, timeout=30)
        d = json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}
    m = re.search(r"\+cu(\d+)", d.get("torch") or "")
    d["cuda"] = f"{m.group(1)[:-1]}.{m.group(1)[-1]}" if m else None
    d["source"] = f"the interpreter of the run, read {dt.date.today().isoformat()} (the environment may have changed since the run): {py}"
    return d


def manifest(run_id, run_dir=None, write=True, force=False):
    """The provenance block of a TRAINING run, made from its run record and its run folder: GT it trained on (gt.json), checkpoint, config, split,
    seed, the run's status, the code tag and commit, and the model bundle (network + losses) from the run's own code snapshot (<run folder>/code;
    the live tree is never used: it is not what ran). Written to the result folder <problem>/results/<id>/provenance.json (`repro` reads it); an existing manifest or
    bundle/ is not rebuilt without force, so a later call cannot replace a good record by one made after the folder was slimmed. A missing piece
    reads "unknown"; nothing here changes the run. The environment comes from the run record: this tool runs in the launcher's interpreter, not
    the trainer's, so torch stays unknown"""
    rj = os.path.join(P.path("run_records"), run_id, "run.json")
    if not os.path.isfile(rj):
        sys.exit(f"no run record {rj}")
    out = os.path.join(results_dir(run_id), "provenance.json")
    if write and os.path.isfile(out) and not force:
        return json.load(open(out, encoding="utf-8"))
    r = json.load(open(rj, encoding="utf-8"))
    run_dir = run_dir or find_run_dir(run_id)
    have_dir = bool(run_dir and os.path.isdir(run_dir))
    sig = cfg = ck = step = seed = None
    if have_dir:
        g = os.path.join(run_dir, "gt.json")
        if os.path.isfile(g):
            try:
                sig = json.load(open(g, encoding="utf-8")).get("gt_signature")
            except (OSError, ValueError, AttributeError):
                sig = None
        cfg = os.path.join(run_dir, "config.json")
        cfg = cfg if os.path.isfile(cfg) else None
        ck, step = final_ckpt(run_dir)
        if cfg:
            try:
                seed = json.load(open(cfg, encoding="utf-8")).get("seed")
            except (OSError, ValueError, AttributeError):
                seed = None
    seed = seed if seed not in (None, "") else (r.get("seed") if r.get("seed") not in (None, "") else None)
    block = collect(gt_key=sig, split=r.get("split"), run=run_id, ckpt=ck, config=cfg, step=step, run_dir=run_dir if have_dir else None, seed=seed)
    block.pop("evaluator", None)  # a training run has no evaluator; its code is the run's commit
    drift = r.get("code_drift") or {}
    block["code"] = {"tag": r.get("tag") or UNKNOWN, "commit": r.get("commit") or UNKNOWN, "code_hash": r.get("code_hash") or UNKNOWN,
                     "drift_files": drift.get("paths", []) if drift.get("n") else [], "command": r.get("command"), "cwd": r.get("cwd")}
    block["run_status"] = {"status": r.get("status") or UNKNOWN, "exit": r.get("exit"), "final": r.get("status") in ("ok", "done") and r.get("exit") == 0}
    snap = os.path.join(run_dir, "code") if have_dir else None
    if snap and os.path.isdir(snap):
        block["bundle"] = bundle(snap, into=os.path.join(results_dir(run_id), "bundle") if write else None, config=cfg)
    else:
        block["bundle"] = {"code_hash": UNKNOWN, "files": [], "note": "the run has no code snapshot (<run folder>/code): the live tree is not what ran, check out the tag " + str(r.get("tag"))}
    env = trainer_env(r)
    block["environment"] = {"python": env.get("python") or r.get("python") or UNKNOWN, "torch": env.get("torch") or UNKNOWN, "cuda": env.get("cuda") or UNKNOWN,
                            "numpy": env.get("numpy") or UNKNOWN, "gpu": r.get("gpu") or UNKNOWN, "host": r.get("host") or UNKNOWN,
                            "conda_env": r.get("conda_env") or UNKNOWN, "source": env.get("source") or "run record (the trainer's interpreter was not found)"}
    if not have_dir:
        block["run_dir"] = UNKNOWN
    if write:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        P.atomic_write(out, json.dumps(block, indent=1) + "\n")
    return block


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
        # repro --run: the tag is checked out in a worktree, data is linked, the current checkout is untouched
        run("tag", "tg/r1")
        real_root = repo_root
        globals()["repo_root"] = lambda start=None: tmp
        try:
            os.makedirs(os.path.join(tmp, "data", "autogt"), exist_ok=True)
            wt, cmd = repro_prepare("r1")
            assert os.path.isfile(os.path.join(wt, "ev", "m.py")) and wt == os.path.join(tmp, "temp", "repro", "r1"), wt
            assert os.path.islink(os.path.join(wt, "data")) and os.path.realpath(os.path.join(wt, "data")) == os.path.realpath(os.path.join(tmp, "data")), "data is linked"
            assert "--id r1.repro" in cmd and "--no-card" in cmd and "bash e.sh" in cmd and cmd.startswith("cd "), cmd
            assert repro_prepare("r1")[0] == wt, "a second call reuses the worktree"
            assert not os.path.exists(os.path.join(tmp, "temp", "repro", "r1", "ckpt.pt")) and os.path.isfile(os.path.join(tmp, "ev", "m.py")), "the checkout is separate"
            try:
                repro_prepare("r2")
                raise AssertionError("a run without its tag cannot be prepared")
            except SystemExit as e:
                assert "does not exist" in str(e) or "no run record" in str(e), e
        finally:
            globals()["repo_root"] = real_root
        # the model bundle (network + losses = the import closure of model.py and losses.py) and the manifest of a training run
        tr = os.path.join(proj, "runs", "r2")

        def put(root, rel, text):
            os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
            open(os.path.join(root, rel), "w").write(text)
        snap = os.path.join(tr, "code")
        for root in (snap,):
            put(root, "model.py", "import fusion\nfrom pkg import head\n")
            put(root, "fusion.py", "import blocks\n")
            put(root, "blocks.py", "X = 1\n")
            put(root, "pkg/__init__.py", "")
            put(root, "pkg/head.py", "from . import act\n")
            put(root, "pkg/act.py", "Y = 2\n")
            put(root, "losses.py", "from loss_extras import f\n")
            put(root, "loss_extras.py", "def f(): pass\n")
            put(root, "data.py", "import model\n")                  # imports the network, is not imported by it: not in the bundle
            put(root, "train.py", "import model, data, losses\n")
        want = ["blocks.py", "fusion.py", "loss_extras.py", "losses.py", "model.py", "pkg/__init__.py", "pkg/act.py", "pkg/head.py"]
        assert import_closure(snap) == want, import_closure(snap)
        h0 = bundle(snap)["code_hash"]
        put(snap, "data.py", "import model  # edited\n")
        put(snap, "train.py", "# edited\n")
        assert bundle(snap)["code_hash"] == h0, "an edit of the data loader or the training loop does not change the bundle"
        put(snap, "blocks.py", "X = 2\n")
        assert bundle(snap)["code_hash"] != h0, "an edit of a network module does"
        put(snap, "blocks.py", "X = 1\n")
        put(snap, "pkg/act.py", "import train\nY = 2\n")                # a network module that imports the training loop: reported, the loop stays out
        bx = bundle(snap, exclude=("train.py",))
        assert "train.py" not in bx["files"] and bx["imports_excluded"] == ["pkg/act.py -> train.py"] and "train.py" in import_closure(snap), bx
        put(snap, "pkg/act.py", "Y = 2\n")
        assert bundle(os.path.join(tmp, "nowhere"))["code_hash"] == UNKNOWN and bundle(tmp, roots=("none.py",))["code_hash"] == UNKNOWN, "nothing found reads unknown"
        json.dump({"gt_signature": "abc123"}, open(os.path.join(tr, "gt.json"), "w"))
        json.dump({"seed": 3}, open(os.path.join(tr, "config.json"), "w"))
        for n in (100, 200):
            open(os.path.join(tr, f"ckpt_{n:08d}.pt"), "wb").write(b"w" * n)
        os.makedirs(os.path.join(proj, "rr", "r2"))
        json.dump({"id": "r2", "status": "ok", "exit": 0, "tag": "tg/r2", "commit": "def", "code_hash": "h2", "command": "bash t.sh", "cwd": "/x", "split": "train",
                   "python": "3.10", "gpu": "RTX"}, open(os.path.join(proj, "rr", "r2", "run.json"), "w"))
        fake = os.path.join(tmp, "envs", "bin")
        os.makedirs(fake)
        open(os.path.join(fake, "python"), "w").write("#!/bin/sh\necho '{\"python\": \"3.10.12\", \"torch\": \"2.8.0+cu128\", \"numpy\": \"1.26.4\"}'\n")
        os.chmod(os.path.join(fake, "python"), 0o755)
        open(os.path.join(tmp, "train.sh"), "w").write("#!/bin/bash\n" + os.path.join(fake, "python") + " -u train.py\n")
        rj2 = os.path.join(proj, "rr", "r2", "run.json")
        d2 = json.load(open(rj2))
        d2.update(command="bash train.sh", cwd=tmp)
        json.dump(d2, open(rj2, "w"))
        m = manifest("r2")
        assert m["environment"]["torch"] == "2.8.0+cu128" and m["environment"]["cuda"] == "12.8" and m["environment"]["numpy"] == "1.26.4" and "envs/bin/python" in m["environment"]["source"], m["environment"]
        assert trainer_env({"command": "bash nothing.sh", "cwd": tmp}) == {} and trainer_env({}) == {}, "an interpreter that is not found reads unknown, not a guess"
        assert m["subject"]["ckpt"] == "ckpt_00000200.pt" and m["subject"]["step"] == 200 and m["seed"] == 3 and m["code"]["tag"] == "tg/r2", m
        assert m["bundle"]["files"] == want and os.path.isfile(os.path.join(proj, "results", "r2", "bundle", "pkg", "head.py")) and os.path.isfile(os.path.join(proj, "results", "r2", "bundle", "config.json")), m["bundle"]
        assert not os.path.exists(os.path.join(proj, "results", "r2", "bundle", "data.py")), "the bundle holds the network and the losses only"
        assert json.load(open(os.path.join(proj, "results", "r2", "provenance.json")))["code"]["commit"] == "def" and "evaluator" not in m and m["environment"]["torch"] == "2.8.0+cu128"
        assert m["run_status"] == {"status": "ok", "exit": 0, "final": True} and m["bundle"]["missing_roots"] == [] and m["bundle"]["copied_to"] == "bundle", m
        put(os.path.join(proj, "results", "r2"), "bundle/stale.py", "old\n")
        assert manifest("r2") == m, "an existing manifest is returned, not rebuilt"
        m2f = manifest("r2", force=True)
        assert "copy_skipped" in m2f["bundle"] and os.path.isfile(os.path.join(proj, "results", "r2", "bundle", "stale.py")), "a bundle/ that exists is left alone"
        # hard cases of the closure: BOM, parent packages with code, level-2 relative imports, missing root, broken file, dynamic import, external local code
        s2 = os.path.join(tmp, "snap2")
        open(os.path.join(s2 if os.path.isdir(s2) else (os.makedirs(s2) or s2), "model.py"), "w", encoding="utf-8-sig").write("import pkg.head\nimport autogt\nimport torch\nimport broken\nimport importlib\nimportlib.import_module('x')\n")
        put(s2, "pkg/__init__.py", "from .reg import R\n")
        put(s2, "pkg/reg.py", "R = 1\n")
        put(s2, "pkg/head.py", "from .. import top\nfrom ..lib import leaf\n")
        put(s2, "top.py", "T = 1\n")
        put(s2, "lib/__init__.py", "")
        put(s2, "lib/leaf.py", "L = 1\n")
        put(s2, "broken.py", "def (:\n")
        os.makedirs(os.path.join(tmp, "autogt"), exist_ok=True)
        b2 = bundle(s2)
        assert b2["files"] == ["broken.py", "lib/__init__.py", "lib/leaf.py", "model.py", "pkg/__init__.py", "pkg/head.py", "pkg/reg.py", "top.py"] or b2["files"] == sorted(b2["files"]), b2["files"]
        for f in ("model.py", "pkg/__init__.py", "pkg/reg.py", "pkg/head.py", "lib/leaf.py", "top.py"):
            assert f in b2["files"], (f, b2["files"])
        assert b2["unparsed"] == ["broken.py"] and b2["missing_roots"] == ["losses.py"] and b2["dynamic_imports"] == ["model.py"], b2
        assert "torch" not in b2["external_local"] and "os" not in b2["external_local"], "stdlib and installed packages are not repository code"
        # a run folder without a code snapshot: the live tree is not used; a run that is not final says so
        os.makedirs(os.path.join(proj, "rr", "r3"))
        os.makedirs(os.path.join(proj, "runs", "r4"))
        os.makedirs(os.path.join(proj, "rr", "r4"))
        json.dump({"id": "r4", "status": "stopped", "exit": 75, "tag": "tg/r4"}, open(os.path.join(proj, "rr", "r4", "run.json"), "w"))
        m4 = manifest("r4")
        assert m4["bundle"]["code_hash"] == UNKNOWN and "no code snapshot" in m4["bundle"]["note"] and m4["run_status"]["final"] is False, m4
        buf4 = io.StringIO()
        with contextlib.redirect_stdout(buf4):
            repro("r4")
        assert "WARNING: the run is not final (status stopped, exit 75)" in buf4.getvalue(), buf4.getvalue()
        json.dump({"id": "r3", "status": "failed", "exit": 1}, open(os.path.join(proj, "rr", "r3", "run.json"), "w"))
        m3 = manifest("r3")
        assert m3["subject"].get("ckpt") is None and m3["run_dir"] == UNKNOWN and m3["bundle"]["code_hash"] == UNKNOWN, m3
        assert m3["gt"]["version"] == UNKNOWN and m3["seed"] == UNKNOWN, "a run without a folder reads unknown, it does not crash"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            repro("r2")
        assert "manifest of the run" in buf.getvalue() and "model bundle (network + losses): 8 files" in buf.getvalue(), buf.getvalue()
        print("provenance selftest passed")
    finally:
        if old_env is None:
            os.environ.pop("AR_PROJECT", None)
        else:
            os.environ["AR_PROJECT"] = old_env
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["collect", "repro", "compare", "manifest"])
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
    ap.add_argument("--force", action="store_true", help="manifest: rebuild an existing provenance.json")
    ap.add_argument("--repeat", action="store_true", help="repro: check the tag out in a worktree and print the repeat command")
    ap.add_argument("--dir", help="repro --run: where to put the worktree (default temp/repro/<id>)")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "collect":
        toks = open(a.frames, encoding="utf-8").read().split() if a.frames else []
        print(json.dumps(collect(a.gt_key, a.split, toks, a.evaluator, a.run, a.ckpt, a.config, a.step, a.run_dir, a.seed), indent=1))
    elif a.cmd == "repro" and a.args:
        repro(a.args[0], a.card)
        if a.repeat:
            wt, cmd = repro_prepare(a.args[0], a.dir)
            print(f"\nworktree (the tag checked out, the current checkout untouched): {wt}\nstart the repetition (check the GT store above first):\n  {cmd}\nremove it afterwards: git worktree remove --force {wt}")
    elif a.cmd == "manifest" and a.args:
        b = manifest(a.args[0], a.run_dir, force=a.force)
        print(f"written: {os.path.join(results_dir(a.args[0]), 'provenance.json')} (GT {_get(b, ('gt', 'version'))}, checkpoint {_get(b, ('subject', 'ckpt'))}, code {_get(b, ('code', 'tag'))})")
    elif a.cmd == "compare" and len(a.args) == 2:
        text, code = verdict(*compare(*(json.load(open(x, encoding="utf-8")) for x in a.args)))
        print(text)
        sys.exit(code)
    else:
        ap.error("collect | repro <run id> | manifest <run id> [--run-dir d] | compare a.json b.json | --selftest")


if __name__ == "__main__":
    main()
