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
    normalised), and the run's config. into = a folder to copy them to (runs/<id>/bundle/), so the model can be rebuilt from that folder and the
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


def manifest(run_id, run_dir=None, write=True, force=False):
    """The provenance block of a TRAINING run, made from its run record and its run folder: GT it trained on (gt.json), checkpoint, config, split,
    seed, the run's status, the code tag and commit, and the model bundle (network + losses) from the run's own code snapshot (<run folder>/code;
    the live tree is never used: it is not what ran). Written to <run_records>/<id>/provenance.json (`repro` reads it); an existing manifest or
    bundle/ is not rebuilt without force, so a later call cannot replace a good record by one made after the folder was slimmed. A missing piece
    reads "unknown"; nothing here changes the run. The environment comes from the run record: this tool runs in the launcher's interpreter, not
    the trainer's, so torch stays unknown"""
    rj = os.path.join(P.path("run_records"), run_id, "run.json")
    if not os.path.isfile(rj):
        sys.exit(f"no run record {rj}")
    out = os.path.join(os.path.dirname(rj), "provenance.json")
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
        block["bundle"] = bundle(snap, into=os.path.join(run_dir, "bundle") if write else None, config=cfg)
    else:
        block["bundle"] = {"code_hash": UNKNOWN, "files": [], "note": "the run has no code snapshot (<run folder>/code): the live tree is not what ran, check out the tag " + str(r.get("tag"))}
    block["environment"] = {"python": r.get("python") or UNKNOWN, "gpu": r.get("gpu") or UNKNOWN, "host": r.get("host") or UNKNOWN,
                            "conda_env": r.get("conda_env") or UNKNOWN, "torch": UNKNOWN, "source": "run record"}
    if not have_dir:
        block["run_dir"] = UNKNOWN
    if write:
        P.atomic_write(out, json.dumps(block, indent=1) + "\n")
    return block


