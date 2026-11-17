"""scratch (E343 code step, round 3): the re-review's findings applied to evaluation/jtest_eval.py"""
import os

P = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "jtest_eval.py")
s = open(P, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


# ---------------------------------------------------------------------------------------------------- allow-list for the write helper
rep('''def is_protected(path):''', '''WRITABLE = ("temp/e343", "runs/eval/e324s_val", "runs/eval/e324s_test", "runs/eval/e312_val")          # the ONLY places this module writes (under ROOT): the deny-list above stays, as a second guard
EXTRA_WRITABLE = ()                                                                                  # more absolute folders, for the tests' temporary folders only


def is_writable(path):
    """True when the real path lies under one of the WRITABLE folders (case-folded). An allow-list: a NTFS short name such as `runs/eval/E312_T~1` is the same folder as E321's cards but is not under any of these"""
    real = os.path.realpath(path).casefold()
    roots = [os.path.realpath(os.path.join(ROOT, p)) for p in WRITABLE] + [os.path.realpath(p) for p in EXTRA_WRITABLE]
    return any(real == r.casefold() or real.startswith(r.casefold() + os.sep) for r in roots)


def is_protected(path):''')
rep('''    if is_protected(path):
        raise PermissionError(f"refused: {path} lies under {' / '.join(PROTECTED)} (the artefacts of E321 are read-only inputs)")
    if not overwrite and os.path.exists(path):''', '''    if is_protected(path):
        raise PermissionError(f"refused: {path} lies under {' / '.join(PROTECTED)} (the artefacts of E321 are read-only inputs)")
    if not is_writable(path):
        raise PermissionError(f"refused: {path} is not under {' / '.join(WRITABLE)} (the only folders this module writes)")
    if not overwrite and os.path.exists(path):''')
rep('''refuses a protected path before anything is created,
    and with `overwrite=False` a file that exists (a result is written once)"""''', '''refuses a protected path and a path outside the WRITABLE folders before anything is created,
    and with `overwrite=False` a file that exists (a result is written once)"""''')

# ---------------------------------------------------------------------------------------------------- code digest: resolver over the project's real search path, caches
a = s.index('CODE_SKIP = ("evaluate", "GtGenerator")')
b = s.index("def provenance(run, device):")
s = s[:a] + '''CODE_SKIP = ("evaluate", "GtGenerator")                                                             # the TEST gate's evaluator pulls in the whole evaluation suite; the GT store is keyed by its own signature
_SHA, _IMP = {}, {}


def _stamp(path):
    st = os.stat(path)
    return path, st.st_mtime_ns, st.st_size


def sha256_cached(path):
    """sha256 of a file, computed again only when its modification time or size changed (the digest is taken often)"""
    key = _stamp(path)
    if key not in _SHA:
        _SHA[key] = sha256_file(path)
    return _SHA[key]


def _imports(path):
    """(level, module name) of every import statement of a source file, also those inside functions; `from m import n` also as m.n (n may be a submodule)"""
    key = _stamp(path)
    if key not in _IMP:
        try:
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except (OSError, SyntaxError, UnicodeDecodeError):
            _IMP[key] = []
            return []
        out = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                out += [(0, a.name) for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                base = n.module or ""
                out += [(n.level, base)] + [(n.level, f"{base}.{a.name}" if base else a.name) for a in n.names]
        _IMP[key] = out
    return _IMP[key]


def code_dirs():
    """the folders a bare module name is looked up in, as the process sees them: what tools/env/bflc_env.sh puts on PYTHONPATH (every folder up to 4 deep under training/, evaluation/, tools/ and reports/build/, no scratch) plus
    the problem root, the repository root and autoresearch/ (the paths the modules insert themselves)"""
    repo = os.path.realpath(os.path.join(ROOT, ".."))
    dirs = [os.path.realpath(ROOT), repo, os.path.join(repo, "autoresearch")]
    for top in ("training", "evaluation", "tools", os.path.join("reports", "build")):
        base = os.path.join(ROOT, top)
        for dp, dn, _ in os.walk(base):
            rel = os.path.relpath(dp, base)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            dn[:] = [] if depth >= 4 else sorted(d for d in dn if d != "__pycache__" and "_session_scratch_" not in d)
            dirs.append(os.path.realpath(dp))
    return dirs


def code_sha256(entry=None):
    """sha256 of every file of the lane code this module runs, keyed by the path from the repository root: the static closure of the `import` statements (also those inside functions) of `entry` (default: this file), each looked up in every folder of
    `code_dirs()` (a name found in several folders counts for all of them), with the packages above a dotted name; library modules are not followed, nor those of CODE_SKIP. A superset of what a run loads"""
    dirs, memo = code_dirs(), {}

    def lookup(name, roots):
        parts, out = name.split("."), []
        for d in roots:
            base = os.path.join(d, *parts)
            for c in (base + ".py", os.path.join(base, "__init__.py")):
                if os.path.isfile(c):
                    out += [c] + [p for p in (os.path.join(d, *parts[:k], "__init__.py") for k in range(1, len(parts))) if os.path.isfile(p)]
        return out
    found, todo = set(), [os.path.realpath(entry or os.path.abspath(__file__))]
    while todo:
        p = todo.pop()
        if p in found:
            continue
        found.add(p)
        for level, name in _imports(p):
            if not name or name.split(".")[0] in CODE_SKIP:
                continue
            if level:
                hits = lookup(name, [os.path.normpath(os.path.join(os.path.dirname(p), *([".."] * (level - 1))))])
            else:
                hits = memo.setdefault(name, lookup(name, dirs))
            todo += [os.path.realpath(h) for h in hits]
    repo = os.path.realpath(os.path.join(ROOT, ".."))
    return {os.path.relpath(p, repo).replace(os.sep, "/"): sha256_cached(p) for p in sorted(found)}


def code_digest(files):
    return hashlib.sha256("\\n".join(f"{k} {v}" for k, v in sorted(files.items())).encode()).hexdigest()


''' + s[b:]

# ---------------------------------------------------------------------------------------------------- sidecar provenance, check
rep('''def check_provenance(path, expect, what):
    """a dump this module made earlier is reused only when its sidecar names the same run, code, device and checkpoint; else it is refused (nothing is deleted: move it away and `all` dumps again)"""
    hp = head_path(path)
    if not os.path.exists(hp):
        sys.exit(f"{what}: {path} has no provenance ({hp}): it was made by other code; move it away and run again")
    with open(hp) as f:
        got = json.load(f).get("provenance") or {}
    bad =''', '''def sidecar_provenance(path):
    """the provenance of a dump this module made, or None: no sidecar, a sidecar WITHOUT a provenance key (E321's `_head.json` files, written by the old `dump_ends`), a sidecar that cannot be read, or a name without .pkl"""
    if not path.endswith(".pkl") or not os.path.exists(head_path(path)):
        return None
    try:
        with open(head_path(path)) as f:
            got = json.load(f).get("provenance")
    except (OSError, ValueError, AttributeError):
        return None
    return got if isinstance(got, dict) and got else None


def check_provenance(path, expect, what, required=True):
    """a dump this module made earlier is reused only when its sidecar names the same run, code, device and checkpoint; else it is refused (nothing is deleted: move it away and `all` dumps again). `required`: a dump without provenance is refused
    (the default paths); a dump named by a flag without provenance (E321's, the stored VAL dumps) is not this module's and is exempt, one with provenance is checked"""
    got = sidecar_provenance(path)
    if got is None:
        if required:
            sys.exit(f"{what}: {path} has no provenance: made by other code OR interrupted between the pickle and its sidecar {head_path(path)}: move it away and run again")
        return
    hp = head_path(path)
    bad =''')
rep('''    elif expect is not None and (make is not None or os.path.exists(head_path(path))):                # a dump of the default path must have its provenance; one named by a flag only when it has a sidecar
        check_provenance(path, expect, what)''', '''    elif expect is not None:
        check_provenance(path, expect, what, required=make is not None)                                  # a dump of the default path must have its provenance; one named by a flag only when it has one''')
rep('''a dump named by a flag without a sidecar (E321's, the stored VAL dumps: validated, E321's also by hash) is exempt, one made by
    this module (with a sidecar) is checked like the others"""''', '''a dump named by a flag without provenance (E321's, the stored VAL dumps: validated, E321's also by hash; E321's old head files have no provenance key) is exempt, one made by
    this module (with provenance) is checked like the others"""''')

# ---------------------------------------------------------------------------------------------------- provenance BEFORE the model is loaded
rep('''    head_path(out)                                                                                     # a dump name ends with .pkl; an existing dump is never overwritten
    refuse_existing(out)
    import lane_gt_data as LD
    import qgraph_readout as RO''', '''    head_path(out)                                                                                     # a dump name ends with .pkl; an existing dump is never overwritten
    refuse_existing(out)
    prov = provenance(run, device)                                                                     # the code and the checkpoint as they are when the dump STARTS (not after 10 - 25 minutes)
    import lane_gt_data as LD
    import qgraph_readout as RO''')
rep('''    write_head(out, dict(split=split, frames=len(preds)), provenance(run, device))''', '''    write_head(out, dict(split=split, frames=len(preds)), prov)''')
rep('''    head_path(out)
    refuse_existing(out)
    import jends_eval as JE''', '''    head_path(out)
    refuse_existing(out)
    prov = provenance(run, device)                                                                     # at the start, as in dump_b
    import jends_eval as JE''')
rep('''    write_head(out, head, provenance(run, device))''', '''    write_head(out, head, prov)''')

# ---------------------------------------------------------------------------------------------------- precheck: provenance of every existing dump before any is made
rep('''    named = ([] if i["b_made"] else [i["b"]]) + ([i["ctl_ends"], i["ctl_b"]] if setup["arms"] and i["ctl_ends"] else []) + ([i["card"]] if setup["arms"] else [])
    missing = [p for p in named if not os.path.exists(p)]
    if missing:
        sys.exit(f"STOP (seed {seed}): not there, and named by a flag or by the split's defaults (B's dump is never made on TEST): {missing}")''', '''    named = ([] if i["b_made"] else [i["b"]]) + ([i["ctl_ends"], i["ctl_b"]] if setup["arms"] and i["ctl_ends"] else []) + ([i["card"]] if setup["arms"] else [])
    missing = [p for p in named if not os.path.exists(p)]
    if missing:
        sys.exit(f"STOP (seed {seed}): not there, and named by a flag or by the split's defaults (B's dump is never made on TEST): {missing}")
    run_b, run_c = sub_seed(setup["graph_run"], seed), sub_seed(setup.get("stack_run"), seed)
    dumps = ([(i["b"], run_b, "B's dump", i["b_made"])] + [(i["ends"], i["run"], f"pair dump of {i['run']}", True)] + ([(i["c2"], run_c, "c2's dump", True)] if stack else [])
             + ([(i["ctl_ends"], sub_seed(CONTROL_RUN, seed), "the control's pair dump", False)] + ([(i["ctl_b"], run_b, "the control's B dump", False)] if i["ctl_b"] != i["b"] else []) if setup["arms"] and i["ctl_ends"] else []))
    for path, run, what, required in dumps:                                                            # a dump that exists under other provenance is refused here, before any dump of any seed is made
        if os.path.exists(path) and (required or sidecar_provenance(path) is not None):
            check_provenance(path, provenance(run, a.device), what, required)''')

# ---------------------------------------------------------------------------------------------------- cmd_all: the digest at the start, --expect-code-digest / --expect-code-card
rep('''    guard_e321("before the run")
    pre = {s: precheck(a, setup, s, stack) for s in a.seeds}''', '''    code0 = code_sha256()
    digest = code_digest(code0)
    print(f"code digest: {digest} ({len(code0)} files)", flush=True)
    if a.expect_code_digest and not (digest == a.expect_code_digest or (len(a.expect_code_digest) >= 16 and digest.startswith(a.expect_code_digest))):
        named = ""
        if a.expect_code_card:
            with open(a.expect_code_card) as f:
                ref = json.load(f).get("code_sha256") or {}
            changed = sorted(k for k in set(ref) | set(code0) if ref.get(k) != code0.get(k))
            named = f"; changed files against {a.expect_code_card}: {', '.join(changed[:12])}{' ...' if len(changed) > 12 else ''}"
        sys.exit(f"STOP: the imported lane code (digest {digest}) is not the code of the rehearsal (--expect-code-digest {a.expect_code_digest}){named or '; give --expect-code-card <the rehearsal card> to name the files'}")
    guard_e321("before the run")
    pre = {s: precheck(a, setup, s, stack) for s in a.seeds}''')
rep('''    frames = frames_of(a.split)[: a.max or None]
    code0, done = code_sha256(), []''', '''    frames = frames_of(a.split)[: a.max or None]
    done = []''')

# ---------------------------------------------------------------------------------------------------- main: flags, explicit raise
rep('''    s.add_argument("--no-stack", action="store_true", help="E324 s without arm k (its check failed)")
    s = sub.add_parser("verify")''', '''    s.add_argument("--no-stack", action="store_true", help="E324 s without arm k (its check failed)")
    s.add_argument("--expect-code-digest", default=None, help="the code digest of the rehearsal's cards (`code_digest`, or its first 16 characters): refused before any dump when the imported lane code differs")
    s.add_argument("--expect-code-card", default=None, help="a card of the rehearsal (its `code_sha256`): names the changed files when --expect-code-digest refuses")
    s = sub.add_parser("verify")''')
rep('''    for attr in ("out", "b_dump", "control_ends_dump", "control_card"):''', '''    for attr in ("out", "b_dump", "control_ends_dump", "control_card", "expect_code_card"):''')
rep('''        assert re.search(rf"_s{a.seed}$", run), f"--b-run {run}: not a run of seed {a.seed}"''', '''        _require(re.search(rf"_s{a.seed}$", run), f"--b-run {run}: not a run of seed {a.seed}")''')
rep('''s.add_argument("--out", default=None)
    s = sub.add_parser("dump-ends")''', '''s.add_argument("--out", default=None, help="the dump file (ends with .pkl; under temp/e343/)")
    s = sub.add_parser("dump-ends")''')

open(P, "w", encoding="utf-8").write(s)
print("patched", len(s.splitlines()), "lines")
