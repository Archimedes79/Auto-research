"""scratch (E343 code step, round 2): the critic's findings applied to evaluation/jtest_eval.py by exact replacements (each old text must be found once)"""
import os

P = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "jtest_eval.py")
s = open(P, encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


# ---------------------------------------------------------------------------------------------------- imports, _require
rep("import argparse\nimport contextlib\n", "import argparse\nimport ast\nimport contextlib\n")
rep('''REHEARSAL = dict(topo=1e-3, real=3e-3, count_rel=5e-4)                                            # criterion 2 of E343: a fresh CPU dump against the stored (GPU) VAL cards
''', '''REHEARSAL = dict(topo=1e-3, real=3e-3, count_rel=5e-4)                                            # criterion 2 of E343: a fresh CPU dump against the stored (GPU) VAL cards


def _require(cond, msg):
    """a check that stays under `python -O` (an `assert` does not)"""
    if not cond:
        raise RuntimeError(msg)
''')

# ---------------------------------------------------------------------------------------------------- stored_counts in the setups
rep('''stored={"unsnapped": "ends_card_thr0.1.json", "snapped": "ends_card_snap_thr0.1.json"}, stored_man="maneuver_val.json", stack=[], arms={}),''',
    '''stored={"unsnapped": "ends_card_thr0.1.json", "snapped": "ends_card_snap_thr0.1.json"}, stored_counts=("snapped",), stored_man="maneuver_val.json", stack=[], arms={}),''')
rep('''"stack": "ends_card_e338_snap_thr0.1.json"}, stored_man=None,
        arms=dict(w="snapped", p="primary", k="stack")),''', '''"stack": "ends_card_e338_snap_thr0.1.json"},
        stored_counts=("snapped", "snap_nogeom", "primary", "stack"), stored_man=None, arms=dict(w="snapped", p="primary", k="stack")),''')

# ---------------------------------------------------------------------------------------------------- is_protected (case), atomic / save_*
rep('''def is_protected(path):
    """True when the real path (symlinks and `..` resolved) lies under one of the PROTECTED folders"""
    real = os.path.realpath(path)
    return any(real == r or real.startswith(r + os.sep) for r in (os.path.realpath(os.path.join(ROOT, p)) for p in PROTECTED))
''', '''def is_protected(path):
    """True when the real path (symlinks and `..` resolved, upper / lower case ignored: /mnt/c is case-insensitive) lies under one of the PROTECTED folders"""
    real = os.path.realpath(path).casefold()
    return any(real == r or real.startswith(r + os.sep) for r in (os.path.realpath(os.path.join(ROOT, p)).casefold() for p in PROTECTED))
''')
rep('''@contextlib.contextmanager
def atomic(path):
    """the ONE way this module writes a dump or a card: yields a temporary file name next to `path`, renames it onto `path` when the block ends (a crash leaves no truncated file that a re-run could reuse); refuses a protected path before anything is created"""
    if is_protected(path):
        raise PermissionError(f"refused: {path} lies under {' / '.join(PROTECTED)} (the artefacts of E321 are read-only inputs)")
''', '''@contextlib.contextmanager
def atomic(path, overwrite=True):
    """the ONE way this module writes a dump or a card: yields a temporary file name next to `path`, renames it onto `path` when the block ends (a crash leaves no truncated file that a re-run could reuse); refuses a protected path before anything is created,
    and with `overwrite=False` a file that exists (a result is written once)"""
    if is_protected(path):
        raise PermissionError(f"refused: {path} lies under {' / '.join(PROTECTED)} (the artefacts of E321 are read-only inputs)")
    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"refused: {path} exists (a result is written once: move it away to write it again)")
''')
rep('''def save_pickle(path, obj):
    with atomic(path) as tmp:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)


def save_json(path, obj):
    save_text(path, json.dumps(obj, indent=1))


def save_text(path, text):
    with atomic(path) as tmp:
        with open(tmp, "w") as f:
            f.write(text)
''', '''def save_pickle(path, obj, overwrite=True):
    with atomic(path, overwrite) as tmp:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)


def save_json(path, obj, overwrite=True):
    save_text(path, json.dumps(obj, indent=1), overwrite)


def save_text(path, text, overwrite=True):
    with atomic(path, overwrite) as tmp:
        with open(tmp, "w") as f:
            f.write(text)
''')

# ---------------------------------------------------------------------------------------------------- e321_changed (a missing artefact is a change), code hash, provenance
rep('''def e321_changed():
    """the artefacts of E321 (the ones that exist) whose sha256 differs from the card's"""
    return [rel for rel, want in E321_SHA256.items() if os.path.exists(os.path.join(ROOT, rel)) and sha256_file(os.path.join(ROOT, rel)) != want]
''', '''CODE_SKIP = ("evaluate", "GtGenerator")                                                             # the TEST gate's evaluator pulls in the whole evaluation suite; the GT store is keyed by its own signature


def _imports(path):
    """(level, module name) of every import statement of a source file; `from m import n` also as m.n (n may be a submodule)"""
    try:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [(0, a.name) for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            out += [(n.level, base)] + [(n.level, f"{base}.{a.name}" if base else a.name) for a in n.names]
    return out


def code_sha256(entry=None):
    """sha256 of every file of the lane code this module runs, keyed by the path from the repository root: the static closure of the `import` statements of `entry` (default: this file) over the project's code folders, found the way `sys.path` finds them
    (the lane's evaluation / training folders, the repository root, autoresearch/, so `Evaluation/stats.py` is in); modules of the library are not followed, nor those of CODE_SKIP"""
    dirs = [os.path.normpath(os.path.join(ROOT, p)) for p in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", "tools/viz", "..", "../autoresearch")]
    repo = os.path.realpath(os.path.join(ROOT, ".."))
    found, todo = set(), [os.path.realpath(entry or os.path.abspath(__file__))]
    while todo:
        p = todo.pop()
        if p in found:
            continue
        found.add(p)
        for level, name in _imports(p):
            if not name or name.split(".")[0] in CODE_SKIP:
                continue
            roots = [os.path.normpath(os.path.join(os.path.dirname(p), *([".."] * (level - 1))))] if level else dirs
            for d in roots:
                base = os.path.join(d, *name.split("."))
                hit = next((c for c in (base + ".py", os.path.join(base, "__init__.py")) if os.path.isfile(c)), None)
                if hit:
                    todo.append(os.path.realpath(hit))
                    break
    return {os.path.relpath(p, repo).replace(os.sep, "/"): sha256_file(p) for p in sorted(found)}


def code_digest(files):
    return hashlib.sha256("\\n".join(f"{k} {v}" for k, v in sorted(files.items())).encode()).hexdigest()


def provenance(run, device):
    """what a dump of `run` made now depends on: the code digest, the device, the checkpoint (a dump is reused only under the same provenance)"""
    return dict(run=run, device=device, code_digest=code_digest(code_sha256()), checkpoint_sha256=sha256_file(os.path.join(ROOT, "runs", run, "model.pt")))


def head_path(out):
    _require(out.endswith(".pkl"), f"{out}: the name of a dump ends with .pkl (its head and provenance go to <name>_head.json, which must not overwrite it)")
    return out[: -len(".pkl")] + "_head.json"


def write_head(out, head, prov):
    save_json(head_path(out), dict(head, provenance=prov))


def check_provenance(path, expect, what):
    """a dump this module made earlier is reused only when its sidecar names the same run, code, device and checkpoint; else it is refused (nothing is deleted: move it away and `all` dumps again)"""
    hp = head_path(path)
    if not os.path.exists(hp):
        sys.exit(f"{what}: {path} has no provenance ({hp}): it was made by other code; move it away and run again")
    with open(hp) as f:
        got = json.load(f).get("provenance") or {}
    bad = [k for k in ("run", "code_digest", "device", "checkpoint_sha256") if got.get(k) != expect.get(k)]
    if bad:
        sys.exit(f"{what}: {path} was made under another {', '.join(bad)} (provenance {hp}): it is not reused; move it away and run again")


def refuse_existing(path):
    if os.path.exists(path):
        raise SystemExit(f"refused: {path} exists: a dump is never overwritten (`all` reuses a valid one; move this one away to dump again)")


def e321_changed():
    """the artefacts of E321 whose sha256 differs from the card's, and the ones that are missing (a deleted artefact is a change)"""
    out = []
    for rel, want in E321_SHA256.items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            out.append(f"{rel} (missing)")
        elif sha256_file(p) != want:
            out.append(rel)
    return out
''')

# ---------------------------------------------------------------------------------------------------- frames / validation: explicit raises
rep('''    assert rows.max() < min(len(tokens), len(scene), len(gt)), (rows.max(), len(tokens), len(scene), len(gt))''',
    '''    _require(rows.max() < min(len(tokens), len(scene), len(gt)), (rows.max(), len(tokens), len(scene), len(gt)))''')
rep('''    assert len(rows) == EXPECT_FRAMES[split], (split, len(rows))
    return frames_from_rows(rows)''', '''    _require(len(rows) == EXPECT_FRAMES[split], (split, len(rows)))
    return frames_from_rows(rows)''')
rep('''    assert len(rows) == EXPECT_FRAMES[split], f"data/splits.json has {len(rows)} frames in {split}, expected {EXPECT_FRAMES[split]}"
    rows, items = rows[: n or None], items[: n or None]
    assert len(items) == len(rows), f"{what}: {len(items)} frames, the split {split} has {len(rows)}"
    assert [int(d["row"]) for d in items] == rows, f"{what}: the rows are not the rows of the split {split} in the order of data/splits.json"
    return items''', '''    _require(len(rows) == EXPECT_FRAMES[split], f"data/splits.json has {len(rows)} frames in {split}, expected {EXPECT_FRAMES[split]}")
    rows, items = rows[: n or None], items[: n or None]
    _require(len(items) == len(rows), f"{what}: {len(items)} frames, the split {split} has {len(rows)}")
    _require([int(d["row"]) for d in items] == rows, f"{what}: the rows are not the rows of the split {split} in the order of data/splits.json")
    return items''')

# ---------------------------------------------------------------------------------------------------- get_dump with provenance
rep('''def get_dump(path, make, split, what, n=0):
    """a dump of the split: read and validated when it exists (never overwritten or made again), else made by `make()` (None: the path was named by a flag, a missing file is an error)"""
    if not os.path.exists(path):
        if make is None:
            sys.exit(f"{what}: {path} does not exist (a dump named by a flag is reused, never made)")
        make()
    return load_dump(path, split, what, n)''', '''def get_dump(path, make, split, what, n=0, expect=None):
    """a dump of the split: read and validated when it exists (never overwritten or made again), else made by `make()` (None: the path was named by a flag, a missing file is an error). `expect`: the provenance of a dump this module made
    (the default paths): an existing one is refused unless its sidecar says the same run, code, device and checkpoint; a dump named by a flag (E321's, the stored VAL dumps: validated, E321's also by hash) has none and is exempt"""
    if not os.path.exists(path):
        if make is None:
            sys.exit(f"{what}: {path} does not exist (a dump named by a flag is reused, never made)")
        make()
    elif expect is not None:
        check_provenance(path, expect, what)
    return load_dump(path, split, what, n)''')

# ---------------------------------------------------------------------------------------------------- dump_b / dump_ends
rep('''    import lane_gt_data as LD
    import qgraph_readout as RO
    import qpredict as QP
    import torch
    frames = frames_of(split)
    m = QP.load_model(os.path.join(ROOT, "runs", run), device)
    ds, dl = LD.loader(SPLITS[split], bs=4, workers=0, shuffle=False, map_mode="exact", pixels=False, strict=True)
    rows = [int(ds.bd.rows[k]) for k in ds.ks]
    assert rows == [f.row for f in frames], "the frame order of the loader differs from the scorer's"''', '''    head_path(out)                                                                                     # a dump name ends with .pkl; an existing dump is never overwritten
    refuse_existing(out)
    import lane_gt_data as LD
    import qgraph_readout as RO
    import qpredict as QP
    import torch
    frames = frames_of(split)
    m = QP.load_model(os.path.join(ROOT, "runs", run), device)
    ds, dl = LD.loader(SPLITS[split], bs=4, workers=0, shuffle=False, map_mode="exact", pixels=False, strict=True)
    rows = [int(ds.bd.rows[k]) for k in ds.ks]
    _require(rows == [f.row for f in frames], "the frame order of the loader differs from the scorer's")''')
rep('''    with atomic(out) as tmp:
        RO.dump_full(preds, rows[: len(preds)], tmp)
    print(f"dump-b {split} {run}''', '''    with atomic(out) as tmp:
        RO.dump_full(preds, rows[: len(preds)], tmp)
    write_head(out, dict(split=split, frames=len(preds)), provenance(run, device))
    print(f"dump-b {split} {run}''')
rep('''    the model reads the features only, the GT ends come from the lane GT"""
    import jends_eval as JE''', '''    the model reads the features only, the GT ends come from the lane GT. The head file next to it holds the recall / precision of the pair head and the provenance (nothing derived from the split's labels is printed)"""
    head_path(out)
    refuse_existing(out)
    import jends_eval as JE''')
rep('''    ds, dl = LD.loader(SPLITS[split], bs=8, workers=0, shuffle=False, map_mode="error", pixels=False, ends=True, strict=True)
    rows = [int(ds.bd.rows[k]) for k in ds.ks]
    assert rows == [f.row for f in frames], "the frame order of the loader differs from the scorer's"''', '''    ds, dl = LD.loader(SPLITS[split], bs=8, workers=0, shuffle=False, map_mode="error", pixels=False, ends=True, strict=True)
    rows = [int(ds.bd.rows[k]) for k in ds.ks]
    _require(rows == [f.row for f in frames], "the frame order of the loader differs from the scorer's")''')
rep('''    save_pickle(out, out_frames)
    save_json(out.replace(".pkl", "_head.json"), head)
    print(f"dump-ends {split} {run}: {len(out_frames)} frames -> {out}  {time.time() - t0:.0f} s; ent recall {head['ent_recall']:.3f} exit recall {head['ext_recall']:.3f} pair recall {head['pair_recall']:.3f}", flush=True)''',
    '''    save_pickle(out, out_frames)
    write_head(out, head, provenance(run, device))
    print(f"dump-ends {split} {run}: {len(out_frames)} frames -> {out}  {time.time() - t0:.0f} s", flush=True)''')
rep('''    assert keys, f"no pyramid tensor of {base_run} is in {run}: the pyramid check would be empty"''', '''    _require(keys, f"no pyramid tensor of {base_run} is in {run}: the pyramid check would be empty")''')

# ---------------------------------------------------------------------------------------------------- control_view
a = s.index("def control_numbers(card):")
b = s.index("def guards_vs(")
s = s[:a] + '''def control_view(card):
    """E312's snapped read-out from either card layout: E321's `card_s<S>.json` (variants.snapped) or a stored VAL card of `qends_eval.py score` -> dict(numbers = the snapped head's numbers, plain = the plain graph's, curves, wired)"""
    c = card["variants"]["snapped"] if "variants" in card else card
    n = c["numbers"]
    return dict(numbers={k: float(n["head"][k]) for k in KEYS}, plain={k: float(n["base"][k]) for k in KEYS}, curves=c["curves"], wired=c["wired_both_ends"])


''' + s[b:]

# ---------------------------------------------------------------------------------------------------- results_md: 6 digits in the reading table, provenance block
rep('''L.append(f"| {c['seed']} | {s['D']:+.4f} | " + (f"{s['lo']:+.4f} | {s['hi']:+.4f}" if s["lo"] is not None else "- | -")''',
    '''L.append(f"| {c['seed']} | {s['D']:+.6f} | " + (f"{s['lo']:+.6f} | {s['hi']:+.6f}" if s["lo"] is not None else "- | -")''')
rep('''    for c in cards:
        L.append(f"| {c['seed']} | " + " | ".join(f"{100 * c['maneuver'][x]['recall']:.1f} %" for x in labels) + " |")
    return "\\n".join(L) + "\\n"''', '''    for c in cards:
        L.append(f"| {c['seed']} | " + " | ".join(f"{100 * c['maneuver'][x]['recall']:.1f} %" for x in labels) + " |")
    if c0.get("device"):
        drift = c0.get("code_drift")
        L += ["", "## provenance", "", f"Device of the dumps and the scoring: {c0['device']}. Imported lane code: {len(c0['code_sha256'])} files, digest {c0['code_digest'][:16]}; " + (f"CODE DRIFT during the run: {', '.join(drift)}." if drift else "no change of those files during the run.")]
    return "\\n".join(L) + "\\n"''')

# ---------------------------------------------------------------------------------------------------- precheck, score_seed, check_stored, cmd_all, cmd_verify
a = s.index("def score_seed(")
b = s.rindex("\n# ---", 0, s.index("def selftest():")) + 1
s = s[:a] + open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "r2_block.py"), encoding="utf-8").read() + "\n" + s[b:]

open(P, "w", encoding="utf-8").write(s)
print("patched", len(s.splitlines()), "lines")
