"""Command line of the autoGT suite (from the repository root):

  python -m autogt selftest
  python -m autogt generate DATAROOT [--out DIR] [--version V] [--set small|full | --scenes scene-0061 ...] [--stride 1] [--res 0.5]
                          [--workers 1] [--no-flow] [--no-radar] [--boxes-at camera|annotation]
      THE entry point: reads nuScenes from DATAROOT (e.g. ~/data/nuscenes_full) and writes one compressed file per keyframe to
      DIR/<signature>/<sample token>.npz (float16 layers + meta) and an index.json (the manifest: gt_version, commit and dirty flag
      of the generator code, the sets held in full, release); DIR default data/autogt (where autogt.load reads); memory-light
      (groups of 60 sequences), resumable (existing files are skipped); the signature changes with every edit of the generator, so GT
      of an older generator is never mixed in
  python -m autogt data DATAROOT [--set small|full | --scenes ...]
      are the raw files the generator reads on disk (json tables, map expansion, CAM_FRONT, the five radars)? exit 1 if not
  python -m autogt have [--set small|full | --scenes ... | --scenes-file F] [--version V] [--stride N]
      is the stored GT of the current generator there for these sequences? exit 0 (all) or 1 (prints how many are missing): the check before
      a training; `generate ... --scenes-file F` fills what is missing (resumable)
  python -m autogt sets       the scene sets small (mini, small experiments) and full (trainval, releases) and how much of each is stored
  python -m autogt versions   the GT versions (versions.csv), the current one marked
  python -m autogt build  --version v1.0-mini --out DIR [--dataroot ~/data/nuscenes] ...   the low-level writer behind it (full devkit)
  python -m autogt check  [--version v1.0-mini] [--scenes ...] [--stride 1] [--holdout]
      automatic consistency checks of the GT over keyframes (see autogt/checks.py); --holdout adds the leave-one-radar-out test
  python -m autogt view   OUT_DIR [scene:keyframe | TOKEN ...] [--total]   sample pictures (camera, semantic BEV, objects + velocity,
      motion relative to the static world; --total shows the total apparent flow instead); scene:keyframe = nuScenes mini,
      generated now; TOKEN = a stored keyframe of mini or trainval (needs the stored GT of the current generator)
  python -m autogt sheets OUT_DIR                            overview sheets of the sample pictures
  python -m autogt video  SCENE OUT.mp4 [seconds] [--version v1.0-trainval] [--dataroot DIR]   driving through the GT at 12 Hz
  python -m autogt report [--version v1.0-mini|v1.0-trainval] [--out REPORT.pdf] [--data DIR | --no-data] [--scenes ..] [--stride N]
                          [--samples N] [--holdout] [--workers N] [--light]
      GT evaluation of all sequences in one pass: store the GT (as `build`), run every check, write ONE PDF report (summary, table
      per sequence, legend, sample pictures drawn straight into the PDF - no picture files exist, nothing to clean up but the report)
  python -m autogt clean  [--videos] [--pictures] [--reports] [--old-outputs] [--stale-caches DIR | --all-caches DIR] [--include-released]
                          [--older-than DAYS] [--move | --delete]
      dry run by default; --move puts the items into _to_delete/, --delete removes them; the caches of released GT versions are left
      alone unless --include-released, and are then moved, never deleted (see autogt/clean.py)
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import types

import numpy as np

from .grid import GridSpec

_STATE = {}


def _flat(g):
    """GT dict -> npz arrays (float16 for float32 layers) + meta as an object array."""
    arrs = {k: (v.astype(np.float16) if v.dtype == np.float32 else v) for k, v in g.items() if isinstance(v, np.ndarray)}
    arrs["meta"] = np.array(g["meta"], dtype=object)
    return arrs


def load_keyframe(sig_dir, token):
    """a keyframe written by `build` -> the GT dict (float32 layers + meta)."""
    out = {}
    with np.load(os.path.join(sig_dir, token + ".npz"), allow_pickle=True) as z:
        for k in z.files:
            a = z[k]                                                    # each member is decompressed once
            out[k] = a.item() if k == "meta" else a.astype(np.float32) if a.dtype == np.float16 else a
    return out


def save_keyframe(path, flat):
    """write one keyframe file atomically; the temporary name holds the process id, so two writers on one folder do not trip over each
    other (".tmp" in the name keeps it out of every listing of stored keyframes)"""
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **flat)
    os.replace(tmp, path)


def _work(args):
    token, sig_dir = args
    gt = _STATE["gt"]
    path = os.path.join(sig_dir, token + ".npz")
    if os.path.exists(path) and os.path.getsize(path) > 0:                  # an empty file (a full disk) is not a keyframe
        return 0
    save_keyframe(path, _flat(gt.keyframe(token)))
    return 1


def _init(dataroot, version, res, ss, options, scenes=None):
    """a worker's generator; scenes = names: the memory-light devkit of just these sequences (`generate`), else the full devkit"""
    from .suite import AutoGT
    _STATE["gt"] = AutoGT(dataroot, version, GridSpec(res=res), ss, light=scenes is not None, scenes=scenes, **options)


def run_manifest(sig_dir, version, dataroot, set_name=None):
    """what a finished run records about itself in index.json: the set asked for, the commit and dirty flag of autogt/, and the
    sets that the folder now holds in full (see sets.py)"""
    from . import sets, versions
    from .suite import code_signature
    return {"set": set_name, **versions.code_state(), "generator": code_signature(), "environment": _environment(),
            "sets_complete": sets.complete_in(sig_dir, version, dataroot), "sets_checked": sets.checkable(version, dataroot)}


def _environment():
    """the versions of the libraries whose behaviour shapes the stored bytes (the generator signature hashes our files, not these)"""
    from importlib import metadata
    out = {}
    for pkg in ("numpy", "shapely", "opencv-python", "matplotlib", "nuscenes-devkit", "pyquaternion"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "?"
    return out


def canonical_of(signature):
    from .versions import canonical
    return canonical(signature)


def write_index(sig_dir, gen, version, names, stride, n_keyframes, options, complete=True, run=None):
    """index.json of a cache folder (marks it as an autogt cache for `clean`, records what it holds). `build` writes it with
    complete=False before the first keyframe (so an interrupted build still says which resolution it holds) and again at the end.
    run = run_manifest(...) of a finished run adds the manifest keys set, commit, dirty, sets_complete and release (a complete folder
    whose signature is a released GT version, made from a clean tree, holding the set `full`); without it the keys of an earlier run
    on this folder stay."""
    names = list(names)
    try:
        with open(os.path.join(sig_dir, "index.json")) as f:
            old = json.load(f)
    except (OSError, ValueError):
        old = {}
    same = old.get("signature") == canonical_of(gen.signature()) and abs(old.get("res", gen.grid.res) - gen.grid.res) < 1e-9 and old.get("ss", gen.ss) == gen.ss
    if same and not set(old.get("scenes", [])) <= set(names):
        # mini is a subset of trainval and both share the folder of their signature: a smaller run (the mini report after the trainval
        # one) must not shrink the record of what the folder holds, nor mark an interrupted bigger run as complete
        if set(old["scenes"]) >= set(names):
            version = old["version"]
        names = sorted(set(old["scenes"]) | set(names))
        complete = complete and old.get("complete", True)
    if same and old.get("stride", stride) != stride:
        stride = "mixed"                                           # runs with different strides wrote into this folder
    if complete and os.path.isdir(sig_dir):
        from .sets import held
        n_keyframes = len(held(sig_dir))                           # what the folder holds, whatever this run wanted
    from .versions import canonical, version_of
    store_sig = canonical(gen.signature())                         # the folder of the GT this code makes (see `same_gt_as` in versions.csv)
    gt_version = version_of(store_sig)                             # None: a signature without a row in versions.csv (unreleased): no key
    idx = {"signature": store_sig, **({"gt_version": gt_version} if gt_version else {}), "version": version, "res": gen.grid.res,
           "ss": gen.ss, "options": {k: repr(v) for k, v in options.items()}, "scenes": names, "stride": stride, "keyframes": n_keyframes,
           "complete": complete}
    kept = old if same else {}                                     # files are only added, so a set that was complete stays complete
    # the sets this run could check are taken from its fresh list (a deleted keyframe removes `full`); a set that this nuScenes version cannot
    # check (small is in mini, full is not) keeps its old entry
    sets_complete = sorted((set(kept.get("sets_complete", [])) - set((run or {}).get("sets_checked", []))) | set((run or {}).get("sets_complete", [])))
    if run is None:
        idx.update({k: kept[k] for k in ("set", "generator", "environment", "commit", "dirty", "release") if k in kept})
        if "release" in idx and not complete:
            idx["release"] = False                                 # a run in progress (its provisional index) is not a release
    else:
        idx.update(set=run.get("set"), generator=run.get("generator"), environment=run.get("environment"), commit=run.get("commit"), dirty=run.get("dirty"),
                   release=bool(complete and gt_version and run.get("dirty") is False and "full" in sets_complete))
    if sets_complete:
        idx["sets_complete"] = sets_complete
    tmp = os.path.join(sig_dir, f"index.json.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(idx, f, indent=1)
    os.replace(tmp, os.path.join(sig_dir, "index.json"))


def _group_task(args):
    """one worker, one group of sequences: its memory-light devkit is loaded once (that takes minutes on a busy box), then its missing
    keyframes are written one after the other -> (label, keyframes written, seconds)"""
    code_sig, label, dataroot, version, res, ss, options, grp, todo, sig_dir, verbose = args
    t0 = time.time()
    _init(dataroot, version, res, ss, options, grp)
    _same_generator(_STATE["gt"], code_sig)
    done = 0
    for i, t in enumerate(todo):
        done += _work((t, sig_dir))
        if verbose and i % 500 == 0:
            print(f"[{label}] {i}/{len(todo)} keyframes, {time.time() - t0:.0f} s", flush=True)
    return label, done, time.time() - t0


def _same_generator(gen, code_sig):
    """a job that runs for hours must not mix two generators: a worker imports the source tree afresh, so a hashed file edited meanwhile
    would write the later groups with new code into the folder of the old signature"""
    if gen.signature() != code_sig:
        raise RuntimeError(f"a generator file changed while this job runs (signature {code_sig} at the start, {gen.signature()} now): "
                           "stop and start it again from a clean tree (a worktree of the tag)")


def check_resume(sig_dir, grid, ss):
    """The folder is named by the generator signature, which contains neither the resolution nor the supersampling: continuing a
    build of another one would reuse its keyframes silently. The index says what the folder holds. A folder of an interrupted build
    from before the provisional index has none: one stored keyframe is compared with the grid instead."""
    idx_path = os.path.join(sig_dir, "index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            held = json.load(f)
        if abs(held.get("res", grid.res) - grid.res) > 1e-9 or held.get("ss", ss) != ss:
            raise SystemExit(f"{sig_dir} holds res {held.get('res')} / ss {held.get('ss')} (asked: res {grid.res} / ss {ss}): "
                             "build into another output folder")
        return
    stored = [f for f in os.listdir(sig_dir) if f.endswith(".npz") and ".tmp" not in f] if os.path.isdir(sig_dir) else []
    if stored:
        with np.load(os.path.join(sig_dir, stored[0]), allow_pickle=True) as z:
            shape = tuple(z["drivable_area"].shape) if "drivable_area" in z.files else None
        if shape is not None and shape != (grid.nx, grid.ny):
            raise SystemExit(f"{sig_dir} has no index.json (an interrupted build) and its keyframes have the cell grid {shape}, not "
                             f"{(grid.nx, grid.ny)} (res {grid.res}): build into another output folder")


def _write_tokens(gt, tokens, sig_dir, workers, init_args, verbose):
    """write the keyframes `tokens` of the generator `gt` into sig_dir (existing files are skipped); -> the number written.
    workers > 1: a pool of spawned processes, each builds its own generator from init_args (see _init)."""
    t0 = time.time()
    done = 0
    if workers <= 1:
        _STATE["gt"] = gt
        for i, t in enumerate(tokens):
            done += _work((t, sig_dir))
            if verbose and i % 50 == 0:
                print(f"{i}/{len(tokens)} keyframes, {time.time() - t0:.0f} s", flush=True)
    else:
        with mp.get_context("spawn").Pool(workers, _init, init_args) as pool:
            for i, r in enumerate(pool.imap_unordered(_work, [(t, sig_dir) for t in tokens], chunksize=4)):
                done += r
                if verbose and i % 200 == 0:
                    print(f"{i}/{len(tokens)} keyframes, {time.time() - t0:.0f} s", flush=True)
    return done


def build(dataroot, version, out, scenes=None, stride=1, workers=1, res=0.5, ss=4, verbose=True, **options):
    """the low-level writer (one full devkit, `out` required; `generate` is the entry point): returns the number of keyframes written
    (existing ones are skipped)."""
    from .suite import AutoGT
    if stride < 1:
        raise ValueError(f"stride must be 1 or more, got {stride}")
    gt = AutoGT(dataroot, version, GridSpec(res=res), ss, **options)
    sig_dir = os.path.join(out, canonical_of(gt.signature()))
    check_resume(sig_dir, gt.grid, ss)
    os.makedirs(sig_dir, exist_ok=True)
    names = gt.scenes() if scenes is None else list(dict.fromkeys(scenes))
    tokens = [t for s in names for t in gt.keyframes(s)[::stride]]
    write_index(sig_dir, gt, version, names, stride, len(tokens), options, complete=False)       # before the first keyframe, also on a resume
    done = _write_tokens(gt, tokens, sig_dir, workers, (dataroot, version, res, ss, options), verbose)
    write_index(sig_dir, gt, version, names, stride, len(tokens), options)
    if verbose:
        print(f"{done} written, {len(tokens) - done} already there: {sig_dir}")
    return done


GROUP_SCENES = 60            # sequences per devkit load: the memory-light devkit of 60 sequences needs 0.4 GB (the full trainval one 6.4 GB)


def detect_version(dataroot):
    """the nuScenes version inside `dataroot`: the one `v1.0-*` folder that holds a scene.json"""
    root = os.path.expanduser(dataroot)
    found = sorted(d for d in os.listdir(root) if d.startswith("v1.0-") and os.path.exists(os.path.join(root, d, "scene.json"))) \
        if os.path.isdir(root) else []
    if len(found) != 1:
        raise ValueError(f"{root}: " + (f"several nuScenes versions {found}, pass version=" if found else "no v1.0-*/scene.json: not a nuScenes folder"))
    return found[0]


def generate(dataroot, out=None, version=None, scenes=None, stride=1, res=0.5, ss=4, workers=1, verbose=True, set_name=None, **options):
    """THE entry point of the suite: read nuScenes from `dataroot` and write the GT of every keyframe.

    dataroot   a nuScenes folder (it holds v1.0-mini/ or v1.0-trainval/, maps/, samples/, sweeps/); ~ is expanded
    out        where the GT goes: OUT/<generator signature>/<sample token>.npz + index.json; default data/autogt, the
               place `autogt.load(token)` reads from (not in git)
    version    v1.0-mini / v1.0-trainval; default: the one found in dataroot (an error if there are several)
    scenes     names such as ["scene-0061"]; default all sequences of the version
    set_name   "small" (the 10 sequences of mini: small experiments) or "full" (all 850 of trainval: releases), the files of
               autogt/sets/; instead of scenes (both given: an error)
    stride     every n-th keyframe (1 = all); res: cell size in m (0.5), ss: supersampling (4)
    workers    processes (1 = this process); options: stationary_fix, use_flow, use_radar, boxes_at, ... (see AutoGT)
    -> {"out": the folder written, "signature", "gt_version" (the key of versions.csv, D29), "version", "scenes", "keyframes": wanted,
        "written": new files, "release": the folder is a release (see write_index)}
    The index.json of the folder is the manifest: gt_version, commit and dirty flag of the generator code, the sets it holds in full.

    Memory-light and resumable: the sequences are read in groups of GROUP_SCENES (fewer with several workers: every worker takes a group
    of its own, so the devkit of a group is loaded once) with the light devkit (the GT is identical to the
    full devkit's), files that exist are skipped, so an interrupted run continues where it stopped. A folder of another resolution
    or supersampling is never continued. The signature changes with every edit of the generator, so GT of an older generator is
    never mixed in. Needs no GPU; trainval takes 5 - 8 hours in one process."""
    import gc
    from . import sets
    from .suite import AutoGT, DATA, code_signature, current_signature, json_tables, scene_tokens
    if set_name and scenes is not None:
        raise ValueError(f"pass either scenes= or set_name= ({set_name!r}), not both")
    if stride < 1:
        raise ValueError(f"stride must be 1 or more, got {stride}")
    dataroot = os.path.expanduser(dataroot)
    version = version or detect_version(dataroot)
    out = os.path.expanduser(out) if out else DATA
    known = [s["name"] for s in json_tables(dataroot, version)[0]]
    names = sets.scenes(set_name) if set_name else known if scenes is None else list(dict.fromkeys(scenes))      # no duplicates
    if not names or [n for n in names if n not in known]:
        raise ValueError(f"unknown scene(s) {[n for n in names if n not in known] or 'none given'} in {version} of {dataroot}")
    tokens = {n: scene_tokens(n, version, dataroot)[::stride] for n in names}                       # json tables only, no devkit
    n_total = sum(len(t) for t in tokens.values())
    sig = current_signature(**options)                                                              # the folder name, without the devkit
    sig_dir = os.path.join(out, sig)
    ident = types.SimpleNamespace(signature=lambda: sig, grid=GridSpec(res=res), ss=ss)             # what write_index needs, without the devkit
    try:
        check_resume(sig_dir, ident.grid, ss)
    except SystemExit as e:                                                                          # a library call raises, it does not exit
        raise ValueError(str(e)) from None
    os.makedirs(sig_dir, exist_ok=True)
    stored = sets.held(sig_dir)
    size = min(GROUP_SCENES, max(1, -(-len(names) // workers)))           # with several workers every worker gets a group of its own
    groups = []
    for g0 in range(0, len(names), size):
        grp = names[g0:g0 + size]
        todo = [t for n in grp for t in tokens[n] if t not in stored]
        if todo:                                                          # a finished group needs no devkit load
            groups.append((f"group {g0 // size + 1}/{-(-len(names) // size)}", grp, todo))
    code_sig = code_signature(**options)
    if groups:                                                        # a finished store is not flagged incomplete by a rerun that has nothing to do
        write_index(sig_dir, ident, version, names, stride, n_total, options, complete=False)       # before the first keyframe, also on a resume
    done = 0
    if workers <= 1 or len(groups) <= 1:
        for label, grp, todo in groups:
            gen = AutoGT(dataroot, version, ident.grid, ss, light=True, scenes=grp, **options)
            _same_generator(gen, code_sig)
            done += _write_tokens(gen, todo, sig_dir, 1, None, verbose)
            del gen
            gc.collect()
    else:                       # one group per worker: the devkit of a group is loaded once, by the worker that writes it, not once per worker
        jobs = [(code_sig, label, dataroot, version, res, ss, options, grp, todo, sig_dir, verbose) for label, grp, todo in groups]
        with mp.get_context("spawn").Pool(min(workers, len(jobs)), maxtasksperchild=1) as pool:
            for label, n, secs in pool.imap_unordered(_group_task, jobs):
                done += n
                if verbose:
                    print(f"{label} done: {n} keyframes written in {secs:.0f} s", flush=True)
    write_index(sig_dir, ident, version, names, stride, n_total, options, run=run_manifest(sig_dir, version, dataroot, set_name))
    with open(os.path.join(sig_dir, "index.json")) as f:
        idx = json.load(f)
    gt_version = idx.get("gt_version", "unreleased")
    if verbose:
        print(f"{done} keyframes written, {n_total - done} already there: {sig_dir}  (GT version {gt_version}, commit {idx['commit']}"
              f"{', dirty' if idx['dirty'] else ''}{', RELEASE' if idx['release'] else ''}, sets complete: {idx.get('sets_complete', [])})\n"
              f"read one with autogt.load(<sample token>{'' if out == DATA else ', data=' + repr(out)})")
    return {"out": sig_dir, "signature": ident.signature(), "gt_version": gt_version, "version": version, "scenes": len(names),
            "keyframes": n_total, "written": done, "release": idx["release"]}


def _print_versions():
    """the table of versions.csv: key, signature, date, CURRENT for this source, stored when its folder exists in data/autogt"""
    from . import versions
    from .suite import DATA, code_signature
    cur, rows = code_signature(), versions.table()
    for r in rows:
        same = f" (the GT of {r['same_gt_as']})" if r.get("same_gt_as") else ""
        print(f"{r['version']:3s} {r['signature']}  {r['date']}  {'CURRENT' if r['signature'] == cur else '       '}  "
              f"{'stored    ' if os.path.isdir(os.path.join(DATA, versions.canonical(r['signature']))) else 'not stored'}  {r['tag']:11s} "
              f"{r['what_changed'][:70]}{same}")
    if cur not in {r["signature"] for r in rows}:
        print(f"this source has the signature {cur}: no row in autogt/versions.csv (unreleased); add a row, commit, tag autogt/v<N>")


def _print_sets():
    """each scene set: its sequences, keyframes, and how much of it the store of the current generator holds"""
    from . import sets
    for name in sets.names():
        try:
            s = sets.status(name)
        except ValueError as e:
            print(f"{name:6s} {len(sets.scenes(name)):4d} scenes  not readable here: {e}")
            continue
        print(f"{name:6s} {len(sets.scenes(name)):4d} scenes  {s['total']:6d} keyframes ({s['version']})  stored for this generator: "
              f"{s['have']}/{s['total']}{'  COMPLETE' if s['complete'] else ''}")


def _scene_names(a):
    """the sequence names asked for on the command line: --scenes, --scenes-file (one name per line, # comments) or None (--set / all)"""
    if getattr(a, "scenes_file", None):
        if a.scenes:
            raise ValueError("pass either --scenes or --scenes-file, not both")
        with open(a.scenes_file, encoding="utf-8") as f:
            return [x for x in (ln.split("#")[0].strip() for ln in f) if x]
    return a.scenes


def _options(a):
    """the generator options of the command line (flow and radar on unless switched off; boxes_at only when it is not the default)"""
    return {"use_flow": not a.no_flow, "use_radar": not a.no_radar, **({} if a.boxes_at == "camera" else {"boxes_at": a.boxes_at})}


def _generator_args():
    """the options that `generate`, `build` and `report` share (a parent parser)"""
    g = argparse.ArgumentParser(add_help=False)
    g.add_argument("--res", type=float, default=0.5)
    g.add_argument("--stride", type=int, default=1, help="every n-th keyframe")
    g.add_argument("--workers", type=int, default=1, help="processes")
    g.add_argument("--no-flow", action="store_true")
    g.add_argument("--no-radar", action="store_true")
    g.add_argument("--boxes-at", choices=("camera", "annotation"), default="camera",
                   help="time of the moving boxes: the camera time (default) or the annotation time")
    return g


def _parser():
    p = argparse.ArgumentParser(prog="python -m autogt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    gen = _generator_args()
    sub.add_parser("selftest")
    sub.add_parser("versions", help="the GT versions (autogt/versions.csv, D29): the current one marked, which GT is stored")
    sub.add_parser("sets", help="the scene sets small / full (autogt/sets/): sequences, keyframes, how much of each is stored")
    da = sub.add_parser("data", help="check that the raw nuScenes files the generator reads are on disk (tables, maps, CAM_FRONT, radars)")
    da.add_argument("dataroot", help="a nuScenes folder, e.g. ~/data/nuscenes_full")
    da.add_argument("--version", default=None)
    da.add_argument("--set", dest="set_name", default=None, help="check only the sequences of a set: small | full")
    da.add_argument("--scenes", nargs="*")
    ge = sub.add_parser("generate", parents=[gen], help="read nuScenes from DATAROOT and write the GT of every keyframe (the entry point)")
    ge.add_argument("dataroot", help="a nuScenes folder, e.g. ~/data/nuscenes_full (it holds v1.0-trainval/, maps/, samples/, sweeps/)")
    ge.add_argument("--out", default=None, help="folder of the stored GT (default data/autogt); the files go to OUT/<signature>/")
    ge.add_argument("--version", default=None, help="v1.0-mini | v1.0-trainval (default: the one found in DATAROOT)")
    ge.add_argument("--set", dest="set_name", default=None, help="a scene set: small (the 10 sequences of mini) or full (all of trainval)")
    ge.add_argument("--scenes", nargs="*", help="names such as scene-0061 (default: all)")
    ge.add_argument("--scenes-file", help="a file of sequence names, one per line (instead of --scenes)")
    dg = sub.add_parser("digest", help="a digest per frame of nuScenes mini: two generators that make the same GT give the same file (same_gt_as)")
    dg.add_argument("out", nargs="?", help="write the digests to this json file")
    dg.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two digest files: exit 1 when any frame differs")
    dg.add_argument("--stride", type=int, default=2)
    hv = sub.add_parser("have", help="is the stored GT of the current generator there for these sequences? exit 0 or 1 (the check before a training)")
    hv.add_argument("--set", dest="set_name", default=None, help="small | full")
    hv.add_argument("--scenes", nargs="*")
    hv.add_argument("--scenes-file", help="a file of sequence names, one per line")
    hv.add_argument("--version", default=None, help="v1.0-mini | v1.0-trainval (default: the first that holds all the sequences)")
    hv.add_argument("--dataroot", default=None)
    hv.add_argument("--stride", type=int, default=1)
    hv.add_argument("--data", default=None, help="the store folder (default data/autogt)")
    b = sub.add_parser("build", parents=[gen], help="the low-level writer of generate (one full devkit, --out required)")
    b.add_argument("--dataroot", default="~/data/nuscenes")
    b.add_argument("--version", default="v1.0-mini")
    b.add_argument("--out", required=True)
    b.add_argument("--scenes", nargs="*")
    c = sub.add_parser("check")
    c.add_argument("--dataroot", default="~/data/nuscenes")
    c.add_argument("--version", default="v1.0-mini")
    c.add_argument("--scenes", nargs="*")
    c.add_argument("--stride", type=int, default=1)
    c.add_argument("--holdout", action="store_true")
    v = sub.add_parser("view")
    v.add_argument("out")
    v.add_argument("picks", nargs="*", help="scene-0061:8 (mini, generated now) or a sample token (stored GT, mini or trainval)")
    v.add_argument("--total", action="store_true", help="4th panel: total apparent flow (ego motion included) instead of motion relative to the static world")
    sub.add_parser("sheets").add_argument("out")
    gl = sub.add_parser("generate-lane", help="the lane GT (lane graph, roads, lines; own version and store) of a nuScenes folder: resumable")
    gl.add_argument("dataroot", help="a nuScenes folder, e.g. ~/data/nuscenes_full")
    gl.add_argument("--out", default=None, help="default data/autogt/lane; the files go to OUT/<lane signature>/")
    gl.add_argument("--version", default=None)
    gl.add_argument("--set", dest="set_name", default=None, help="small (the 10 sequences of mini) or full (all of trainval)")
    gl.add_argument("--scenes", nargs="*")
    gl.add_argument("--workers", type=int, default=1)
    lgp = sub.add_parser("lanegraph", help="picture of the lane graph of a keyframe: the six cameras and the top view")
    lgp.add_argument("out", help="the png to write")
    lgp.add_argument("pick", help="scene-0061:8 (the scene's 9th keyframe)")
    lgp.add_argument("--version", default="v1.0-mini")
    lgp.add_argument("--dataroot", default=None, help="default ~/data/nuscenes (mini) or ~/data/nuscenes_full (trainval)")
    r = sub.add_parser("report", parents=[gen], help="evaluate the GT of all sequences, store it, write one PDF report with samples")
    r.add_argument("--dataroot", default=None, help="default ~/data/nuscenes (mini) or ~/data/nuscenes_full (trainval)")
    r.add_argument("--version", default="v1.0-mini")
    r.add_argument("--light", action="store_true", help="memory-light devkit (automatic for trainval)")
    r.add_argument("--out", default=None, help="report PDF (default data/autogt/reports/autogt_report_<version>_<date>.pdf)")
    r.add_argument("--data", default=None, help="cache directory (default data/autogt); --no-data: do not store")
    r.add_argument("--no-data", action="store_true")
    r.add_argument("--scenes", nargs="*")
    r.add_argument("--samples", type=int, default=10, help="sample pictures in the report (evenly spread over the sequences)")
    r.add_argument("--holdout", action="store_true", help="add the leave-one-radar-out test of the radar correction")
    cl = sub.add_parser("clean", help="list / move / delete generated videos, pictures, old outputs, stale caches")
    for flag in ("--videos", "--pictures", "--reports", "--old-outputs", "--move", "--delete"):
        cl.add_argument(flag, action="store_true")
    cl.add_argument("--all-caches", metavar="DIR")
    cl.add_argument("--stale-caches", metavar="DIR")
    cl.add_argument("--include-released", action="store_true", help="also list the caches of released GT versions (versions.csv): --move only")
    cl.add_argument("--older-than", type=float, metavar="DAYS")
    d = sub.add_parser("video", help="drive through the GT of one sequence at 12 Hz (H.264); mini by default, any trainval sequence with --version")
    d.add_argument("scene")
    d.add_argument("mp4")
    d.add_argument("seconds", nargs="?", type=float, default=0.0)
    d.add_argument("--version", default="v1.0-mini", help="v1.0-mini | v1.0-trainval")
    d.add_argument("--dataroot", default=None, help="default ~/data/nuscenes (mini) or ~/data/nuscenes_full (trainval)")
    return p


def _cmd_selftest(a):
    from .selftest import run
    run()


def _cmd_data(a):
    from . import sets
    r = sets.check_raw(a.dataroot, a.version, a.set_name, a.scenes)
    print(sets.report_raw(r))
    if r["missing"] or r["maps"]:
        raise SystemExit(1)


def _cmd_generate(a):
    generate(a.dataroot, a.out, a.version, _scene_names(a), a.stride, a.res, workers=a.workers, set_name=a.set_name, **_options(a))


def _cmd_digest(a):
    import json
    if a.compare:
        x, y = (json.load(open(p)) for p in a.compare)
        bad = [k for k in x["digests"] if x["digests"][k] != y["digests"].get(k)] + [k for k in y["digests"] if k not in x["digests"]]
        print(f"{len(x['digests'])} frames, signatures {x['signature']} and {y['signature']}: {len(bad)} differ {bad[:3]}")
        if bad:
            raise SystemExit(1)
        return
    if not a.out:
        raise ValueError("give the file to write, or --compare A B")
    from .checks import digests
    r = digests("~/data/nuscenes", "v1.0-mini", a.stride)
    json.dump(r, open(a.out, "w"))
    print(f"{len(r['digests'])} digests of signature {r['signature']} -> {a.out}")


def _cmd_have(a):
    from . import current_version, sets
    want = sets.scenes(a.set_name) if a.set_name else _scene_names(a)
    if not want:
        raise ValueError("give --set, --scenes or --scenes-file")
    r = sets.stored_for(want, a.data, a.version, a.dataroot, a.stride)
    print(f"{r['have']}/{r['total']} keyframes of {r['scenes']} sequences ({r['version']}) are stored for GT version "
          f"{current_version()}; missing {len(r['missing'])}")
    if r["missing"]:
        raise SystemExit(1)


def _cmd_build(a):
    build(os.path.expanduser(a.dataroot), a.version, a.out, a.scenes, a.stride, a.workers, a.res, **_options(a))


def _cmd_check(a):
    from .checks import run
    run(os.path.expanduser(a.dataroot), a.version, a.scenes, a.stride, a.holdout)


def _cmd_view(a):
    from .view import run
    run(a.out, [x.split(":") for x in a.picks] or None, total=a.total)


def _cmd_lanegraph(a):
    import cv2
    from . import lane_graph_view as V
    from .light_nusc import light_nuscenes
    from .suite import AutoGT, default_root
    root = os.path.expanduser(a.dataroot or default_root(a.version))
    name, k = a.pick.split(":")
    from .lane_gt import LaneGT
    nusc = light_nuscenes(root, a.version, [name], cameras=V.CAMERAS, radars=False)
    gt = AutoGT(root, a.version, nusc=nusc)
    tok = gt.keyframes(name)[int(k)]
    g = LaneGT(root, a.version, nusc=nusc).graph(tok)
    cv2.imwrite(a.out, V.mosaic(gt, tok, g, title=f"{name}:{k} {g['location']}  " + V.legend_text(g)))
    print(f"{a.out}: {V.legend_text(g)}")


def _cmd_generate_lane(a):
    from .lane_gt import generate
    generate(a.dataroot, a.out, a.version, a.scenes, a.set_name, a.workers)


def _cmd_sheets(a):
    from .view import sheets
    sheets(a.out)


def _cmd_report(a):
    import datetime
    from .report import report
    from .suite import DATA, default_root
    out = a.out or os.path.join(DATA, "reports", f"autogt_report_{a.version}_{datetime.date.today().isoformat()}.pdf")
    data = None if a.no_data else (a.data or DATA)
    report(os.path.expanduser(a.dataroot or default_root(a.version)), a.version, out, data, a.scenes, a.stride, a.holdout, a.samples,
           a.workers, a.light or "trainval" in a.version, **_options(a))


def _cmd_clean(a):
    from .clean import clean
    clean(delete=a.delete, move=a.move, videos=a.videos, pictures=a.pictures, reports=a.reports, old_outputs=a.old_outputs,
          stale_caches=a.stale_caches, all_caches=a.all_caches, older_than=a.older_than, include_released=a.include_released)


def _cmd_video(a):
    from .suite import default_root
    from .video import run
    run(a.scene, a.mp4, a.seconds, dataroot=a.dataroot or default_root(a.version), version=a.version)


COMMANDS = {"selftest": _cmd_selftest, "versions": lambda a: _print_versions(), "sets": lambda a: _print_sets(), "data": _cmd_data, "digest": _cmd_digest, "have": _cmd_have,
            "generate": _cmd_generate, "build": _cmd_build, "check": _cmd_check, "view": _cmd_view, "sheets": _cmd_sheets, "lanegraph": _cmd_lanegraph, "generate-lane": _cmd_generate_lane,
            "report": _cmd_report, "clean": _cmd_clean, "video": _cmd_video}
CHECKS_ARGUMENTS = ("data", "digest", "generate", "generate-lane", "have", "clean")           # these commands raise ValueError for a wrong argument: the message, no traceback


def main(argv=None):
    a = _parser().parse_args(argv)
    try:
        COMMANDS[a.cmd](a)
    except ValueError as e:
        if a.cmd not in CHECKS_ARGUMENTS:
            raise
        raise SystemExit(f"error: {e}") from None


if __name__ == "__main__":
    main()
