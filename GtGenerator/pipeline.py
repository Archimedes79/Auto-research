"""The writer of the front GT: `generate` (THE entry point: read nuScenes from a folder, write the GT of every keyframe, resumable, memory-light, optionally with the lane GT and the masks)
and `build` (the low-level writer behind `report`). The files and the index come from `store.py`. `python -m GtGenerator generate ...` and `GtGenerator.generate(...)` are this module."""
import multiprocessing as mp
import os
import time
import types

from .grid import GridSpec
from .store import _flat, canonical_of, check_arguments, check_resume, detect_version, finish_index, front_keys, read_index, save_keyframe, write_index

_STATE = {}
GROUP_SCENES = 60            # sequences per devkit load: the memory-light devkit of 60 sequences needs 0.4 GB (the full trainval one 6.4 GB)


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
    from .suite import GtGenerator
    _STATE["gt"] = GtGenerator(dataroot, version, GridSpec(res=res), ss, light=scenes is not None, scenes=scenes, **options)


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


def _write_tokens(gt, tokens, sig_dir, workers, init_args, verbose):
    """write the keyframes `tokens` of the generator `gt` into sig_dir (existing files are skipped); -> the number written.
    workers > 1: a pool of spawned processes, each builds its own generator from init_args (see _init)."""
    t0 = time.time()
    done = 0
    if workers <= 1:
        _STATE["gt"] = gt
        try:
            for i, t in enumerate(tokens):
                done += _work((t, sig_dir))
                if verbose and i % 50 == 0:
                    print(f"{i}/{len(tokens)} keyframes, {time.time() - t0:.0f} s", flush=True)
        finally:
            _STATE.pop("gt", None)                                                       # the next group's generator is the only one alive
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
    from .suite import GtGenerator
    check_arguments(stride=stride, res=res, ss=ss, workers=workers)
    gt = GtGenerator(dataroot, version, GridSpec(res=res), ss, **options)
    sig_dir = os.path.join(out, canonical_of(gt.signature()))
    check_resume(sig_dir, gt.grid, ss)                                                               # raises SystemExit for a folder of another resolution (the contract of build; the class interface turns it into a ValueError)
    os.makedirs(sig_dir, exist_ok=True)
    names = gt.scenes() if scenes is None else list(dict.fromkeys(scenes))
    tokens = [t for s in names for t in gt.keyframes(s)[::stride]]
    before = read_index(sig_dir).get("complete", True)
    write_index(sig_dir, gt, version, names, stride, len(tokens), options, complete=False)       # before the first keyframe, also on a resume
    done = _write_tokens(gt, tokens, sig_dir, workers, (dataroot, version, res, ss, options), verbose)
    write_index(sig_dir, gt, version, names, stride, len(tokens), options, old_complete=before)
    if verbose:
        print(f"{done} written, {len(tokens) - done} already there: {sig_dir}")
    return done


def generate(dataroot, out=None, version=None, scenes=None, stride=1, res=0.5, ss=4, workers=1, verbose=True, set_name=None, lane=True, **options):
    """THE entry point of the suite: read nuScenes from `dataroot` and write the GT of every keyframe.

    dataroot   a nuScenes folder (it holds v1.0-mini/ or v1.0-trainval/, maps/, samples/, sweeps/); ~ is expanded
    out        where the GT goes: OUT/<generator signature>/<sample token>.npz + index.json; default data/autogt, the
               place `GtGenerator.load(token)` reads from (not in git)
    version    v1.0-mini / v1.0-trainval; default: the one found in dataroot (an error if there are several)
    scenes     names such as ["scene-0061"]; default all sequences of the version
    set_name   "small" (the 10 sequences of mini: small experiments) or "full" (all 850 of trainval: releases), the files of
               GtGenerator/sets/; instead of scenes (both given: an error)
    stride     every n-th keyframe (1 = all); res: cell size in m (0.5), ss: supersampling (4)
    workers    processes (1 = this process); options: use_flow, use_radar, boxes_at, ... (see GtGenerator)
    lane       also make the lane GT (lane graph, roads, lines, polygons: GtGenerator/lane_gt.py) of the same sequences into OUT/lane/<lane signature>/, the other
               store of the same release (releases.csv); only with stride 1, the lane GT is made for every keyframe
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
    from .suite import GtGenerator, DATA, code_signature, current_signature, json_tables, scene_tokens
    check_arguments(stride=stride, res=res, ss=ss, workers=workers, set_name=set_name, scenes=scenes)
    dataroot = os.path.expanduser(dataroot)
    version = version or detect_version(dataroot)
    out = os.path.expanduser(out) if out else DATA
    known = [s["name"] for s in json_tables(dataroot, version)[0]]
    names = sets.scenes(set_name) if set_name else known if scenes is None else list(dict.fromkeys(scenes))      # no duplicates
    unknown = [n for n in names if n not in known]
    if not names or unknown:
        raise ValueError(f"unknown scene(s) {unknown[:5]}{' ...' if len(unknown) > 5 else ''} in {version} of {dataroot}" if unknown else "no scenes given")
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
    stored = front_keys(sig_dir, nonempty=True)                       # a zero-byte file (full disk, a copy) is damage: written again
    size = min(GROUP_SCENES, max(1, -(-len(names) // workers)))           # with several workers every worker gets a group of its own
    groups = []
    for g0 in range(0, len(names), size):
        grp = names[g0:g0 + size]
        todo = [t for n in grp for t in tokens[n] if t not in stored]
        if todo:                                                          # a finished group needs no devkit load
            groups.append((f"group {g0 // size + 1}/{-(-len(names) // size)}", grp, todo))
    code_sig = code_signature(**options)
    if (groups or (lane and stride == 1)) and "trainval" in version:      # the big tables once, not once per group (31 s each, light_nusc.py): also for the lane GT that follows
        from .light_nusc import ensure_cache
        ensure_cache(dataroot, version, verbose=verbose)
    before = read_index(sig_dir).get("complete", True)                  # what the folder said before this run: the provisional index below must not make a finished run incomplete
    if groups:                                                        # a finished store is not flagged incomplete by a rerun that has nothing to do
        write_index(sig_dir, ident, version, names, stride, n_total, options, complete=False)       # before the first keyframe, also on a resume
    done = 0
    if workers <= 1 or len(groups) <= 1:
        for label, grp, todo in groups:
            gen = GtGenerator(dataroot, version, ident.grid, ss, light=True, scenes=grp, **options)
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
    idx, _ = finish_index(sig_dir, ident, version, names, stride, n_total, options, done, dataroot, set_name, old_complete=before)
    gt_version = idx.get("gt_version", "unreleased")
    if verbose:
        print(f"{done} keyframes written, {n_total - done} already there: {sig_dir}  (GT version {gt_version}, commit {idx.get('commit')}"
              f"{', dirty' if idx.get('dirty') else ''}{', RELEASE' if idx.get('release') else ''}, sets complete: {idx.get('sets_complete', [])})\n"
              f"read one with GtGenerator.load(<sample token>{'' if out == DATA else ', data=' + repr(out)})")
    lane_result = lane_rigs = masks_error = None
    if lane and stride == 1:
        from .gt_api import make_lane_gt                                    # both systems' lane GT + the masks derived from the BEV Fusion one: the same code as generate_bev_gt / generate_lane_gt
        lane_rigs, masks_error = make_lane_gt(dataroot, out, version, names, workers, verbose, masks=True, strict=False)      # a failing mask cache does not lose the GT that is written: it is reported (exit 1)
        lane_result = lane_rigs["lane_center"]
    elif lane and verbose:
        print("the lane GT is made for every keyframe: skipped with stride", stride)
    return {"out": sig_dir, "signature": ident.signature(), "gt_version": gt_version, "version": version, "scenes": len(names),
            "keyframes": n_total, "written": done, "release": idx.get("release", False), "lane": lane_result, "lane_rigs": lane_rigs, "masks_error": masks_error}

