"""Storage of the front GT: the keyframe files (`<signature>/<sample token>.npz`, float16 layers + the meta as a pickled object), the index.json of a store folder (the manifest: what it holds,
which generator made it, whether it is a release), the resume checks, and the listing of a store folder by file name. No argparse, no devkit: everything that reads or writes a store comes here
(`pipeline.py` writes with it, `suite.py` / `report.py` / `sets.py` read with it; the hashed `lane_gt.py` uses `names`, `read_index`, `environment` and `detect_version` of it).
"""
import json
import os

import numpy as np


def names(folder, suffix, nonempty=False):
    """the keys of the files `<key><suffix>` in a store folder, by name only (no stat: 34k stat calls cost 80 s on the 9p mounts one after the other, a scandir 0.2 s; a file that is being written has a
    `.tmp` name: every writer renames it into place). `nonempty` also drops a zero-byte file (a full disk, a bad copy): the stats then run on 16 threads, the wait is the round trip, not the CPU
    (12 - 16 s for a store of 34k files); a writer of this suite never leaves one, so the check is for the questions that must not guess (`generate`, the GT key)"""
    if not os.path.isdir(folder):
        return set()
    with os.scandir(folder) as it:
        files = [e.path for e in it if e.name.endswith(suffix) and ".tmp" not in e.name]
    if nonempty:
        if len(files) < 64:
            sizes = [os.path.getsize(f) for f in files]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(16) as pool:
                sizes = list(pool.map(os.path.getsize, files, chunksize=256))
        files = [f for f, n in zip(files, sizes) if n > 0]
    return {os.path.basename(f)[:-len(suffix)] for f in files}


def front_keys(sig_dir, nonempty=False):
    """the sample tokens with a stored front GT in the folder (see `names`; `generate` asks with nonempty=True and rewrites a zero-byte file)"""
    return names(sig_dir, ".npz", nonempty)


def check_arguments(stride=1, res=0.5, ss=4, workers=1, set_name=None, scenes=None):
    """the arguments of generate / build / report that would otherwise end in a ZeroDivisionError or a silent wrong run -> ValueError with the message"""
    if stride < 1:
        raise ValueError(f"stride must be 1 or more, got {stride}")
    if res <= 0 or ss < 2 or ss % 2:
        raise ValueError(f"res must be positive and ss even (2, 4, ...), got res {res}, ss {ss}")
    if workers < 1:
        raise ValueError(f"workers must be 1 or more, got {workers}")
    if set_name and scenes is not None:
        raise ValueError(f"pass either scenes= or set_name= ({set_name!r}), not both")


def read_index(folder):
    """the index.json of a store folder as a dict ({} when there is none or it is damaged)"""
    try:
        with open(os.path.join(folder, "index.json")) as f:
            idx = json.load(f)
    except (OSError, ValueError):
        return {}
    return idx if isinstance(idx, dict) else {}


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


def run_manifest(sig_dir, version, dataroot, set_name=None, options=None):
    """what a finished run records about itself in index.json: the set asked for, the commit and dirty flag of GtGenerator/, and the
    sets that the folder now holds in full (see sets.py)"""
    from . import sets, versions
    from .suite import code_signature
    return {"set": set_name, **versions.code_state(), "generator": code_signature(**(options or {})), "environment": environment(),
            "sets_complete": sets.complete_in(sig_dir, version, dataroot), "sets_checked": sets.checkable(version, dataroot),
            "suite_release": versions.current_release()}


def environment():
    """the versions of the libraries whose behaviour shapes the stored bytes (the generator signature hashes our files, not these); the index.json of a store records them"""
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


def write_index(sig_dir, gen, version, names, stride, n_keyframes, options, complete=True, run=None, old_complete=None):
    """index.json of a cache folder (marks it as a GtGenerator cache for `clean`, records what it holds). `build` writes it with
    complete=False before the first keyframe (so an interrupted build still says which resolution it holds) and again at the end.
    run = run_manifest(...) of a finished run adds the manifest keys set, commit, dirty, sets_complete and release (a complete folder
    whose signature is a released GT version, made from a clean tree, holding the set `full`); without it the keys of an earlier run
    on this folder stay. old_complete = the `complete` of the index before this run's provisional write (the provisional index of this very run says False and must not make a finished run incomplete)."""
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
        complete = complete and (old.get("complete", True) if old_complete is None else old_complete)      # old_complete: what the folder said BEFORE this run wrote its provisional index
    if same and old.get("stride", stride) != stride:
        stride = "mixed"                                           # runs with different strides wrote into this folder
    if complete and os.path.isdir(sig_dir):
        n_keyframes = len(front_keys(sig_dir, nonempty=True))      # what the folder holds, whatever this run wanted
    from .versions import canonical, version_of
    store_sig = canonical(gen.signature())                         # the folder of the GT this code makes (see `same_gt_as` in versions.csv)
    gt_version = version_of(store_sig)                             # None: a signature without a row in versions.csv (unreleased): no key
    idx = {"signature": store_sig, **({"gt_version": gt_version} if gt_version else {}), "version": version, "res": gen.grid.res,
           "ss": gen.ss, "options": {k: repr(v) for k, v in options.items()}, "scenes": names, "stride": stride, "keyframes": n_keyframes,
           "complete": complete}
    kept = old if same else {}                                     # files are only added, so a set that was complete stays complete
    from .gt_api import data_identity                              # the identity of the raw nuScenes files this GT was made from (the GT key reads it back)
    was, now = kept.get("data_identity"), data_identity()
    data_id = now if not was or was == "unknown" else was if now in ("unknown", was) else "mixed"      # files written under another manifest.json than the first ones: the folder says so, the first never wins silently
    if data_id != "unknown":
        idx["data_identity"] = data_id
    # the sets this run could check are taken from its fresh list (a deleted keyframe removes `full`); a set that this nuScenes version cannot
    # check (small is in mini, full is not) keeps its old entry
    sets_complete = sorted((set(kept.get("sets_complete", [])) - set((run or {}).get("sets_checked", []))) | set((run or {}).get("sets_complete", [])))
    if run is None:
        idx.update({k: kept[k] for k in ("set", "generator", "environment", "commit", "dirty", "release", "suite_release") if k in kept})
        if "release" in idx and not complete:
            idx["release"] = False                                 # a run in progress (its provisional index) is not a release
    else:
        idx.update(set=run.get("set"), generator=run.get("generator"), environment=run.get("environment"), commit=run.get("commit"), dirty=run.get("dirty"),
                   suite_release=run.get("suite_release"),
                   release=bool(complete and gt_version and run.get("dirty") is False and "full" in sets_complete))
    if sets_complete:
        idx["sets_complete"] = sets_complete
    tmp = os.path.join(sig_dir, f"index.json.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(idx, f, indent=1)
    os.replace(tmp, os.path.join(sig_dir, "index.json"))


def finish_index(sig_dir, gen, version, names, stride, n_keyframes, options, written, dataroot, set_name=None, old_complete=None):
    """the index.json at the end of a run (generate, report): written when this run wrote keyframes, when the index has no manifest yet (no commit: a `build` store, an older one) or when a clean tree can stamp
    the index of an earlier, dirty run (same signature, same GT); a run that wrote nothing leaves a finished index as it was, whatever stride or sequences it was asked for (it must not turn stride 1 into
    "mixed", drop the release or re-stamp the commit). -> (the index, whether it was written)"""
    from . import versions
    old = read_index(sig_dir)
    can_upgrade = versions.code_state()["dirty"] is False and not old.get("release")
    if not written and old.get("complete") and set(names) <= set(old.get("scenes", [])) and "commit" in old and not can_upgrade:
        return old, False
    write_index(sig_dir, gen, version, names, stride, n_keyframes, options, run=run_manifest(sig_dir, version, dataroot, set_name, options), old_complete=old_complete)
    return read_index(sig_dir), True


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


def detect_version(dataroot):
    """the nuScenes version inside `dataroot`: the one `v1.0-*` folder that holds a scene.json"""
    root = os.path.expanduser(dataroot)
    found = sorted(d for d in os.listdir(root) if d.startswith("v1.0-") and os.path.exists(os.path.join(root, d, "scene.json"))) \
        if os.path.isdir(root) else []
    if len(found) != 1:
        raise ValueError(f"{root}: " + (f"several nuScenes versions {found}, pass version=" if found else "no v1.0-*/scene.json: not a nuScenes folder"))
    return found[0]
