"""Named scene sets: the small run and the full run (autogt/sets/<name>.txt).

    small  the 10 sequences of nuScenes mini: small experiments (404 keyframes, about 25 minutes to generate)
    full   all 850 sequences of nuScenes trainval: releases (34,149 keyframes, 5 - 8 hours to generate)

A set file holds one scene name per line, `#` starts a comment. `autogt.generate(dataroot, set_name="small")` and
`python -m autogt generate DATAROOT --set small` generate exactly those sequences; `python -m autogt sets` and `sets.status(name)` say how
much of a set the store of the current generator holds. The store folder is shared by the sets (small is a part of full), so a set is
"complete" when every keyframe of its scenes has a file; the index.json of a store lists the complete sets (key `sets_complete`).

    sets.names()                  ["full", "small"]
    sets.scenes(name)             the scene names of a set
    sets.tokens(name, ...)        the sample tokens of those scenes (json tables only, no devkit)
    sets.status(name, ...)        {"set", "version", "have", "total", "complete"} for the current generator signature
"""
import os

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sets")


def names():
    return sorted(f[:-4] for f in os.listdir(DIR) if f.endswith(".txt"))


def scenes(name):
    """the scene names of a set"""
    path = os.path.join(DIR, name + ".txt")
    if not os.path.exists(path):
        raise ValueError(f"no scene set {name!r}: the sets are {names()} (files in {DIR})")
    with open(path, encoding="utf-8") as f:
        lines = [ln.split("#")[0].strip() for ln in f]
    return [x for x in lines if x]


def _tables(name, version=None, dataroot=None):
    """(version, dataroot) whose json tables hold every scene of the set: the asked version, else trainval before mini (the first
    that is on this machine and holds all of them; `dataroot` default ~/data/nuscenes_full / ~/data/nuscenes)"""
    from .suite import default_root, json_tables
    want = scenes(name)
    for v in ([version] if version else ["v1.0-trainval", "v1.0-mini"]):
        root = os.path.expanduser(dataroot) if dataroot else default_root(v)
        if os.path.exists(os.path.join(root, v, "scene.json")) and set(want) <= {s["name"] for s in json_tables(root, v)[0]}:
            return v, root
    raise ValueError(f"set {name!r}: no nuScenes version {version or 'v1.0-trainval / v1.0-mini'} under {dataroot or 'the default folders'} "
                     f"holds all its {len(want)} scenes")


def tokens(name, version=None, dataroot=None):
    """the sample tokens of the sequences of a set in time order, from the nuScenes json tables"""
    from .suite import scene_tokens
    version, root = _tables(name, version, dataroot)
    return [t for s in scenes(name) for t in scene_tokens(s, version, root)]


def held(sig_dir):
    """the sample tokens that have a stored keyframe in the folder (a leftover temporary file or an empty file does not count)"""
    if not os.path.isdir(sig_dir):
        return set()
    with os.scandir(sig_dir) as it:
        return {e.name[:-4] for e in it if e.name.endswith(".npz") and ".tmp" not in e.name and e.stat().st_size > 0}


def status(name, data=None, version=None, dataroot=None, **options):
    """how much of a set the store of the current generator (and options) holds"""
    from .suite import DATA, current_signature
    version, root = _tables(name, version, dataroot)
    toks = tokens(name, version, root)
    stored = held(os.path.join(data or DATA, current_signature(**options)))
    have = sum(1 for t in toks if t in stored)
    return {"set": name, "version": version, "have": have, "total": len(toks), "complete": bool(toks) and have == len(toks)}


def complete_in(sig_dir, version, dataroot):
    """the sets (by name) whose every keyframe has a file in the store folder sig_dir; a set with scenes that `version` does not have
    is skipped (small can be complete in a trainval store, full cannot be in a mini store)"""
    have = held(sig_dir)
    out = []
    for name in names():
        try:
            toks = tokens(name, version, dataroot)
        except ValueError:
            continue
        if toks and have >= set(toks):
            out.append(name)
    return out
