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
    sets.stored_for(scenes, ...)  the same for any list of sequence names, with the missing sample tokens: `python -m autogt have`
    sets.check_raw(dataroot, ...) are the raw files the generator reads on disk (tables, map expansion, CAM_FRONT, the five radars)?
                                  `python -m autogt data`; sets.report_raw(result) prints it. The tools that fetch, verify and link the raw
                                  data are `datasets/NuScenes/tools/` (its README describes the layout on D:)
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


def _find(want, version=None, dataroot=None):
    """(version, dataroot) whose json tables hold every scene name of `want`: the asked version, else trainval before mini (the first
    that is on this machine and holds all of them; `dataroot` default ~/data/nuscenes_full / ~/data/nuscenes)"""
    from .suite import default_root, json_tables
    for v in ([version] if version else ["v1.0-trainval", "v1.0-mini"]):
        root = os.path.expanduser(dataroot) if dataroot else default_root(v)
        if os.path.exists(os.path.join(root, v, "scene.json")) and set(want) <= {s["name"] for s in json_tables(root, v)[0]}:
            return v, root
    raise ValueError(f"no nuScenes version {version or 'v1.0-trainval / v1.0-mini'} under {dataroot or 'the default folders'} "
                     f"holds all {len(want)} scenes asked for")


def _tables(name, version=None, dataroot=None):
    return _find(scenes(name), version, dataroot)


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


def stored_for(want, data=None, version=None, dataroot=None, stride=1, **options):
    """the keyframes (every `stride`-th) of the sequences `want` (names) that the store of the current generator holds: the check before a
    training or an evaluation -> {"version", "scenes", "total", "have", "missing": [sample tokens]}; `python -m autogt have`"""
    from .suite import DATA, current_signature, scene_tokens
    version, root = _find(list(want), version, dataroot)
    toks = [t for s in want for t in scene_tokens(s, version, root)[::stride]]
    stored = held(os.path.join(data or DATA, current_signature(**options)))
    missing = [t for t in toks if t not in stored]
    return {"version": version, "scenes": len(list(want)), "total": len(toks), "have": len(toks) - len(missing), "missing": missing}


def checkable(version, dataroot):
    """the sets (by name) whose sequences all are in this nuScenes version (small is in mini and in trainval, full only in trainval)"""
    out = []
    for name in names():
        try:
            tokens(name, version, dataroot)
        except ValueError:
            continue
        out.append(name)
    return out


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


# ---- the raw files a run needs
# autogt reads the json tables, the map expansion of the cities of the sequences, and per keyframe and radar sweep the files of CAM_FRONT
# and the five radars (the image is needed to look at the GT, the radar files for the velocity correction). One pass over the tables (no
# devkit) and one listing per sensor folder; a file counts as present by its name, its content is not read.


def check_raw(dataroot, version=None, set_name=None, scenes=None, examples=10):
    from .cli import detect_version
    from .light_nusc import _KEEP, _records, _value
    import json
    root = os.path.expanduser(dataroot)
    version = version or detect_version(root)
    tab = os.path.join(root, version)
    with open(os.path.join(tab, "scene.json")) as f:
        by_name = {s["name"]: s for s in json.load(f)}
    names = scenes(set_name) if set_name else list(by_name) if scenes is None else list(scenes)
    lost = [n for n in names if n not in by_name]
    if lost or not names:
        raise ValueError(f"unknown scene(s) {lost or 'none given'} in {version} of {root}")
    want = {by_name[n]["token"] for n in names}
    with open(os.path.join(tab, "sample.json")) as f:
        samples = {s["token"] for s in json.load(f) if s["scene_token"] in want}
    with open(os.path.join(tab, "log.json")) as f:
        logs = {r["token"]: r["location"] for r in json.load(f)}
    cities = sorted({logs[by_name[n]["log_token"]] for n in names})
    maps = [f"maps/expansion/{c}.json" for c in cities if not os.path.exists(os.path.join(root, "maps", "expansion", f"{c}.json"))]
    n_cam = n_radar = n_missing = 0
    shown = []
    listed = {}                                                    # folder -> the names in it: one listing per folder, not one call per file

    def present(rel):
        d, name = os.path.split(rel)
        if d not in listed:
            try:
                listed[d] = set(os.listdir(os.path.join(root, d)))
            except OSError:
                listed[d] = set()
        return name in listed[d]
    for text in _records(os.path.join(tab, "sample_data.json")):
        if not any(k in text for k in _KEEP) or _value(text, "sample_token") not in samples:
            continue
        rel = json.loads(text)["filename"]
        n_cam, n_radar = n_cam + ("/CAM_FRONT/" in rel), n_radar + ("/RADAR_" in rel)
        if not present(rel):
            n_missing += 1
            if len(shown) < examples:
                shown.append(rel)
    return {"version": version, "scenes": len(names), "camera": n_cam, "radar": n_radar, "missing": n_missing, "examples": shown, "maps": maps}


def report_raw(r):
    """a few lines for a result of `check`"""
    ok = not r["missing"] and not r["maps"]
    lines = [f"{r['version']}: {r['scenes']} sequences, {r['camera']} CAM_FRONT and {r['radar']} radar files listed: "
             f"{'all on disk' if ok else str(r['missing']) + ' missing'}"]
    lines += [f"  missing: {p}" for p in r["examples"]] + [f"  missing map file: {p}" for p in r["maps"]]
    return "\n".join(lines)
