"""Is the raw nuScenes data on disk complete for the GT generator? `python -m autogt data DATAROOT [--set small|full | --scenes ...]`

autogt reads the json tables, the map expansion of the cities of the sequences, and per keyframe and radar sweep the files of CAM_FRONT and
the five radars (the front camera image is needed to look at the GT, the radar files for the velocity correction). This check lists what
of that is missing, in one pass over the tables (no devkit) and one listing per sensor folder (a file counts as present by its name; its
content is not read). The tools that fetch, verify and link the raw data
are `datasets/NuScenes/tools/` (its README describes the layout on D:); this is the check that the data they leave is enough for the GT.

    check(dataroot, version=None, set_name=None, scenes=None) -> {"version", "scenes", "camera", "radar", "missing": n, "examples": [...],
                                                                  "maps": [missing expansion files]}
"""
import json
import os

from .light_nusc import _KEEP, _records, _value


def check(dataroot, version=None, set_name=None, scenes=None, examples=10):
    from . import sets
    from .cli import detect_version
    root = os.path.expanduser(dataroot)
    version = version or detect_version(root)
    tab = os.path.join(root, version)
    with open(os.path.join(tab, "scene.json")) as f:
        by_name = {s["name"]: s for s in json.load(f)}
    names = sets.scenes(set_name) if set_name else list(by_name) if scenes is None else list(scenes)
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
    for buf, text in _records(os.path.join(tab, "sample_data.json")):
        if not any(k in text for k in _KEEP) or _value(buf, "sample_token") not in samples:
            continue
        rel = json.loads(text)["filename"]
        n_cam, n_radar = n_cam + ("/CAM_FRONT/" in rel), n_radar + ("/RADAR_" in rel)
        if not present(rel):
            n_missing += 1
            if len(shown) < examples:
                shown.append(rel)
    return {"version": version, "scenes": len(names), "camera": n_cam, "radar": n_radar, "missing": n_missing, "examples": shown, "maps": maps}


def report(r):
    """a few lines for a result of `check`"""
    ok = not r["missing"] and not r["maps"]
    lines = [f"{r['version']}: {r['scenes']} sequences, {r['camera']} CAM_FRONT and {r['radar']} radar files listed: "
             f"{'all on disk' if ok else str(r['missing']) + ' missing'}"]
    lines += [f"  missing: {p}" for p in r["examples"]] + [f"  missing map file: {p}" for p in r["maps"]]
    return "\n".join(lines)
