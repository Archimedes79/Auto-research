"""The lane GT of autogt: the lane graph, the road structure and the lines of a keyframe window, with its OWN version and store.

It is not part of the front generator (`nuscenes_gt.py`): the front GT, its signature and its store stay as they are, whatever is changed here. The lane GT
is made from the nuScenes map and the ego pose at the front-camera time only (no image, radar or box), about 40 ms per keyframe.

    from autogt import lane_gt
    g = lane_gt.load(sample_token)                                    # the stored graph (data/autogt/lane/<lane signature>/<token>.pkl.gz)
    g = lane_gt.LaneGT("~/data/nuscenes_full", "v1.0-trainval", scenes=[name]).graph(sample_token)       # made on demand
    lane_gt.generate("~/data/nuscenes_full", set_name="full")         # the complete set: resumable, groups of sequences, workers=N

What a graph holds: lane_graph.py (pieces, events), road_graph.py (neighbours, lane counts, roads), lane_lines.py (dividers, boundary, corners, the road and intersection
areas as polygons, not dilated); the fields are
listed in AGENTS.md, section 3b. Frame "lane" (x right, y forward, ego rear axle at the camera time; `lane_graph.in_frame(g, "ego")` for x forward, y left).

VERSION. The signature of the lane GT is the hash of the source of lane_graph.py, road_graph.py, lane_lines.py and this file plus the parameters that differ from
DEFAULTS; the table `lane_versions.csv` (next to this file) names each signature (version l1, l2, ..., a git tag autogt/lane-lN). A parameter that is open with Mirko
(boundary layers, corner angle, stop rule) is a parameter: another value is another signature, stored next to the first. Generate / load use the signature of THIS
source, like the front GT does.
"""
import csv
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import time

from . import lane_graph as LG
from . import lane_lines as LL
from . import lane_tiers as LT
from . import road_graph as RG
from .nuscenes_adapter import pose_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "autogt", "lane")          # <repository root>/data/autogt/lane
SOURCE_FILES = ("lane_graph.py", "road_graph.py", "lane_lines.py", "lane_gt.py", "lane_tiers.py", "rig_masks.py")          # l4: the tiers and the camera rig they use
DEFAULTS = dict(half=50.0, min_len=2.0, step=0.5, inter_frac=0.5, stop_rule="intersection_branch", boundary_layers=("drivable_area",),
                corner_deg=30.0, corner_base=2.0,
                tier_back=25.0, tier_ahead=55.0, path_tol=1.5, path_cos=0.7, neighbour_tol=10.0, min_ahead=40.0, min_back=20.0, extend=True, opposite_tier=3)      # l4: lane_tiers.py
GROUP_SCENES = 60
_SIGNATURES = {}


def _params(params):
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown lane GT parameter(s) {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    out = {**DEFAULTS, **params}
    out["boundary_layers"] = (out["boundary_layers"],) if isinstance(out["boundary_layers"], str) else tuple(out["boundary_layers"])
    for k in ("half", "min_len", "step", "inter_frac", "corner_deg", "corner_base", "tier_back", "tier_ahead", "path_tol", "path_cos", "neighbour_tol", "min_ahead", "min_back"):
        out[k] = float(out[k])                                       # 60 and 60.0 are one parameter, one signature
    out["extend"], out["opposite_tier"] = bool(out["extend"]), int(out["opposite_tier"])
    if out["opposite_tier"] not in (2, 3):
        raise ValueError("opposite_tier is 2 or 3")
    if out["step"] != 0.5:
        raise ValueError("the lane GT is made for a map sampling of 0.5 m (step); other values are not supported")
    return out


def signature(**params):
    """10 hex chars: the source of SOURCE_FILES plus the parameters that differ from DEFAULTS. Computed once per process and parameter set, so a process that runs
    while a file is edited keeps the key of the code it started with."""
    p = _params(params)
    changed = sorted((k, v) for k, v in p.items() if v != DEFAULTS[k])
    key = repr(changed)
    if key not in _SIGNATURES:
        h = hashlib.sha1()
        for name in SOURCE_FILES:
            with open(os.path.join(HERE, name), "rb") as f:
                h.update(f.read())
        h.update(key.encode())
        _SIGNATURES[key] = h.hexdigest()[:10]
    return _SIGNATURES[key]


def table():
    """the rows of lane_versions.csv"""
    with open(os.path.join(HERE, "lane_versions.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def version_of(sig):
    """"l1" or None (a signature without a row: unreleased)"""
    return next((r["version"] for r in table() if r["signature"] == sig), None)


class LaneGT:
    """LaneGT(dataroot, version, nusc=None, scenes=None, **params): the lane graph of keyframes. `nusc`: an existing devkit object (anything with `get`),
    otherwise the memory-light devkit of `scenes` (CAM_FRONT only). The maps are turned into graphs once per location."""

    def __init__(self, dataroot, version, nusc=None, scenes=None, **params):
        self.dataroot = os.path.expanduser(dataroot)
        self.params = _params(params)
        if nusc is None:
            from .light_nusc import light_nuscenes
            nusc = light_nuscenes(self.dataroot, version, scenes, radars=False)
        self.nusc = nusc
        self._maps = {}

    def signature(self):
        return signature(**self.params)

    def _map(self, location):
        if location not in self._maps:
            from nuscenes.map_expansion.map_api import NuScenesMap
            p = self.params
            nmap = NuScenesMap(dataroot=self.dataroot, map_name=location)
            mg = LG.build_map_graph(nmap, step=p["step"], inter_frac=p["inter_frac"], stop_rule=p["stop_rule"])
            rg = RG.build_road_graph(mg)
            rg["area_chunks"] = LL.polygon_chunks(rg["area_geoms"])
            self._maps[location] = (mg, rg, LL.build_line_graph(nmap, boundary_layers=p["boundary_layers"]))
        return self._maps[location]

    def pose(self, sample_token):
        """T_e2g (4 x 4): the ego pose at the front-camera time of the keyframe"""
        sample = self.nusc.get("sample", sample_token)
        sd = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        return pose_matrix(self.nusc.get("ego_pose", sd["ego_pose_token"]))

    def graph(self, sample_token, frame="lane"):
        """the lane graph of a keyframe (see the module docstring)"""
        p = self.params
        sample = self.nusc.get("sample", sample_token)
        loc = self.nusc.get("log", self.nusc.get("scene", sample["scene_token"])["log_token"])["location"]
        mg, rg, lg = self._map(loc)
        T = self.pose(sample_token)
        g = LG.crop(mg, T, half=p["half"], min_len=p["min_len"], frame=frame, rg=rg)
        g["lines"] = LL.crop_lines(lg, T, half=p["half"], frame=frame, corner_deg=p["corner_deg"], corner_base=p["corner_base"])
        g["areas"] = {"road": LL.crop_areas(lg["road"], T, half=p["half"], frame=frame, merge=True), "intersection": LL.crop_areas(rg["area_chunks"], T, half=p["half"], frame=frame)}
        g["params"].update(step=p["step"], boundary_layers=list(p["boundary_layers"]), corner_deg=p["corner_deg"], corner_base=p["corner_base"])
        g["location"], g["sample"], g["signature"] = loc, sample_token, self.signature()
        g["params"].update({k: p[k] for k in DEFAULTS if k.startswith("tier_") or k in ("path_tol", "path_cos", "neighbour_tol", "min_ahead", "min_back", "extend", "opposite_tier")})
        LT.add_tiers(g, self.nusc, sample_token, T, p)                                                    # l4: tier / vis per point, the driven path (lane_tiers.py)
        return g


# ---------------------------------------------------------------------------------------------------------------- the store
def _file(sig_dir, token):
    return os.path.join(sig_dir, token + ".pkl.gz")


def _save(sig_dir, token, g):
    tmp = _file(sig_dir, token) + f".{os.getpid()}.tmp"
    with gzip.open(tmp, "wb", compresslevel=3) as f:
        pickle.dump(g, f, protocol=4)
    os.replace(tmp, _file(sig_dir, token))


def held(sig_dir):
    """the sample tokens with a stored graph in the folder (a temporary or an empty file does not count)"""
    if not os.path.isdir(sig_dir):
        return set()
    with os.scandir(sig_dir) as it:
        return {e.name[:-7] for e in it if e.name.endswith(".pkl.gz") and ".tmp" not in e.name and e.stat().st_size > 0}


def load(token, data=DATA, **params):
    """the stored lane graph of a keyframe for the lane signature of THIS source and `params`. Raises FileNotFoundError that names the folders that exist."""
    sig_dir = os.path.join(data, signature(**params))
    path = _file(sig_dir, token)
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        have = sorted(d for d in os.listdir(data) if os.path.isdir(os.path.join(data, d))) if os.path.isdir(data) else []
        raise FileNotFoundError(f"no stored lane graph for keyframe {token} in {sig_dir} (folders in {data}: {have}); generate it with "
                                "`python -m autogt generate-lane <nuScenes folder> --set full`, or the lane GT source changed since it was stored")
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def missing(tokens, data=DATA, **params):
    """the tokens without a stored graph for this source ([] = the store covers them)"""
    have = held(os.path.join(data, signature(**params)))
    return [t for t in tokens if t not in have]


def _environment():
    from importlib import metadata
    out = {}
    for pkg in ("numpy", "shapely", "opencv-python", "nuscenes-devkit", "pyquaternion"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "?"
    return out


def write_index(sig_dir, version, names, params, sets_complete, complete=True):
    """index.json of a lane store: what it holds and whether it is a release (complete, a released version, a clean tree, the set `full`)"""
    from . import versions
    held_now = held(sig_dir)
    state = versions.code_state()
    sig = os.path.basename(sig_dir)
    lane_version = version_of(sig)
    idx = dict(signature=sig, lane_version=lane_version, version=version, scenes=list(names), keyframes=len(held_now), complete=bool(complete),
               params={k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}, sets_complete=sorted(sets_complete),
               commit=state["commit"], dirty=state["dirty"], environment=_environment(),
               release=bool(complete and lane_version and state["dirty"] is False and "full" in sets_complete))
    tmp = os.path.join(sig_dir, f"index.json.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(idx, f, indent=1)
    os.replace(tmp, os.path.join(sig_dir, "index.json"))
    return idx


def _group_task(args):
    code_sig, dataroot, version, params, grp, todo, sig_dir = args
    t0 = time.time()
    lg = LaneGT(dataroot, version, scenes=grp, **params)
    if lg.signature() != code_sig:                         # a worker imports the source afresh: it must be the source the job started with
        raise RuntimeError(f"a lane GT source file changed while this job runs (signature {code_sig} at the start, {lg.signature()} now): stop and start again")
    for tok in todo:
        _save(sig_dir, tok, lg.graph(tok))
    return len(todo), time.time() - t0


def generate(dataroot, out=None, version=None, scenes=None, set_name=None, workers=1, verbose=True, **params):
    """make the lane graph of every keyframe of `scenes` (names; default: all sequences, or the `set_name` set "small" / "full") and store it under
    OUT/<lane signature>/ (default data/autogt/lane); files that exist are skipped (resumable); `workers` processes, one group of sequences each.
    -> {"out", "signature", "lane_version", "scenes", "keyframes", "written", "release"}"""
    from . import sets
    from .cli import detect_version
    from .suite import json_tables, scene_tokens
    if set_name and scenes is not None:
        raise ValueError(f"pass either scenes= or set_name= ({set_name!r}), not both")
    dataroot = os.path.expanduser(dataroot)
    version = version or detect_version(dataroot)
    out = os.path.expanduser(out) if out else DATA
    known = [s["name"] for s in json_tables(dataroot, version)[0]]
    names = sets.scenes(set_name) if set_name else known if scenes is None else list(dict.fromkeys(scenes))
    if not names or [n for n in names if n not in known]:
        raise ValueError(f"unknown scene(s) {[n for n in names if n not in known] or 'none given'} in {version} of {dataroot}")
    p = _params(params)
    code_sig = signature(**p)
    sig_dir = os.path.join(out, code_sig)
    os.makedirs(sig_dir, exist_ok=True)
    tokens = {n: scene_tokens(n, version, dataroot) for n in names}
    n_total = sum(len(t) for t in tokens.values())
    stored = held(sig_dir)
    size = min(GROUP_SCENES, max(1, -(-len(names) // workers)))
    groups = []
    for g0 in range(0, len(names), size):
        grp = names[g0:g0 + size]
        todo = [t for n in grp for t in tokens[n] if t not in stored]
        if todo:
            groups.append((grp, todo))
    if groups:
        write_index(sig_dir, version, names, p, [], complete=False)
    done = 0
    t0 = time.time()
    if workers <= 1 or len(groups) <= 1:
        for i, (grp, todo) in enumerate(groups):
            n, secs = _group_task((code_sig, dataroot, version, p, grp, todo, sig_dir))
            done += n
            if verbose:
                print(f"group {i + 1}/{len(groups)}: {n} keyframes in {secs:.0f} s ({time.time() - t0:.0f} s in all)", flush=True)
    else:
        jobs = [(code_sig, dataroot, version, p, grp, todo, sig_dir) for grp, todo in groups]
        with mp.get_context("spawn").Pool(min(workers, len(jobs)), maxtasksperchild=1) as pool:
            for n, secs in pool.imap_unordered(_group_task, jobs):
                done += n
                if verbose:
                    print(f"a group done: {n} keyframes in {secs:.0f} s ({time.time() - t0:.0f} s in all)", flush=True)
    complete_sets = []
    now = held(sig_dir)
    for name in sets.names():
        try:
            want = sets.tokens(name, version, dataroot)
        except ValueError:
            continue
        if all(t in now for t in want):
            complete_sets.append(name)
    idx = write_index(sig_dir, version, names, p, complete_sets, complete=all(t in now for n in names for t in tokens[n]))
    if verbose:
        print(f"{done} keyframes written, {n_total - done} already there: {sig_dir}  (lane version {idx['lane_version']}, commit {idx['commit']}"
              f"{', dirty' if idx['dirty'] else ''}{', RELEASE' if idx['release'] else ''}, sets complete: {idx['sets_complete']})")
    return {"out": sig_dir, "signature": code_sig, "lane_version": idx["lane_version"], "scenes": len(names), "keyframes": n_total, "written": done,
            "release": idx["release"]}
