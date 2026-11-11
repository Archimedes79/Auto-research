"""The lane GT of GtGenerator: the lane graph, the road structure and the lines of a keyframe window, with its OWN version and store.

It is not part of the front generator (`nuscenes_gt.py`): the front GT, its signature and its store stay as they are, whatever is changed here. The lane graph is made from the
nuScenes map and the ego pose at the front-camera time only (no image, radar); the l4 labels (tiers, visibility: lane_tiers.py) also read the CAM_FRONT ego poses around the keyframe
(non-causal: labels only, never an input) and the annotated boxes (scipy and pyquaternion are needed for them); l6: about 43 ms (lane_center) / 34 ms (bev_fusion) per keyframe for graph and labels on an idle core.

    from GtGenerator import lane_gt
    g = lane_gt.load(sample_token)                                    # the stored graph (data/autogt/lane/<lane signature>/<token>.pkl.gz)
    g = lane_gt.LaneGT("~/data/nuscenes_full", "v1.0-trainval", scenes=[name]).graph(sample_token)       # made on demand
    lane_gt.generate("~/data/nuscenes_full", set_name="full")         # the complete set: resumable, groups of sequences, workers=N

What a graph holds: lane_graph.py (pieces, events), road_graph.py (neighbours, lane counts, roads), lane_lines.py (dividers, boundary, corners, the road and intersection
areas as polygons, not dilated); the fields are
listed in AGENTS.md, section 3b. Frame "lane" (x right, y forward, ego rear axle at the camera time; `lane_graph.in_frame(g, "ego")` for x forward, y left).

RIGS (l5). The suite makes the lane GT for the two systems of the repository, each with its own frame, window and cameras (`RIGS`, parameter `rig`):
    lane_center   Spurmitte: frame "lane", the square |x|, |y| <= 50 m, `vis` / `occ` from the six cameras                      lane_gt.load(token)
    bev_fusion    BEV Fusion: frame "ego" (x forward, y left), the window of its BEV grid x 0 .. 60 m, y -26 .. 26 m, `vis` / `occ` from CAM_FRONT   lane_gt.load(token, rig="bev_fusion")
The geometry, the ids, the events and the tiers come from the same code and the same rules; the rig changes only the frame, the cut and the cameras of the visibility (l6: the tiers are decided on a wide
window the same for both rigs, so a point that two rigs share has the same tier; in l5 a neighbour whose tier-1 lane lay outside the smaller window could be tier 3 instead of tier 2). Each rig has its own store.

VERSION. The signature of the lane GT is the hash of the source of lane_graph.py, road_graph.py, lane_lines.py, lane_tiers.py, rig_cameras.py and this file plus the parameters
that differ from DEFAULTS (the rig is one of them); the table `lane_versions.csv` (next to this file) names each signature (version l1, l2, ..., a git tag autogt/lane-lN; one row
per rig). A parameter that is open with Mirko (boundary layers, corner angle, stop rule) is a parameter: another value is another signature, stored next to the first.
Generate / load use the signature of THIS source, like the front GT does. `code_signature` is the hash; `signature` is the folder of the store: the same, except that a row
with `same_gt_as` (a version whose code makes the GT of an earlier one, like v9 -> v8 of the front GT) points to the earlier version's folder (l5 lane_center -> l4).
What this file imports from the package must not shape a label: the table readers, the devkit loader, the listing of a store (`store`, `sets`, `suite`, `light_nusc`, `versions`); a test pins it.
"""
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import time
import warnings

import numpy as np

from . import lane_graph as LG
from . import lane_lines as LL
from . import lane_tiers as LT
from . import rig_cameras as RC
from . import road_graph as RG
from . import store

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "autogt", "lane")          # <repository root>/data/autogt/lane
SOURCE_FILES = ("lane_graph.py", "road_graph.py", "lane_lines.py", "lane_gt.py", "lane_tiers.py", "rig_cameras.py")          # l4: the tiers and the camera rig they use
DEFAULTS = dict(half=50.0, min_len=2.0, step=0.5, inter_frac=0.5, stop_rule="intersection_branch", boundary_layers=("drivable_area",),
                corner_deg=30.0, corner_base=2.0,
                tier_back=25.0, tier_ahead=55.0, path_tol=1.5, path_cos=0.7, neighbour_tol=10.0, min_ahead=40.0, min_back=20.0, extend=True, opposite_tier=3,      # l4: lane_tiers.py
                rig="lane_center")                                                                                                                              # l5: which system the GT is made for
# The two systems the suite makes lane GT for (decision of Mirko 2026-11-07: BEV Fusion and Spurmitte are different systems with different frames, windows and cameras):
#   lane_center  Spurmitte (builds on GaussianCaR): frame "lane" (x right, y forward), the square window |x|, |y| <= half (50 m), `vis` / `occ` from the six cameras
#   bev_fusion   BEV Fusion (front camera + radars): frame "ego" (x forward, y left), the window of its BEV grid x 0 .. 60 m, y -26 .. 26 m (grid.GridSpec), `vis` / `occ` from CAM_FRONT
# `window` None = the square of `half`. The tiers (which lane is the ego lane, its neighbours) do not depend on the rig; only the frame, the window and the cameras do.
RIGS = {"lane_center": dict(frame="lane", window=None, cameras=RC.SIX),
        "bev_fusion": dict(frame="ego", window=(0.0, 60.0, -26.0, 26.0), cameras=("CAM_FRONT",))}
# l6: the tiers (lane_tiers.py) are decided on the keyframe cut to the square |x|, |y| <= max(TIER_HALF, the bound of the rig's window + TIER_MARGIN), whatever the window of the rig is. The driven
# path reaches 80 m, the extension marks the rest of its piece (only about the first 77 m of it can change a label inside the grid) and a neighbour counts within 10 m of tier 1; 100 m (the same for
# both rigs) gave the tiers of 250 m on every point of the 970 frames of the l6 check (a road that leaves the 100 m square and comes back inside the extension is excluded by that check, not by argument)
TIER_HALF, TIER_MARGIN = 100.0, 40.0
GROUP_SCENES = 60
_SIGNATURES = {}
_STORE_KEYS = {}


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
    if out["rig"] not in RIGS:
        raise ValueError(f"unknown rig {out['rig']!r}; known: {sorted(RIGS)}")
    if RIGS[out["rig"]]["window"] is not None and out["half"] != DEFAULTS["half"]:
        raise ValueError(f"rig {out['rig']!r} has a rectangular window: `half` does not apply")
    if out["step"] != 0.5:
        raise ValueError("the lane GT is made for a map sampling of 0.5 m (step); other values are not supported")
    return out


def code_signature(**params):
    """10 hex chars: the source of SOURCE_FILES plus the parameters that differ from DEFAULTS. Computed once per process and parameter set, so a process that runs
    while a file is edited keeps the key of the code it started with. This names the CODE (a row of lane_versions.csv); the folder of a store is `signature`."""
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
    """the rows of lane_versions.csv: version, signature, tag, date, what_changed, rig (empty = lane_center), same_gt_as (empty, or the version whose store this
    version makes: its GT for this rig is the same, checked on stored frames)"""
    with open(os.path.join(HERE, "lane_versions.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["rig"] = r.get("rig") or "lane_center"
        r["same_gt_as"] = r.get("same_gt_as") or ""
    return rows


def signature(**params):
    """the KEY OF THE STORE for this source and `params`: data/autogt/lane/<key>/. It is the code signature, except when the table says that this code makes the same GT as an
    earlier version (column same_gt_as, like v9 -> v8 of the front GT): then it is that version's signature, so that a change of the code that does not change the GT does not
    start a new store, and the jobs that read the earlier store keep working. Only the parameters of a table row are aliased (the default parameters of a rig)."""
    return store_of(code_signature(**params))


def store_of(sig):
    """the store key of a code signature (see `signature`): `sig` itself, or the signature of the earlier version whose GT the table says this code makes"""
    if sig not in _STORE_KEYS:
        key, rows, seen = sig, table(), set()
        while True:
            row = next((r for r in rows if r["signature"] == key and r["same_gt_as"]), None)
            if row is None:
                break
            if key in seen:
                raise ValueError(f"lane_versions.csv: the rows with same_gt_as form a cycle at {row['version']} ({row['rig']})")
            seen.add(key)
            earlier = next((r for r in rows if r["version"] == row["same_gt_as"] and r["rig"] == row["rig"]), None)
            if earlier is None:
                raise ValueError(f"lane_versions.csv: {row['version']} ({row['rig']}) is the same GT as {row['same_gt_as']}, which has no row for this rig")
            key = earlier["signature"]
        _STORE_KEYS[sig] = key
    return _STORE_KEYS[sig]


def check_table(rows=None):
    """the rows of lane_versions.csv are consistent: every rig is known, every same_gt_as names an earlier row of the same rig, no row names itself or an unknown version, a signature
    belongs to one version and rig; raises ValueError (the self-test calls it)"""
    rows = table() if rows is None else rows
    versions = [r["version"] for r in rows]
    for i, r in enumerate(rows):
        if r["rig"] not in RIGS:
            raise ValueError(f"lane_versions.csv: {r['version']} has the unknown rig {r['rig']!r}")
        if r["same_gt_as"]:
            earlier = [x for x in rows[:i] if x["version"] == r["same_gt_as"] and x["rig"] == r["rig"]]
            if not earlier or r["same_gt_as"] == r["version"]:
                raise ValueError(f"lane_versions.csv: {r['version']} ({r['rig']}) says same_gt_as {r['same_gt_as']}, which is no EARLIER row of this rig")
        if sum(1 for x in rows if x["signature"] == r["signature"]) > 1:
            raise ValueError(f"lane_versions.csv: the signature {r['signature']} is in more than one row")
    return versions


def version_of(sig):
    """"l1" or None (a signature without a row: unreleased); for a store key the version that made the store, for a code signature its own version"""
    return next((r["version"] for r in table() if r["signature"] == sig), None)


def pose_matrix(rec):
    """T_e2g (4 x 4) of an ego_pose record (a copy of nuscenes_adapter.pose_matrix: that file belongs to the signature of the front GT, this one to the lane GT's)"""
    from pyquaternion import Quaternion
    T = np.eye(4)
    T[:3, :3] = Quaternion(rec["rotation"]).rotation_matrix
    T[:3, 3] = rec["translation"]
    return T


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

    def code_signature(self):
        return code_signature(**self.params)

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

    def graph(self, sample_token, frame=None):
        """the lane graph of a keyframe (see the module docstring); `frame` defaults to the frame of the rig (lane_center "lane"; bev_fusion "ego", the only frame its window is defined in)"""
        p = self.params
        rig = RIGS[p["rig"]]
        window = rig["window"]
        frame = frame or rig["frame"]
        if window is not None and frame != rig["frame"]:
            raise ValueError(f"rig {p['rig']!r} is defined in the frame {rig['frame']!r}; take the graph in it and turn it with lane_graph.in_frame if needed")
        half = p["half"] if window is None else max(abs(v) for v in window)               # the bound that is stored; the window itself is stored as g["window"]
        sample = self.nusc.get("sample", sample_token)
        loc = self.nusc.get("log", self.nusc.get("scene", sample["scene_token"])["log_token"])["location"]
        mg, rg, lg = self._map(loc)
        T = self.pose(sample_token)
        g = LG.crop(mg, T, half=half, min_len=p["min_len"], frame=frame, rg=rg, window=window)
        g["lines"] = LL.crop_lines(lg, T, half=half, frame=frame, corner_deg=p["corner_deg"], corner_base=p["corner_base"], window=window)
        g["areas"] = {"road": LL.crop_areas(lg["road"], T, half=half, frame=frame, merge=True, window=window),
                      "intersection": LL.crop_areas(rg["area_chunks"], T, half=half, frame=frame, window=window)}
        g["params"].update(step=p["step"], boundary_layers=list(p["boundary_layers"]), corner_deg=p["corner_deg"], corner_base=p["corner_base"])
        g["location"], g["sample"], g["signature"] = loc, sample_token, self.signature()
        g["params"].update({k: p[k] for k in DEFAULTS if k.startswith("tier_") or k in ("path_tol", "path_cos", "neighbour_tol", "min_ahead", "min_back", "extend", "opposite_tier")})
        wide = LG.crop(mg, T, half=max(TIER_HALF, half + TIER_MARGIN), min_len=p["min_len"], frame=frame, rg=rg, full=False)         # l6: the tiers are decided on this window, the same for every rig, and carried over
        LT.add_tiers(g, self.nusc, sample_token, T, p, rig["cameras"], wide=wide)                                       # l4: tier / vis per point, the driven path (lane_tiers.py); l5: the cameras of the rig
        for q in g["pieces"]:
            del q["src"]                                                                                  # the vertex of the map piece each point stands for: only needed to carry the tiers over
        if p["rig"] != DEFAULTS["rig"]:
            g["rig"] = p["rig"]                                                                           # only a graph of another rig than lane_center names it: the l4 graphs are unchanged
        return g


# ---------------------------------------------------------------------------------------------------------------- the store
def _file(sig_dir, token):
    return os.path.join(sig_dir, token + ".pkl.gz")


def _save(sig_dir, token, g):
    tmp = _file(sig_dir, token) + f".{os.getpid()}.tmp"
    with gzip.open(tmp, "wb", compresslevel=3) as f:
        pickle.dump(g, f, protocol=4)
    os.replace(tmp, _file(sig_dir, token))


def held(sig_dir, nonempty=False):
    """the sample tokens with a stored graph in the folder, by file name: one listing, 0.2 s for 34k files on the 9p mounts (the stat per file that an empty-file check needs took 101 s); a temporary
    file does not count. `nonempty` also drops a zero-byte file (a stat per file on 16 threads, 12 - 16 s: `generate` asks once per run, so that a damaged file is made again). A file is written
    whole and renamed (`_save`), so a zero-byte file is damage from outside; `missing()` and the readers do not look for it (`load` refuses it with FileNotFoundError): run `generate` to repair a store"""
    return store.names(sig_dir, ".pkl.gz", nonempty)


def load(token, data=DATA, **params):
    """the stored lane graph of a keyframe for the lane signature of THIS source and `params`. Raises FileNotFoundError that names the folders that exist."""
    sig_dir = os.path.join(data, signature(**params))
    path = _file(sig_dir, token)
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        have = sorted(d for d in os.listdir(data) if os.path.isdir(os.path.join(data, d))) if os.path.isdir(data) else []
        raise FileNotFoundError(f"no stored lane graph for keyframe {token} in {sig_dir} (folders in {data}: {have}); generate it with "
                                "`python -m GtGenerator generate-lane <nuScenes folder> --set full`, or the lane GT source changed since it was stored")
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def missing(tokens, data=DATA, **params):
    """the tokens without a stored graph for this source ([] = the store covers them)"""
    have = held(os.path.join(data, signature(**params)))
    return [t for t in tokens if t not in have]


def write_index(sig_dir, version, names, params, sets_complete, complete=True):
    """index.json of a lane store: what it holds and whether it is a release (complete, a released version, a clean tree, the set `full`). `names` = the sequences the store was asked for, those of
    every run on this folder (`generate` adds the earlier ones); `complete` = every keyframe of all of them is stored. One writer per folder: `generate` reads the old index and writes a new one, so two
    runs on the same folder at the same time can lose scene names (the frames are safe: each file is written whole); `environment` is that of the LAST run (a change is warned about, not listed)"""
    from . import versions
    held_now = held(sig_dir)
    state = versions.code_state()
    sig = os.path.basename(sig_dir)
    lane_version = version_of(sig)
    shown = {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}
    if RIGS[params.get("rig", DEFAULTS["rig"])]["window"] is not None:
        shown["window"] = list(RIGS[params["rig"]]["window"])                 # a rectangular rig: `half` is only a bound, the window is what the graphs are cut to
    idx = dict(signature=sig, lane_version=lane_version, version=version, scenes=list(names), keyframes=len(held_now), complete=bool(complete),
               params=shown, sets_complete=sorted(sets_complete),
               commit=state["commit"], dirty=state["dirty"], environment=store.environment(),
               release=bool(complete and lane_version and state["dirty"] is False and "full" in sets_complete))
    tmp = os.path.join(sig_dir, f"index.json.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(idx, f, indent=1)
    os.replace(tmp, os.path.join(sig_dir, "index.json"))
    return idx


def _warn_environment(before):
    """a store that grows under other versions of numpy, shapely or opencv than its index records (`before` = its `environment`): the frames of the runs can differ in the last digits.
    A warning, not a refusal: the index then records the versions of this run"""
    now = store.environment()
    changed = {k: (before[k], now[k]) for k in ("numpy", "shapely", "opencv-python") if k in (before or {}) and before[k] != now[k]}
    if changed:
        warnings.warn(f"this lane store was written under other library versions (index: {changed}, now): its frames may differ from the new ones in the last digits", RuntimeWarning, stacklevel=3)


def _group_task(args):
    code_sig, dataroot, version, params, grp, todo, sig_dir = args
    t0 = time.time()
    lg = LaneGT(dataroot, version, scenes=grp, **params)
    if lg.code_signature() != code_sig:                    # a worker imports the source afresh: it must be the source the job started with
        raise RuntimeError(f"a lane GT source file changed while this job runs (signature {code_sig} at the start, {lg.code_signature()} now): stop and start again")
    if lg.signature() != os.path.basename(sig_dir):        # ... and the table of stores (lane_versions.csv is not hashed) must still give the folder the job writes into
        raise RuntimeError(f"lane_versions.csv changed while this job runs (the store {os.path.basename(sig_dir)} at the start, {lg.signature()} now): stop and start again")
    for tok in todo:
        _save(sig_dir, tok, lg.graph(tok))
    return len(todo), time.time() - t0


def generate(dataroot, out=None, version=None, scenes=None, set_name=None, workers=1, verbose=True, **params):
    """make the lane graph of every keyframe of `scenes` (names; default: all sequences, or the `set_name` set "small" / "full") and store it under
    OUT/<lane signature>/ (default data/autogt/lane); files that exist are skipped (resumable); `workers` processes, one group of sequences each.
    The index.json of the folder names the sequences of every run on it (a key over several runs reads them all) and is `complete` when every keyframe of all of them is stored; a folder that was
    written under other versions of numpy / shapely / opencv gives a RuntimeWarning. -> {"out", "signature", "lane_version", "scenes", "keyframes", "written", "release"}"""
    from . import sets
    from .suite import json_tables, scene_tokens
    if set_name and scenes is not None:
        raise ValueError(f"pass either scenes= or set_name= ({set_name!r}), not both")
    dataroot = os.path.expanduser(dataroot)
    version = version or store.detect_version(dataroot)
    out = os.path.expanduser(out) if out else DATA
    known = [s["name"] for s in json_tables(dataroot, version)[0]]
    names = sets.scenes(set_name) if set_name else known if scenes is None else list(dict.fromkeys(scenes))
    if not names or [n for n in names if n not in known]:
        raise ValueError(f"unknown scene(s) {[n for n in names if n not in known] or 'none given'} in {version} of {dataroot}")
    p = _params(params)
    code_sig = code_signature(**p)
    sig_dir = os.path.join(out, signature(**p))                    # the store of this code: an earlier version's when the table says it makes the same GT
    os.makedirs(sig_dir, exist_ok=True)
    prior = store.read_index(sig_dir)
    _warn_environment(prior.get("environment"))
    asked = list(dict.fromkeys(prior.get("scenes", []) + names))   # the sequences of the earlier runs on this folder stay in its index
    tokens = {n: scene_tokens(n, version, dataroot) for n in asked if n in known}
    n_total = sum(len(tokens[n]) for n in names)
    stored = held(sig_dir, nonempty=True)
    size = min(GROUP_SCENES, max(1, -(-len(names) // workers)))
    groups = []
    for g0 in range(0, len(names), size):
        grp = names[g0:g0 + size]
        todo = [t for n in grp for t in tokens[n] if t not in stored]
        if todo:
            groups.append((grp, todo))
    if groups:
        write_index(sig_dir, version, asked, p, [], complete=False)
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
    complete = all(t in now for ts in tokens.values() for t in ts) and (len(tokens) == len(asked) or bool(prior.get("complete")))      # earlier sequences this nuScenes folder does not have: as the index said
    if done == 0 and complete and prior:                           # nothing to add to a complete store: its index keeps the commit and the environment that made it (a damaged index is written again)
        from . import versions
        can_upgrade = versions.code_state()["dirty"] is False and not prior.get("release")      # a clean tree can turn the index of an earlier, dirty run into a release
        if prior.get("complete") and set(names) <= set(prior.get("scenes", [])) and set(complete_sets) <= set(prior.get("sets_complete", [])) and not can_upgrade:
            if verbose:
                print(f"0 keyframes written, {n_total} already there: {sig_dir}  (index kept: lane version {prior['lane_version']}, commit {prior['commit']})")
            return {"out": sig_dir, "signature": signature(**p), "code_signature": code_sig, "lane_version": prior["lane_version"], "scenes": len(names), "keyframes": n_total,
                    "written": 0, "release": prior["release"]}
    idx = write_index(sig_dir, version, asked, p, complete_sets, complete=complete)
    if verbose:
        print(f"{done} keyframes written, {n_total - done} already there: {sig_dir}  (lane version {idx['lane_version']}, commit {idx['commit']}"
              f"{', dirty' if idx['dirty'] else ''}{', RELEASE' if idx['release'] else ''}, sets complete: {idx['sets_complete']})")
    return {"out": sig_dir, "signature": signature(**p), "code_signature": code_sig, "lane_version": idx["lane_version"], "scenes": len(names), "keyframes": n_total, "written": done,
            "release": idx["release"]}
