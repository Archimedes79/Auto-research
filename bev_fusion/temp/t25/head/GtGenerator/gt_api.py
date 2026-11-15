"""The two functions of the suite, one per system, and the key they return.

    import GtGenerator
    key = GtGenerator.generate_bev_gt("~/data/nuscenes_full", set_name="full")     # BEV Fusion: the front GT + the lane GT of the rig bev_fusion + the area masks (intersection, borders 0 and 0.5 m)
    key = GtGenerator.generate_lane_gt("~/data/nuscenes_full", rig="lane_center")  # Spurmitte: the lane GT of the rig lane_center (rig="all": a dict of both keys)
    print(key)   # bev_fusion | data f615a7387e73 | v1.0-trainval full | release v12 | front 6a7aa7864a (the GT of store c643a7539d) @ 42c1792 | lane db288acca5 @ 181f72fe | masks of store db288acca5 | complete
    key.load(token)     # BEV Fusion: the front GT of a keyframe (what GtGenerator.load gives) ...
    key.lane(token)     # ... the lane graph of the system (frame ego for bev_fusion, lane for lane_center)
    key.mask(token, "intersection", border_m=0.5)    # ... and an area mask in the BEV grid, (120, 104) uint8
    GtGenerator.gt_key("bev_fusion", "v1.0-trainval", "full")    # the key of what is stored now, nothing is generated

The same as methods of the class: `GtGenerator.GtGenerator.generate_bev_gt(...)`, `GtGenerator.generate_lane_gt(...)`, `GtGenerator.gt_key(...)` (suite.py). Command line: `python -m GtGenerator generate-bev ROOT --set full`,
`generate-lane ROOT --rig lane_center|bev_fusion|all`, `generate ROOT` (everything), `key SYSTEM --set full` (cli.py: each command is one of these calls).

THE KEY says which GT it is: the DATA (the identity of the raw nuScenes files, `data_identity` of data/NuScenes/manifest.json: the hash of its tables and files; read from the store when the store
recorded it (the front index and the mask cache do), else from the manifest as it is now) and the CODE (per part the signature = hash of the generator files and options, the git commit of GtGenerator/ that made
the store and whether it was clean, the GT version, the suite release of releases.csv). `key.id` is a hash of the data and of the signatures of ALL parts of the system (also those a partial key does not
list) and of the rasteriser: two keys with the same id are the same GT, whatever set was asked for and whichever function made them. Not to be confused with the nuScenes SAMPLE TOKEN of a keyframe,
which the methods take. `key.complete` is True when every part is stored for what was asked for: the sets `full` / `small` by the sets the index.json of the store says it holds, a list of scenes by
those scenes, and, when the nuScenes folder is given, by every keyframe of the json tables; a store written with a stride is never complete. It never guesses: no index, no data identity = incomplete.

Parts: bev_fusion = front (data/autogt/<signature>/), lane (data/autogt/lane/<signature of the rig>/), masks (data/autogt/lane/<that signature>/masks/); lane_center = lane only. A version whose code makes the GT of an
older one (v9 -> v8, l5 lane_center -> l4) shows its own signature and "the GT of store ...". `out` is always the ROOT that holds the stores (default data/autogt; the lane stores are below `out/lane`).
"""
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

SYSTEMS = ("bev_fusion", "lane_center")                          # the rigs of lane_gt.RIGS (a test pins that they are the same)
BORDERS = (0.0, 0.5)
KINDS = ("intersection",)
_DATA_MEMO = {}


@dataclass(frozen=True)
class Part:
    name: str                       # front | lane | masks
    signature: str                  # the code signature (hash of the generator files and options); for the masks: the lane code signature / the rasteriser version
    store: str                      # the folder name of the stored GT below data/autogt (the store of an older version when the code makes its GT)
    version: Optional[str]          # the GT version of the STORE (v8 / l4), None = unreleased
    commit: Optional[str]           # git commit of GtGenerator/ when the store was made
    dirty: Optional[bool]           # GtGenerator/ had uncommitted changes then (None: git could not tell)
    complete: bool                  # the store holds everything asked for
    path: str                       # the folder


@dataclass(frozen=True)
class GtKey:
    system: str
    dataset: str                    # v1.0-trainval / v1.0-mini
    set_name: str                   # full | small | custom (scenes) | all
    data: str                       # data_identity[:12]: the identity of the raw nuScenes files ("unknown" when it cannot be read)
    release: Optional[str]          # the suite release of this source (releases.csv): v12
    parts: Tuple[Part, ...]
    out: str                        # data/autogt (the folder that holds the stores)
    id: str                         # 12 hex characters over the data and the signatures of ALL parts of the system: the same GT has the same id, whatever was asked for

    @property
    def complete(self):
        return self.data != "unknown" and all(p.complete for p in self.parts)

    def part(self, name):
        for p in self.parts:
            if p.name == name:
                return p
        raise ValueError(f"this key has no part {name!r} (it lists {[p.name for p in self.parts]})")

    def __str__(self):
        bits = [self.system, f"data {self.data}", f"{self.dataset} {self.set_name}", f"release {self.release or '-'}"]
        for p in self.parts:
            if p.name == "masks":
                bits.append(f"masks of store {p.store}")
                continue
            own = f" (the GT of store {p.store})" if p.store != p.signature else ""                 # a version whose code makes the GT of an older one: v9 -> v8, l5 lane_center -> l4
            bits.append(f"{p.name} {p.signature}{own} @ {p.commit or '?'}{'+dirty' if p.dirty else ''}")
        bits.append("complete" if self.complete else "INCOMPLETE: " + ", ".join([p.name for p in self.parts if not p.complete] + (["data identity"] if self.data == "unknown" else [])))
        return " | ".join(bits)

    def as_dict(self):
        return dict(id=self.id, system=self.system, dataset=self.dataset, set=self.set_name, data=self.data, release=self.release, complete=self.complete,
                    parts=[p.__dict__ for p in self.parts])

    # reading what the key names
    def load(self, token):
        """the front GT of a keyframe (BEV Fusion) from the store this key names (also when it was made with other generator options): the dict of layers + meta"""
        from .store import load_keyframe
        try:
            return load_keyframe(self.part("front").path, token)
        except FileNotFoundError:
            raise FileNotFoundError(f"no stored front GT for keyframe {token} in {self.part('front').path}") from None

    def lane(self, token):
        """the lane graph of the system: the rig bev_fusion (frame ego) or lane_center (frame lane)"""
        from . import lane_gt
        return lane_gt.load(token, data=os.path.join(self.out, "lane"), rig=self.system)

    def mask(self, token, kind="intersection", border_m=0.0):
        """an area mask in the BEV grid from the packed cache, (120, 104) uint8 (BEV Fusion only)"""
        from . import lane_raster
        return lane_raster.load_mask(token, kind, border_m, store_dir=self.part("lane").path)


def data_identity():
    """data_identity (12 hex characters) of data/NuScenes/manifest.json: the identity of the raw nuScenes files, "unknown" when the file cannot be read (a run snapshot has none)"""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "NuScenes", "manifest.json")
    try:
        st = os.stat(path)
        sig = (st.st_mtime_ns, st.st_size)
        if _DATA_MEMO.get(path, (None,))[0] != sig:
            with open(path, encoding="utf-8") as f:
                _DATA_MEMO[path] = (sig, str(json.load(f)["data_identity"])[:12])
        return _DATA_MEMO[path][1]
    except (OSError, ValueError, KeyError):
        return "unknown"


def _scenes_label(scenes):
    """what a key calls a scene list that is not a named set: `scenes:<id>`, the id of the list (sets.set_id)"""
    from . import sets
    return f"scenes:{sets.set_id(scenes)}"


def _tokens(dataroot, version, set_name, scenes):
    """the keyframe tokens asked for, from the json tables (None without a dataroot: then the index of the store says what it holds)"""
    if dataroot is None:
        return None
    from . import sets
    from .suite import json_tables, scene_tokens
    dataroot = os.path.expanduser(dataroot)
    names = sets.scenes(set_name) if set_name in ("full", "small") else scenes
    if names is None:
        names = [s["name"] for s in json_tables(dataroot, version)[0]]
    known = {s["name"] for s in json_tables(dataroot, version)[0]}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"unknown scene(s) {unknown[:5]}{' ...' if len(unknown) > 5 else ''} in {version} of {dataroot}")
    return {t for n in names for t in scene_tokens(n, version, dataroot)}


def _complete(idx, held, want, set_name, scenes):
    """a store holds what was asked for: every token of the json tables (`want`), else what its index.json says (the sets it holds in full, the scenes of its last run, written without a stride); never a guess"""
    if want is not None:
        return bool(held) and want <= held
    if not idx or not held or idx.get("stride", 1) != 1 or len(held) < idx.get("keyframes", 0):          # a thinned store (a lost or zero-byte file) is not complete whatever its index says
        return False
    if set_name in ("full", "small"):
        return set_name in idx.get("sets_complete", [])
    if scenes:
        return bool(idx.get("complete")) and set(scenes) <= set(idx.get("scenes", []))
    return False


def key_of(system, dataset="v1.0-trainval", set_name="full", out=None, parts=None, dataroot=None, scenes=None, options=None, borders=BORDERS, deep=True):
    """the key of what is stored for `system` ("bev_fusion" | "lane_center") below `out` (default data/autogt): nothing is generated. `parts` limits the parts listed (default all of the system); the id is the
    same either way. `options`: the generator options of the front GT (they change its signature), `borders`: the mask borders of the cache. `deep`: a zero-byte file does not count as a keyframe (a stat per
    file on 16 threads: 12 - 16 s per store of 34k files; False = by name only). See the module doc for what complete means"""
    from . import lane_gt, lane_raster, versions
    from .store import front_keys, names, read_index
    from .suite import DATA, code_signature, current_signature, current_version
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; known: {SYSTEMS}")
    all_parts = ("front", "lane", "masks") if system == "bev_fusion" else ("lane",)
    shown = tuple(parts) if parts else all_parts
    if [p for p in shown if p not in all_parts]:
        raise ValueError(f"the system {system} has the parts {all_parts}, not {tuple(shown)}")
    options = dict(options or {})
    out = os.path.expanduser(out) if out else DATA
    want = _tokens(dataroot, dataset, set_name, scenes)
    lane_store, bf_store = lane_gt.signature(rig=system), lane_gt.signature(rig="bev_fusion")
    raster = f"{lane_gt.code_signature(rig='bev_fusion')}/{lane_raster.RASTER_VERSION}"
    ident = {"front": (code_signature(**options), current_signature(**options)), "lane": (lane_gt.code_signature(rig=system), lane_store), "masks": (raster, bf_store)}
    lane_held = {}

    def lane_state(store):
        folder = os.path.join(out, "lane", store)
        if folder not in lane_held:
            lane_held[folder] = (read_index(folder), names(folder, ".pkl.gz", nonempty=deep))
        return folder, lane_held[folder]

    built = []
    for name in shown:
        sig, store = ident[name]
        if name == "front":
            folder = os.path.join(out, store)
            idx, held = read_index(folder), front_keys(folder, nonempty=deep)
            ver = current_version(**options)
            built.append(Part("front", sig, store, None if ver == "unreleased" else ver, idx.get("commit"), idx.get("dirty"), _complete(idx, held, want, set_name, scenes), folder))
        elif name == "lane":
            folder, (idx, held) = lane_state(store)
            built.append(Part("lane", sig, store, lane_gt.version_of(store), idx.get("commit"), idx.get("dirty"), _complete(idx, held, want, set_name, scenes), folder))
        else:
            folder, (idx, held) = lane_state(store)
            ok = _complete(idx, held, want, set_name, scenes) and all(lane_raster.cache_complete(folder, k, b, held=held) for k in KINDS for b in borders)
            built.append(Part("masks", sig, store, lane_raster.RASTER_VERSION, idx.get("commit"), idx.get("dirty"), ok, folder))
    # the data identity comes from the stores that recorded it, whichever parts are listed (so a partial key and the whole one agree after the manifest changed): the front index, the mask cache and
    # the lane index (written by make_lane_gt). Two stores that name different identities, or one that says "mixed", make the key say "mixed"; none recorded: the manifest of today
    seen = [str(read_index(os.path.join(out, "lane", lane_store)).get("data_identity", ""))[:12]]
    if system == "bev_fusion":
        seen.append(str(read_index(os.path.join(out, ident["front"][1])).get("data_identity", ""))[:12])
        info = lane_raster.cache_info(os.path.join(out, "lane", bf_store), KINDS[0], borders[0])
        seen.append(info["data"][:12] if info else "")
    known = {d for d in seen if d}
    data = "mixed" if len(known) > 1 or "mixed" in known else next(iter(known), None) or data_identity()
    h = hashlib.sha1("|".join([system, data] + [f"{n}:{ident[n][0]}:{ident[n][1]}" for n in all_parts]).encode())
    return GtKey(system, dataset, set_name if set_name else (_scenes_label(scenes) if scenes else "all"), data, versions.current_release(), tuple(built), out, h.hexdigest()[:12])


def _check(options, borders=BORDERS):
    for bad in ("stride", "res", "ss"):
        if bad in options:
            raise ValueError(f"{bad} is not an option here: the lane GT is made for every keyframe and the BEV grid is 120 x 104 cells of 0.5 m")
    if not borders:
        raise ValueError("borders: at least one (0.0 and 0.5 m are the usual)")


def _lane_identity(folder):
    """the data identity a lane store recorded (None: none, the store was made before the key recorded it)"""
    from .store import read_index
    return read_index(folder).get("data_identity")


def _record_lane_identity(folder, was, written):
    """the lane writer (a hashed file) leaves no data identity in its index.json and may write the index again (a clean tree upgrades it): this makes sure the index holds one after the run.
    `was` = what the index said before the run; frames written under a manifest.json other than the recorded one make it "mixed"; a store without a record that gets new frames stays unrecorded
    (the old frames have no known identity: no record rather than a wrong one); a store without a record and nothing written gets the identity of today (what the key assumed until now)"""
    path = os.path.join(folder, "index.json")
    now = data_identity()
    if now == "unknown" or not os.path.exists(path) or (was is None and written):
        return
    ident = now if was in (None, now) else "mixed"
    try:
        with open(path) as f:
            idx = json.load(f)
    except (OSError, ValueError):
        return
    if idx.get("data_identity") == ident:
        return
    idx["data_identity"] = ident
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(idx, f, indent=1)
    os.replace(tmp, path)


def make_lane_gt(dataroot, out, version, scenes=None, workers=1, verbose=True, rigs=SYSTEMS, masks=False, borders=BORDERS, set_name=None, strict=True):
    """the lane GT of the `rigs` into OUT/lane/ and, with `masks`, the packed area masks of the BEV Fusion store: the one place that does it (generate_bev_gt, generate_lane_gt and `generate` call it).
    -> ({rig: the result of lane_gt.generate}, masks_error): with strict=False a failure of the derived masks is returned as a message, not raised"""
    from . import lane_gt, lane_raster
    results, error = {}, None
    for rig in rigs:
        folder = os.path.join(out, "lane", lane_gt.signature(rig=rig))
        was = _lane_identity(folder)
        results[rig] = lane_gt.generate(dataroot, out=os.path.join(out, "lane"), version=version, scenes=scenes, set_name=set_name, workers=workers, verbose=verbose, rig=rig)
        _record_lane_identity(folder, was, written=results[rig].get("written", 0))
    if masks:
        try:
            lane_raster.build_mask_cache(os.path.join(out, "lane", lane_gt.signature(rig="bev_fusion")), kinds=KINDS, borders=tuple(borders), verbose=verbose, workers=workers)
        except Exception as e:                                           # the GT is written: with strict=False a failure of the derived masks does not lose the result
            if strict:
                raise
            error = f"{type(e).__name__}: {e}"
            print("WARNING: the mask cache was not built:", error, flush=True)
    return results, error


def _roots(dataroot, out, version):
    from .store import detect_version
    from .suite import DATA
    dataroot_ = os.path.expanduser(dataroot)
    return dataroot_, (version or detect_version(dataroot_)), (os.path.expanduser(out) if out else DATA)


def generate_bev_gt(dataroot, set_name=None, scenes=None, out=None, version=None, workers=1, borders=BORDERS, verbose=True, **options):
    """BEV Fusion: the front GT (GtGenerator.generate without the lane GT; `options` are its generator options), the lane GT of the rig bev_fusion and the packed area masks (intersection at the `borders` in metres),
    for the sequences of `set_name` ("small" | "full") or `scenes`; all resumable. -> GtKey"""
    from .pipeline import generate
    _check(options, borders)
    dataroot_, version, out_ = _roots(dataroot, out, version)
    generate(dataroot_, out_, version, scenes, set_name=set_name, workers=workers, lane=False, verbose=verbose, **options)
    from .light_nusc import ensure_cache
    if "trainval" in version:
        ensure_cache(dataroot_, version, verbose=verbose)
    make_lane_gt(dataroot_, out_, version, scenes, workers, verbose, rigs=("bev_fusion",), masks=True, borders=tuple(borders), set_name=set_name)
    return key_of("bev_fusion", version, set_name or None, out_, dataroot=dataroot_, scenes=scenes, options=options, borders=tuple(borders))


def generate_lane_gt(dataroot, rig="all", set_name=None, scenes=None, out=None, version=None, workers=1, verbose=True):
    """the lane GT of one system (rig "lane_center" = Spurmitte, "bev_fusion" = BEV Fusion's lane part only; the masks come with generate_bev_gt) or both ("all": a dict rig -> GtKey);
    for the sequences of `set_name` or `scenes`; resumable. -> GtKey (or a dict of them)"""
    if rig != "all" and rig not in SYSTEMS:
        raise ValueError(f"unknown rig {rig!r}; known: {SYSTEMS + ('all',)}")
    dataroot_, version, out_ = _roots(dataroot, out, version)
    from .light_nusc import ensure_cache
    if "trainval" in version:
        ensure_cache(dataroot_, version, verbose=verbose)
    rigs = SYSTEMS if rig == "all" else (rig,)
    make_lane_gt(dataroot_, out_, version, scenes, workers, verbose, rigs=rigs, set_name=set_name)
    keys = {r: key_of(r, version, set_name or None, out_, parts=("lane",), dataroot=dataroot_, scenes=scenes) for r in rigs}
    return keys if rig == "all" else keys[rig]


DATASETS = ("v1.0-mini", "v1.0-trainval")


def gt_key(system, dataset="v1.0-trainval", set_name="full", out=None, dataroot=None, scenes=None, deep=True, borders=BORDERS):
    """the key of the stored GT of `system` for a set (or `scenes`), without generating anything (see key_of; complete is judged by the index.json of each store AND the files that are there, or by the json tables with `dataroot`; `deep=False`: by file name only, 30 s faster on the full set)"""
    if dataset not in DATASETS:
        raise ValueError(f"unknown nuScenes version {dataset!r}: {' | '.join(DATASETS)}")
    if not scenes:
        from . import sets
        sets.scenes(set_name)                                           # an unknown set is a wrong argument (ValueError), not a store that is missing
    return key_of(system, dataset, None if scenes else set_name, out, dataroot=dataroot, scenes=scenes, deep=deep, borders=tuple(borders))
