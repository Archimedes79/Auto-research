"""The two functions of the suite, one per system, and the key they return.

    import autogt
    key = autogt.generate_bev_gt("~/data/nuscenes_full", set_name="full")     # BEV Fusion: the front GT + the lane GT of the rig bev_fusion + the area masks (intersection, borders 0 and 0.5 m)
    key = autogt.generate_lane_gt("~/data/nuscenes_full", rig="lane_center")  # Spurmitte: the lane GT of the rig lane_center (rig="all": a dict of both keys)
    print(key)   # bev_fusion | data f615a7387e73 | v1.0-trainval full | release v12 | front 6a7aa7864a (the GT of store c643a7539d) @ 42c1792 | lane db288acca5 @ 181f72fe | masks of store db288acca5 | complete
    key.load(token)     # BEV Fusion: the front GT of a keyframe (what autogt.load gives) ...
    key.lane(token)     # ... the lane graph of the system (frame ego for bev_fusion, lane for lane_center)
    key.mask(token, "intersection", border_m=0.5)    # ... and an area mask in the BEV grid, (120, 104) uint8
    autogt.gt_key("bev_fusion", "v1.0-trainval", "full")    # the key of what is stored now, nothing is generated

Command line: `python -m autogt generate-bev ROOT --set full`, `generate-lane ROOT --rig lane_center|bev_fusion|all` (both print the key), `generate ROOT` (everything).

THE KEY says which GT it is: the DATA (the identity of the raw nuScenes files, `data_identity` of data/NuScenes/manifest.json: the hash of its tables and files; read from the store when the store
recorded it (the front index and the mask cache do), else from the manifest as it is now) and the CODE (per part the signature = hash of the generator files and options, the git commit of autogt/ that made
the store and whether it was clean, the GT version, the suite release of releases.csv). `key.id` is a hash of the data and of the signatures of ALL parts of the system (also those a partial key does not
list) and of the rasteriser: two keys with the same id are the same GT, whatever set was asked for and whichever function made them. Not to be confused with the nuScenes SAMPLE TOKEN of a keyframe,
which the methods take. `key.complete` is True when every part is stored for what was asked for: the sets `full` / `small` by the sets the index.json of the store says it holds, a list of scenes by
those scenes, and, when the nuScenes folder is given, by every keyframe of the json tables. It never guesses: no index, no data identity = incomplete.

Parts: bev_fusion = front (data/autogt/<signature>/), lane (data/autogt/lane/<signature of the rig>/), masks (data/autogt/lane/<that signature>/masks/); lane_center = lane only. A version whose code makes the GT of an
older one (v9 -> v8, l5 lane_center -> l4) shows its own signature and "the GT of store ...".
"""
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

SYSTEMS = ("bev_fusion", "lane_center")
BORDERS = (0.0, 0.5)
KINDS = ("intersection",)
_DATA_MEMO = {}


@dataclass(frozen=True)
class Part:
    name: str                       # front | lane | masks
    signature: str                  # the code signature (hash of the generator files and options); for the masks: the lane code signature / the rasteriser version
    store: str                      # the folder name of the stored GT below data/autogt (the store of an older version when the code makes its GT)
    version: Optional[str]          # the GT version of the STORE (v8 / l4), None = unreleased
    commit: Optional[str]           # git commit of autogt/ when the store was made
    dirty: Optional[bool]           # autogt/ had uncommitted changes then (None: git could not tell)
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
    ident: str                      # the id (see `id`)

    @property
    def complete(self):
        return self.data != "unknown" and all(p.complete for p in self.parts)

    @property
    def id(self):
        """12 hex characters over the data and the signatures of ALL parts of the system: the same GT has the same id, whatever was asked for"""
        return self.ident

    def part(self, name):
        return next(p for p in self.parts if p.name == name)

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
        """the front GT of a keyframe (BEV Fusion): the dict of layers + meta, as autogt.load"""
        from .suite import load
        return load(token, data=self.out)

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


def _index(folder):
    try:
        with open(os.path.join(folder, "index.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _names(folder, suffix):
    if not os.path.isdir(folder):
        return set()
    with os.scandir(folder) as it:
        return {e.name[:-len(suffix)] for e in it if e.name.endswith(suffix) and ".tmp" not in e.name}


def _tokens(dataroot, version, set_name, scenes):
    """the keyframe tokens asked for, from the json tables (None without a dataroot: then the index of the store says what it holds)"""
    if dataroot is None:
        return None
    from . import sets
    from .suite import scene_tokens
    dataroot = os.path.expanduser(dataroot)
    names = sets.scenes(set_name) if set_name in ("full", "small") else scenes
    if names is None:
        names = [s["name"] for s in json.load(open(os.path.join(dataroot, version, "scene.json")))]
    return {t for n in names for t in scene_tokens(n, version, dataroot)}


def _complete(idx, held, want, set_name, scenes):
    """a store holds what was asked for: every token of the json tables (`want`), else what its index.json says (the sets it holds in full, the scenes of its last run); never a guess"""
    if want is not None:
        return bool(held) and want <= held
    if not idx or not held:
        return False
    if set_name in ("full", "small"):
        return set_name in idx.get("sets_complete", [])
    if scenes:
        return bool(idx.get("complete")) and set(scenes) <= set(idx.get("scenes", []))
    return False


def key_of(system, dataset="v1.0-trainval", set_name="full", out=None, parts=None, dataroot=None, scenes=None, options=None, borders=BORDERS):
    """the key of what is stored for `system` ("bev_fusion" | "lane_center") below `out` (default data/autogt): nothing is generated. `parts` limits the parts listed (default all of the system); the id is the
    same either way. `options`: the generator options of the front GT (they change its signature), `borders`: the mask borders of the cache. See the module doc for what complete means"""
    from . import lane_gt, lane_raster, versions
    from .suite import DATA, code_signature, current_signature, current_version
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; known: {SYSTEMS}")
    options = dict(options or {})
    out = os.path.expanduser(out) if out else DATA
    want = _tokens(dataroot, dataset, set_name, scenes)
    all_parts = ("front", "lane", "masks") if system == "bev_fusion" else ("lane",)
    shown = tuple(parts) if parts else all_parts
    lane_store = lane_gt.signature(rig=system)
    bf_store = lane_gt.signature(rig="bev_fusion")
    raster = f"{lane_gt.code_signature(rig='bev_fusion')}/{lane_raster.RASTER_VERSION}"
    ident = {"front": (code_signature(**options), current_signature(**options)), "lane": (lane_gt.code_signature(rig=system), lane_store), "masks": (raster, bf_store)}
    built, data_seen, lane_cache = [], [], {}
    for name in shown:
        sig, store = ident[name]
        if name == "front":
            folder = os.path.join(out, store)
            idx = _index(folder)
            held = _names(folder, ".npz")
            ver = current_version(**options)
            built.append(Part("front", sig, store, None if ver == "unreleased" else ver, idx.get("commit"), idx.get("dirty"), _complete(idx, held, want, set_name, scenes), folder))
            if idx.get("data_identity"):
                data_seen.append(str(idx["data_identity"])[:12])
        elif name == "lane":
            folder = os.path.join(out, "lane", store)
            idx = _index(folder)
            held = _names(folder, ".pkl.gz")
            lane_cache[folder] = (idx, held)
            built.append(Part("lane", sig, store, lane_gt.version_of(store), idx.get("commit"), idx.get("dirty"), _complete(idx, held, want, set_name, scenes), folder))
        else:
            folder = os.path.join(out, "lane", store)
            idx, held = lane_cache.get(folder) or (_index(folder), _names(folder, ".pkl.gz"))
            ok = _complete(idx, held, want, set_name, scenes) and all(lane_raster.cache_complete(folder, k, b, held=held) for k in KINDS for b in borders)
            info = lane_raster.cache_info(folder, KINDS[0], borders[0])
            if info and info["data"]:
                data_seen.append(info["data"][:12])
            built.append(Part("masks", sig, store, lane_raster.RASTER_VERSION, idx.get("commit"), idx.get("dirty"), ok, folder))
    now = data_identity()
    data = data_seen[0] if data_seen else now
    h = hashlib.sha1("|".join([system, data] + [f"{n}:{ident[n][0]}:{ident[n][1]}" for n in all_parts]).encode())
    return GtKey(system, dataset, set_name if set_name else ("custom" if scenes else "all"), data, versions.current_release(), tuple(built), out, h.hexdigest()[:12])


def _check(options, borders):
    if "stride" in options:
        raise ValueError("the lane GT is made for every keyframe: no stride")
    if not borders:
        raise ValueError("borders: at least one (0.0 and 0.5 m are the usual)")


def generate_bev_gt(dataroot, set_name=None, scenes=None, out=None, version=None, workers=1, borders=BORDERS, verbose=True, **options):
    """BEV Fusion: the front GT (autogt.generate without the lane GT; `options` are its generator options), the lane GT of the rig bev_fusion and the packed area masks (intersection at the `borders` in metres),
    for the sequences of `set_name` ("small" | "full") or `scenes`; all resumable. -> GtKey"""
    from . import lane_gt, lane_raster
    from .cli import detect_version, generate
    from .light_nusc import ensure_cache
    from .suite import DATA
    _check(options, borders)
    dataroot_ = os.path.expanduser(dataroot)
    version = version or detect_version(dataroot_)
    out_ = os.path.expanduser(out) if out else DATA
    if "trainval" in version:
        ensure_cache(dataroot_, version, verbose=verbose)
    generate(dataroot_, out, version, scenes, set_name=set_name, workers=workers, lane=False, verbose=verbose, **options)
    lane_gt.generate(dataroot_, out=os.path.join(out_, "lane"), version=version, scenes=scenes, set_name=set_name, workers=workers, verbose=verbose, rig="bev_fusion")
    lane_raster.build_mask_cache(os.path.join(out_, "lane", lane_gt.signature(rig="bev_fusion")), kinds=KINDS, borders=tuple(borders), verbose=verbose)
    return key_of("bev_fusion", version, set_name or ("custom" if scenes else None), out_, dataroot=dataroot_, scenes=scenes, options=options, borders=tuple(borders))


def generate_lane_gt(dataroot, rig="all", set_name=None, scenes=None, out=None, version=None, workers=1, verbose=True):
    """the lane GT of one system (rig "lane_center" = Spurmitte, "bev_fusion" = BEV Fusion's lane part only; the masks come with generate_bev_gt) or both ("all": a dict rig -> GtKey);
    for the sequences of `set_name` or `scenes`; resumable. -> GtKey (or a dict of them)"""
    from . import lane_gt
    from .cli import detect_version
    from .light_nusc import ensure_cache
    from .suite import DATA
    rigs = SYSTEMS if rig == "all" else (rig,)
    dataroot_ = os.path.expanduser(dataroot)
    version = version or detect_version(dataroot_)
    out_ = os.path.expanduser(out) if out else DATA
    if "trainval" in version:
        ensure_cache(dataroot_, version, verbose=verbose)
    keys = {}
    for r in rigs:
        lane_gt.generate(dataroot_, out=os.path.join(out_, "lane"), version=version, scenes=scenes, set_name=set_name, workers=workers, verbose=verbose, rig=r)
        keys[r] = key_of(r, version, set_name or ("custom" if scenes else None), out_, parts=("lane",), dataroot=dataroot_, scenes=scenes)
    return keys if rig == "all" else keys[rig]


def gt_key(system, dataset="v1.0-trainval", set_name="full", out=None):
    """the key of the stored GT of `system` for a set, without generating anything (see key_of; complete is judged by the index.json of each store: the sets it holds in full)"""
    return key_of(system, dataset, set_name, out)
