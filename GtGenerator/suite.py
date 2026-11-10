"""The one-object interface of GtGenerator (see GtGenerator/AGENTS.md for the short guide)."""
import functools
import json
import os
import types

from .store import load_keyframe
from .grid import GridSpec
from .api import Interface
from .nuscenes_gt import NuScenesGT

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "autogt")       # <repository root>/data/autogt


_SIGNATURES = {}


def code_signature(**options):
    """the signature (hash of the generator files and options) of THIS source, without loading nuScenes. Computed on the first call of a
    process and kept: like a GtGenerator object it holds the key of the source it started with, so a process that reads the store (`load`,
    `missing`) does not change folders when a signature file is edited while it runs."""
    key = repr(sorted(options.items()))
    if key not in _SIGNATURES:
        opt = dict(stationary_fix=True, map_offsets={}, use_flow=True, use_radar=True, flow_config=None, boxes_at="camera")
        opt.update(options)
        _SIGNATURES[key] = NuScenesGT.signature(types.SimpleNamespace(**opt))
    return _SIGNATURES[key]


def current_signature(**options):
    """the folder name of the stored GT that THIS source makes and reads (data/autogt/<this>/): its own signature, or that of the older
    version whose GT it makes byte for byte (`same_gt_as` in versions.csv)"""
    from .versions import canonical
    return canonical(code_signature(**options))


def current_version(**options):
    """the version key (GtGenerator/versions.csv, decision D29) of THIS source: "v5"; with non-default options "v5+boxes_at='annotation'";
    "unreleased" when the signature has no row yet (an edit of a signature file that was not versioned: `GtGenerator selftest` fails then)"""
    from .versions import version_of
    base = version_of(current_signature())
    if base is None:
        return "unreleased"
    return base + ("+" + ",".join(f"{k}={v!r}" for k, v in sorted(options.items())) if options else "")


def missing(tokens, data=DATA, **options):
    """the sample tokens of `tokens` without a stored GT for the current source and options ([] = the store covers them): the check
    before a training starts, so that nothing has to be generated in a data-loader worker"""
    sig_dir = os.path.join(data, current_signature(**options))
    tokens = list(tokens)
    if len(tokens) > 50:                                                  # one listing of the folder instead of a stat per token (2 ms each on the 9p mounts); a zero-byte file is no keyframe
        from .store import front_keys
        have = front_keys(sig_dir, nonempty=True)
        return [t for t in tokens if t not in have]

    def stored(t):
        try:
            return os.path.getsize(os.path.join(sig_dir, t + ".npz")) > 0
        except OSError:
            return False
    return [t for t in tokens if not stored(t)]


def load(token, data=DATA, **options):
    """the stored GT of one keyframe (written by `python -m GtGenerator report` / `build`) for the current source and options:
    dict of float32 layers + meta, no nuScenes needed. Raises FileNotFoundError that names the folders that do exist."""
    sig_dir = os.path.join(data, current_signature(**options))
    try:
        return load_keyframe(sig_dir, token)
    except FileNotFoundError:
        have = sorted(d for d in os.listdir(data) if os.path.isdir(os.path.join(data, d)) and d not in ("reports", "examples", "lane", "cache")) if os.path.isdir(data) else []
        raise FileNotFoundError(f"no stored GT for keyframe {token} in {sig_dir} (folders in {data}: {have}); generate it with "
                                "`GtGenerator.generate(<nuScenes folder>)` (shell: `python -m GtGenerator generate`), or the generator source changed "
                                "since it was stored") from None


def default_root(version):
    return os.path.expanduser("~/data/nuscenes_full" if "trainval" in version else "~/data/nuscenes")


@functools.lru_cache(maxsize=4)
def json_tables(dataroot, version):
    """(scene records, sample token -> next sample token) from the nuScenes json tables, read once per process."""
    tab = os.path.join(dataroot, version)
    return (json.load(open(os.path.join(tab, "scene.json"))),
            {r["token"]: r["next"] for r in json.load(open(os.path.join(tab, "sample.json")))})


def scene_tokens(scene, version="v1.0-mini", dataroot=None):
    """the sample tokens of a sequence in time order, from the nuScenes json tables only: a few seconds even for trainval, no devkit
    (`GtGenerator(...).keyframes(scene)` gives the same list). dataroot default: ~/data/nuscenes (mini) / ~/data/nuscenes_full (trainval)."""
    scenes, nxt = json_tables(os.path.expanduser(dataroot) if dataroot else default_root(version), version)
    first = next((s["first_sample_token"] for s in scenes if s["name"] == scene), None)
    if first is None:
        raise ValueError(f"unknown scene {scene!r} in {version}")
    out, tok = [], first
    while tok:
        out.append(tok)
        tok = nxt[tok]
    return out


class GtGenerator(NuScenesGT, Interface):
    """GtGenerator(dataroot, version, grid=GridSpec(), ss=4, **options).

    keyframe(token)          every layer of one keyframe: dict of float32 arrays [ix, iy] + meta (see nuscenes_gt.py)
    training_sample(token)   {"prev", "cur", "next"} keyframes with meta["dt"] and meta["steps"] (ego v, omega per 12 Hz frame)
    keyframes(scene_name)    sample tokens of a scene in time order
    scenes()                 scene names
    signature()              cache key of the generator (source + options)
    options: stationary_fix, map_offsets, use_flow, use_radar, flow_config (GtGenerator.flow.Config), boxes_at ("camera" = moving
    boxes moved to the camera time by v * dt, default; "annotation" = as annotated)
    light=True: a memory-light devkit (CAM_FRONT + radars only, GtGenerator/light_nusc.py); scenes=[names]: only these sequences are
    loaded (v1.0-trainval: ~0.7 GB for 85 sequences instead of 8+ GB for all)

    The same class is the interface of the whole suite: its STATIC methods (GtGenerator/api.py) are what the command line calls, `GtGenerator.generate_bev_gt(root)`, `GtGenerator.generate_lane_gt(root)`,
    `GtGenerator.gt_key(system)`, `GtGenerator.generate`, `have`, `check_data`, `audit_lane`, `area_masks`, `sets`, `versions`, `view`, `video`, `check`, `report`, `clean`, `selftest` ...
    """

    def __init__(self, dataroot, version, grid=None, ss=4, light=False, scenes=None, **options):
        dataroot = os.path.expanduser(dataroot)
        if light:
            from .light_nusc import light_nuscenes
            options["nusc"] = light_nuscenes(dataroot, version, scenes)
        super().__init__(dataroot, version, **options)
        self.grid, self.ss = grid or GridSpec(), ss

    def keyframe(self, sample_token, grid=None, ss=None, sd=None):
        return self.keyframe_gt(sample_token, grid or self.grid, ss or self.ss, sd)

    def training_sample(self, sample_token, grid=None, ss=None):
        return super().training_sample(sample_token, grid or self.grid, ss or self.ss)

    # one function per system and its key (gt_api.py): the class offers them next to the single-keyframe methods
    def scenes(self):
        return [s["name"] for s in self.nusc.scene]

    def keyframes(self, scene_name):
        sc = next((s for s in self.nusc.scene if s["name"] == scene_name), None)
        if sc is None:
            raise ValueError(f"unknown scene {scene_name!r} (this devkit holds {len(self.nusc.scene)} sequences)")
        out, tok = [], sc["first_sample_token"]
        while tok:
            out.append(tok)
            tok = self.nusc.get("sample", tok)["next"]
        return out


AutoGT = GtGenerator        # the old name of the class: the hashed generator files and the BEV project import it (a rename of those files would change the signature)
