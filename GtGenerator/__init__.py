"""GtGenerator (package GtGenerator): ground truth for BEV perception from nuScenes (static layers, objects, masks, velocity / flow).

    import GtGenerator
    GtGenerator.generate("~/data/nuscenes_full")               # THE entry point: read nuScenes from a folder, write the GT of every keyframe
                                                          #   to data/autogt/<signature>/ (or out=...); resumable, memory-light
    GtGenerator.generate("~/data/nuscenes", set_name="small")  # a scene set: small (mini, small experiments) or full (trainval, releases)
    from GtGenerator import GtGenerator
    gt = GtGenerator("~/data/nuscenes", "v1.0-mini")          # one object, for single keyframes in code
    g = gt.keyframe(sample_token)                         # every layer of one keyframe (dict of arrays + meta)
    s = gt.training_sample(sample_token)                  # {"prev", "cur", "next"} + the ego steps between them
    gt.signature()                                        # cache key of this generator
    from GtGenerator import load                               # stored GT of a keyframe, no nuScenes needed
    g = load(sample_token)                                #   (data/autogt/<current_signature()>/<token>.npz)
    GtGenerator.scene_tokens("scene-0561", "v1.0-trainval")    # the tokens of a sequence from the json tables, no devkit

    key = GtGenerator.generate_bev_gt("~/data/nuscenes_full", set_name="full")   # BEV Fusion in one call: the front GT + the lane GT of its rig + the area masks -> the GT key (data + code signature)
    key = GtGenerator.generate_lane_gt(root, rig="lane_center")                  # Spurmitte (rig="all": both); key.load(token) / key.lane(token) / key.mask(token, "intersection", 0.5)

Command line (from the repository root): python -m GtGenerator COMMAND ... ; every command is one method of GtGenerator (generate, generate-bev, generate-lane, area-masks, area-mask, key, have, data, sets, versions, audit-lane, check, digest,
report, view, sheets, masks, lanegraph, video, clean, build, selftest: `python -m GtGenerator --help` has the map)
Guide for agents: GtGenerator/AGENTS.md; reference: GtGenerator/README.md
The velocity / flow estimator alone (dataset independent, numpy only): GtGenerator.flow.estimate_flow.
"""
from .flow import Config, FlowGT, ObjectFlow, Track, estimate_flow  # noqa: F401
from .pipeline import generate  # noqa: F401
from .grid import GridSpec  # noqa: F401
from . import sets, versions  # noqa: F401
from .suite import AutoGT, GtGenerator, current_signature, current_version, load, missing, scene_tokens  # noqa: F401
from .gt_api import GtKey, generate_bev_gt, generate_lane_gt, gt_key, key_of  # noqa: F401  (one function per system and the key it returns: GtGenerator/gt_api.py)


_LAZY = ("lane_gt", "lane_raster", "lane_graph", "rig_masks", "gt_api", "store", "pipeline", "cli", "clean", "checks", "coverage")


def __getattr__(name):
    """`GtGenerator.lane_gt`, `GtGenerator.rig_masks`, ... without importing them at `import GtGenerator` (PEP 562): the documented `GtGenerator.lane_gt.load(token)` works"""
    if name in _LAZY:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'GtGenerator' has no attribute {name!r}")

