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
# Nothing below is imported at `import GtGenerator` (PEP 562): a name or a submodule is imported when it is first asked for (`GtGenerator.generate`, `GtGenerator.versions`, `from GtGenerator import GtKey`),
# so the package imports without numpy. The job runner reads `GtGenerator.JOBS` in its own interpreter (plain python3), and `import GtGenerator` costs nothing for the tools that only need a name.
_NAMES = {"flow": ("Config", "FlowGT", "ObjectFlow", "Track", "estimate_flow"), "pipeline": ("generate",), "grid": ("GridSpec",),
          "suite": ("AutoGT", "GtGenerator", "current_signature", "current_version", "load", "missing", "scene_tokens"),
          "gt_api": ("GtKey", "generate_bev_gt", "generate_lane_gt", "gt_key", "key_of"),      # one function per system and the key it returns: GtGenerator/gt_api.py
          "jobs": ("JOBS",)}                                                                  # the calls a job file can name (GtGenerator/jobs.py)
_OWNER = {name: module for module, names in _NAMES.items() for name in names}


def __getattr__(name):
    """`GtGenerator.generate`, `GtGenerator.versions`, `GtGenerator.lane_gt.load(token)` ... imported on first use and kept as plain attributes (a name of a module of this folder is that module)"""
    import importlib
    import os
    if name in _OWNER:
        value = getattr(importlib.import_module(f".{_OWNER[name]}", __name__), name)
        globals()[name] = value
        return value
    if name.isidentifier() and os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), name + ".py")):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'GtGenerator' has no attribute {name!r}")


def __dir__():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    return sorted(set(globals()) | set(_OWNER) | {f[:-3] for f in os.listdir(here) if f.endswith(".py") and f[:-3].isidentifier()})
