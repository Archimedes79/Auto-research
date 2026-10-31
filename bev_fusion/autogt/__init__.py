"""autoGT: automatic ground truth for BEV perception from nuScenes (static layers, objects, masks, velocity / flow).

    from autogt import AutoGT
    gt = AutoGT("~/data/nuscenes", "v1.0-mini")          # one object
    g = gt.keyframe(sample_token)                         # every layer of one keyframe (dict of arrays + meta)
    s = gt.training_sample(sample_token)                  # {"prev", "cur", "next"} + the ego steps between them
    gt.signature()                                        # cache key of this generator
    from autogt import load                               # stored GT of a keyframe, no nuScenes needed
    g = load(sample_token)                                #   (data/autogt/<current_signature()>/<token>.npz)
    autogt.scene_tokens("scene-0561", "v1.0-trainval")    # the tokens of a sequence from the json tables, no devkit

Command line (from the bev_fusion folder): python -m autogt {selftest,build,check,view,sheets,video,report,clean} ...
Guide for agents: autogt/AGENTS.md; reference: autogt/README.md
The velocity / flow estimator alone (dataset independent, numpy only): autogt.flow.estimate_flow.
"""
from .flow import Config, FlowGT, ObjectFlow, Track, estimate_flow  # noqa: F401
from .grid import GridSpec  # noqa: F401
from .suite import AutoGT, current_signature, load, scene_tokens  # noqa: F401
