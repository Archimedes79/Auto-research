"""The one-object interface of the autoGT suite."""
import os

from .grid import GridSpec
from .nuscenes_gt import NuScenesGT


class AutoGT(NuScenesGT):
    """AutoGT(dataroot, version, grid=GridSpec(), ss=4, **options).

    keyframe(token)          every layer of one keyframe: dict of float32 arrays [ix, iy] + meta (see nuscenes_gt.py)
    training_sample(token)   {"prev", "cur", "next"} keyframes with meta["dt"] and meta["steps"] (ego v, omega per 12 Hz frame)
    keyframes(scene_name)    sample tokens of a scene in time order
    scenes()                 scene names
    signature()              cache key of the generator (source + options)
    options: stationary_fix, map_offsets, use_flow, use_radar, flow_config (autogt.flow.Config), boxes_at ("camera" = moving
    boxes moved to the camera time by v * dt, default; "annotation" = as annotated)
    light=True: a memory-light devkit (CAM_FRONT + radars only, autogt/light_nusc.py); scenes=[names]: only these sequences are
    loaded (v1.0-trainval: ~0.7 GB for 85 sequences instead of 8+ GB for all)
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

    def scenes(self):
        return [s["name"] for s in self.nusc.scene]

    def keyframes(self, scene_name):
        sc = next(s for s in self.nusc.scene if s["name"] == scene_name)
        out, tok = [], sc["first_sample_token"]
        while tok:
            out.append(tok)
            tok = self.nusc.get("sample", tok)["next"]
        return out
