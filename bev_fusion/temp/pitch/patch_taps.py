p = "evaluation/bev_adapter.py"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new, 1)


rep('''            TapSpec("radar_feat", "feature_map", "radar", "the raster smeared by a 3x3 context", "features without raster dots", **kw),
            TapSpec("fused", "feature_map", "fusion", "camera and radar features together", "dead channels", **kw),''',
    '''            TapSpec("radar_feat", "feature_map", "radar", "the raster smeared by a 3x3 context (delta_slots: the features of the slot encoder)", "features without raster dots", **kw),
            TapSpec("radar_state", "feature_map", "radar", "delta_slots only: the lossless slot raster (valid, dr, dpsi, Doppler code, rcs, age, offsets per slot) next to the features: "
                    "blobs where the detections are", "empty although there are detections, or blobs away from the objects", **kw),
            TapSpec("radar_memory", "feature_map", "radar", "the radar memory (rad_temporal, when on): older sweeps moved to the present", "never changes from frame to frame", **kw),
            TapSpec("fused", "feature_map", "fusion", "camera and radar features together", "dead channels", **kw),''')
rep('''            TapSpec("trunk", "feature_map", "trunk", "feature regions: road, off-road, objects", "noise or one single colour",
                    extent=ext, flip_x=True)]''',
    '''            TapSpec("trunk", "feature_map", "trunk", "feature regions: road, off-road, objects", "noise or one single colour",
                    extent=ext, flip_x=True),
            TapSpec("motion_l0", "feature_map", "motion", "motion block, coarsest level: |v| smooth, bright on moving objects", "uniform, or a field along the ego motion", **kw),
            TapSpec("motion_l1", "feature_map", "motion", "motion block, middle level: the coarse field refined", "noise, or the coarse field copied", **kw),
            TapSpec("motion_l2", "feature_map", "motion", "motion block, finest level = v_hat: moving objects bright, static cells ~ 0", "bright on static cells (ego motion not removed)", **kw),
            TapSpec("hurdle_mov", "logits", "heads", "the hurdle's moving / static logits per class (max sigmoid shown): high on moving objects", "high everywhere, or nowhere",
                    **{**kw, "alive_check": False}),
            TapSpec("radar_aux", "logits", "heads", "the radar auxiliary head on the radar branch (evaluation-time view; trained only with aux_heads): radar-visible layers", "empty", **kw)]''')
rep('''            "radar_feat": Hook("radar"),
            "fused": Hook("fusion")}''',
    '''            "radar_feat": Hook("radar", pick=_first_of_pair),
            "radar_state": Hook("radar", pick=_state_of_pair),
            "radar_memory": Hook("rad_temporal", pick=lambda o: o[0]),
            "fused": Hook("fusion"),
            "motion_l0": Hook("motion_block", pick=lambda o: o[0]),
            "motion_l1": Hook("motion_block", pick=lambda o: o[1]),
            "motion_l2": Hook("motion_block", pick=lambda o: o[-1]),                    # the finest level, whatever the number of levels
            "_rad_branch": Hook("rad_temporal", pick=lambda o: o[0])}                   # private (no TapSpec): the input of the radar aux head when the radar memory is on''')
rep('''def bev_hooks() -> dict:''', '''def _first_of_pair(o):
    """the radar encoder returns (features, lossless state) for radar_kind 'delta_slots' (with_state=True), the features alone otherwise"""
    return o[0] if isinstance(o, tuple) else o


def _state_of_pair(o):
    """the lossless raster of a delta_slots radar encoder; anything else has none (the recorder skips a tap that cannot be read)"""
    return o[1] if isinstance(o, tuple) and len(o) > 1 else None


def bev_hooks() -> dict:''')
rep('''            self.rec.arm([n for n in names if n in hooks])''',
    '''            want = set(names)
            if "radar_aux" in names:                                       # the aux head reads the radar branch: its input is recorded too (not returned)
                want |= {"radar_feat", "_rad_branch"}
            self.rec.arm([n for n in want if n in hooks])''')
rep('''                if "camera_visibility" in names and hasattr(self.model, "camera_visibility"):''',
    '''                if "hurdle_mov" in names and "mov" in res["out"][0.0]["obj"]:
                    arrays["hurdle_mov"] = res["out"][0.0]["obj"]["mov"][0].float().cpu().numpy()
                if "radar_aux" in names and getattr(self.model, "aux_rad", None) is not None:        # the head is only run in training; here it reads the branch's features
                    feat = arrays.get("_rad_branch", arrays.get("radar_feat"))
                    if feat is not None:
                        arrays["radar_aux"] = self.model.aux_rad(torch.from_numpy(feat)[None].to(self.device))[0].float().cpu().numpy()
                if "camera_visibility" in names and hasattr(self.model, "camera_visibility"):''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
