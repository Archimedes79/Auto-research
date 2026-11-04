p = "training/model.py"
s = open(p, encoding="utf-8").read()

def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a)
    s = s.replace(a, b)

# 1 module docstring
rep('''                "none"           current frame only
  rad_temporal''', '''                "none"           current frame only
  cam_buffer    "window" (default) the batch holds the window of n_cam frames, all lifted on every call; a static world memory is the only
                state | "pair" the window is ONE frame (t-1): the batch holds 2 frames, or in stream mode ONE frame and the previous raw
                camera BEV comes from state["buf"] | "recurrent" the temporal block's own output of the previous frame is the buffer
                (state["buf"] = raw + fused): each frame is lifted once and fused with the previous fused state (splat_window on it, its
                velocity head), the batch unrolls its F frames oldest -> newest from cold. The radar keeps its own 0.5 s point buffer
  rad_temporal''')

# 2 imports
rep("    from .temporal_fusion import TemporalFusion\n", "    from .temporal_fusion import TemporalFusion, advance_chain, as_chain, retreat_pose\n")
rep("    from temporal_fusion import TemporalFusion\n", "    from temporal_fusion import TemporalFusion, advance_chain, as_chain, retreat_pose\n")

# 3 ModelConfig field
rep('''    cam_temporal: str = "window_memory"
''', '''    cam_temporal: str = "window_memory"
    cam_buffer: str = "window"         # t21: what the camera temporal block carries from the past: "window" (the batch's n_cam frames, today) | "pair" (the
                                       # previous raw BEV) | "recurrent" (the previous fused output); the last two need cam_temporal 'window_memory'
''')

# 4 table + check
rep('''FUSIONS = {"concat"''', '''CAM_BUFFERS = ("window", "pair", "recurrent")                          # what the camera temporal block gets from the past (forward: _camera_buffered)
FUSIONS = {"concat"''')
rep('''             ("cam_temporal", cfg.cam_temporal, TEMPORALS), ("rad_temporal", cfg.rad_temporal, TEMPORALS),''',
    '''             ("cam_temporal", cfg.cam_temporal, TEMPORALS), ("cam_buffer", cfg.cam_buffer, CAM_BUFFERS), ("rad_temporal", cfg.rad_temporal, TEMPORALS),''')
rep('''    if cfg.radar_kind == "delta_slots" and cfg.c_rad < n_channels():''', '''    if cfg.cam_buffer != "window" and cfg.cam_temporal != "window_memory":
        errs.append(f"cam_buffer={cfg.cam_buffer!r} buffers a frame for the window-memory block: it needs cam_temporal 'window_memory' (is {cfg.cam_temporal!r})")
    if cfg.radar_kind == "delta_slots" and cfg.c_rad < n_channels():''')
open(p, "w", encoding="utf-8", newline="").write(s)
