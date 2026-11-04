p = "training/model.py"
s = open(p, encoding="utf-8").read()

def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a)
    s = s.replace(a, b)

rep('''        v_levels = None
        if self.motion_block is not None:                        # t15: the velocity field from two camera frames + radar (motion_block.py)
            v_levels = self.motion_block(*block_inputs(cam_frames, steps, batch["radar"], self.grid))
        cam, cam_state = self._branch_temporal(self.cam_temporal, cam_frames, steps, state.get("cam"), step, cam_obs)
''', '''        v_levels, buf = None, None
        if self.cfg.cam_buffer != "window":                      # t21: one buffered frame instead of the window (_camera_buffered)
            cam, cam_state, v_levels, buf = self._camera_buffered(cam_frames, steps, batch["radar"], state, step, cam_obs)
        else:
            if self.motion_block is not None:                    # t15: the velocity field from two camera frames + radar (motion_block.py)
                v_levels = self.motion_block(*block_inputs(cam_frames, steps, batch["radar"], self.grid))
            cam, cam_state = self._branch_temporal(self.cam_temporal, cam_frames, steps, state.get("cam"), step, cam_obs)
''')
rep('''        res = {"out": out, "state": {"cam": cam_state, "rad": rad_state}, "features": S}
''', '''        res = {"out": out, "state": {"cam": cam_state, "rad": rad_state}, "features": S}
        if buf is not None:
            res["state"]["buf"] = buf
''')
rep('''        since that call, scalars or per-sample (B,) tensors. Returns''', '''        since that call, scalars or per-sample (B,) tensors. cam_buffer 'pair' / 'recurrent' (t21): the state also holds "buf" (the previous
        frame's raw camera BEV, and its fused output); a batch of ONE frame is a stream step: the previous frame comes from state["buf"] (none:
        cold start) and `step` is the ego step since it, one triple or the per-sweep chain (a list of triples like batch["steps"][k], which
        is what the batch path uses); with 'recurrent' a batch of several frames continues from state["buf"] too when `step` is given (the
        trainer's keyframe gap step), 'pair' never. Returns''')
rep('''                (state["buf"] = raw + fused): each frame is lifted once and fused with the previous fused state (splat_window on it, its
                velocity head), the batch unrolls its F frames oldest -> newest from cold. The radar keeps its own 0.5 s point buffer
''', '''                (state["buf"] = raw + fused): each frame is lifted once and fused with the previous fused state (splat_window on it, its
                velocity head); a batch unrolls its F frames oldest -> newest, starting from the previous call's buffer (training: the
                previous keyframe; none: cold). Train / deploy: training chains up to 3 keyframes x F frames, a stream is unbounded. The motion
                block reads two RAW BEVs (current + the frame before it) in both modes. The radar keeps its own 0.5 s point buffer
''')
open(p, "w", encoding="utf-8", newline="").write(s)
