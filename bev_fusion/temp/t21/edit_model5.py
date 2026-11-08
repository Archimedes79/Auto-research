p = "training/model.py"
s = open(p, encoding="utf-8").read()

def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a)
    s = s.replace(a, b)

rep('''        K = 1). 'recurrent': `_unroll`. -> (camera BEV, cam state, motion-block levels or None, the new state["buf"])"""''',
    '''        K = 1). 'recurrent': `_unroll`. radar None: no motion block (`advance`). -> (camera BEV, cam state, motion-block levels or None,
        the new state["buf"])"""''')
rep('''        if self.motion_block is not None:
            v_levels = self.motion_block(*block_inputs(pair, pair_steps, radar, self.grid))
        if mode == "pair":''', '''        if self.motion_block is not None and radar is not None:
            v_levels = self.motion_block(*block_inputs(pair, pair_steps, radar, self.grid))
        if mode == "pair":''')
rep('''    def _refine_train(self, obj, feat, gts, dt):''', '''    def advance(self, batch, state=None, step=None):
        """stream mode, a frame that needs no output (cam_buffer 'pair' / 'recurrent'): lift + the camera temporal block only - no motion block,
        radar, trunk or heads - and the new state, the one `forward` would return for the cam / buf entries; forward() on the next frame
        continues from it. One frame, `step` as in forward. Needs rad_temporal 'none' (its state would go stale); a mirror-augmented batch is
        refused (training only)."""
        if self.cfg.cam_buffer == "window" or self.rad_temporal is not None or (batch.get("flip") is not None and bool(batch["flip"].any())):
            raise ValueError("advance: cam_buffer 'pair' / 'recurrent', rad_temporal 'none' and no flipped sample")
        batch = Precompute.from_model(self)(dict(batch), ("vis_cam",))
        frames = self.camera_bevs(batch)
        steps = [[tuple(torch.as_tensor(t, dtype=frames.dtype, device=frames.device) for t in s) for s in f] for f in batch["steps"]]
        _, cam_state, _, buf = self._camera_buffered(frames, steps, None, state or {}, step, batch["vis_cam"].to(frames))
        return dict(state or {}, cam=cam_state, buf=buf)

    def _refine_train(self, obj, feat, gts, dt):''')
open(p, "w", encoding="utf-8", newline="").write(s)
