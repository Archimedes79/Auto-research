"""one-off: model.py = HEAD + only the temporal-memory obs_mass changes of this session (another session's uncommitted
refine hunks stay out of the commit). Writes temp/checks/model_mine/model.py."""
import os
import subprocess

head = subprocess.run(["git", "show", "HEAD:bev_fusion/training/model.py"], capture_output=True, text=True, check=True,
                      cwd="..").stdout
s = head


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new, 1)


rep("    from .lift_pd import Grid, LiftPD\n", "    from .lift_pd import Grid, LiftPD, camera_visibility\n")
rep("    from lift_pd import Grid, LiftPD\n", "    from lift_pd import Grid, LiftPD, camera_visibility\n")
rep('''    @staticmethod
    def _branch_temporal(block, bevs, steps, state, step):
        """bevs (B, F, C, X, Y) with frame 0 = current; steps[k]: list of (dt, v, omega) per-sample (B,) tensors from
        frame k to the current time (collate); one batched call, past frames oldest first."""
        if block is None:
            return bevs[:, 0], state
        past = [(bevs[:, k], steps[k]) for k in range(bevs.shape[1] - 1, 0, -1)]
        out, new_state, _ = block(bevs[:, 0], past, state, step)
        return out, new_state''', '''    def camera_visibility(self, batch, like):
        """(B, 1, X, Y): the cells the current camera frame observes (LiftPD.visibility per sample) - the obs_mass of the
        camera branch's world memory, so the empty BEV outside the field of view does not overwrite what the memory saw
        (before 2026-10-26 the write used obs_mass = 1 everywhere)."""
        hw = batch["hw"] if "hw" in batch else batch["img"].shape[-2:]
        vis = [camera_visibility(self.grid, self.cfg.z_levels, self.lift.depth_centres, batch["K"][b, 0],
                                 batch["T_cam_car"][b, 0], hw, batch.get("ida"), self.cfg.oversample)
               for b in range(batch["K"].shape[0])]
        return torch.stack(vis)[:, None].to(like)

    @staticmethod
    def _branch_temporal(block, bevs, steps, state, step, obs_mass=None):
        """bevs (B, F, C, X, Y) with frame 0 = current; steps[k]: list of (dt, v, omega) per-sample (B,) tensors from
        frame k to the current time (collate); one batched call, past frames oldest first; obs_mass (B, 1, X, Y) where
        the sensor observed (memory write mask) or None (everywhere)."""
        if block is None:
            return bevs[:, 0], state
        past = [(bevs[:, k], steps[k]) for k in range(bevs.shape[1] - 1, 0, -1)]
        out, new_state, _ = block(bevs[:, 0], past, state, step, obs_mass)
        return out, new_state''')
rep('''        cam, cam_state = self._branch_temporal(self.cam_temporal, cam_frames, steps, state.get("cam"), step)''',
    '''        cam_obs = self.camera_visibility(batch, cam_frames) if isinstance(self.cam_temporal, TemporalFusion) else None
        cam, cam_state = self._branch_temporal(self.cam_temporal, cam_frames, steps, state.get("cam"), step, cam_obs)''')
rep('''    assert res2["out"][0.0]["sem"].shape == o["sem"].shape and "cam_aux" not in res2["out"][0.0]
''', '''    assert res2["out"][0.0]["sem"].shape == o["sem"].shape and "cam_aux" not in res2["out"][0.0]
    # the camera memory writes only where the camera observes (obs_mass = LiftPD.visibility), nothing outside the view
    if isinstance(model.cam_temporal, TemporalFusion):
        with torch.no_grad():
            vis = model.camera_visibility(batch, o["sem"])                                  # (B, 1, X, Y)
            st = model(batch)["state"]["cam"]
            from temporal_fusion import read_memory as _rm
            _, conf = _rm(st["M"], st["W"], st["pose"], model.cam_temporal.ego_pts, vis.shape[-2], vis.shape[-1],
                          model.cam_temporal.mem_grid)
        assert 0.05 < float(vis.mean()) < 0.95, f"visibility covers part of the grid ({float(vis.mean()):.2f})"
        assert float(conf[vis < 0.5].max()) < 1e-4 and float(conf[vis > 0.5].mean()) > 0.1, \\
            "memory written inside the camera view only"
''')
os.makedirs("temp/checks/model_mine", exist_ok=True)
open("temp/checks/model_mine/model.py", "w", encoding="utf-8", newline="\n").write(s)
print("ok")
