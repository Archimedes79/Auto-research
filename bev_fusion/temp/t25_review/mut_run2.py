"""run tests/test_radar_recursive.py with ONE extra mutant patched in before it runs: python mut_run.py <mutant name>"""
import os
import runpy
import sys

import torch

ROOT = "/mnt/c/Auto-research/bev_fusion"
for sub in ("tests", "tests/support", "training", ".."):
    sys.path.insert(0, os.path.join(ROOT, sub))
import model as M  # noqa: E402
from precompute import Precompute  # noqa: E402

name = sys.argv[1]
B = M.BEVFusionModel
if name == "carry_block_output":
    orig = B._recur

    def recur(block, bevs, moves, state, fused, obs, residual=None):
        if residual is None:
            return orig(block, bevs, moves, state, fused, obs, residual)
        past, res = fused, None
        for j, k in enumerate(range(bevs.shape[1] - 1, -1, -1)):
            out, state, _ = block(bevs[:, k], [] if past is None else [(past, moves[j])], state, moves[j], obs[k])
            past = out
            res = bevs[:, k] + residual(out)
        return res, state
    B._recur = staticmethod(recur)
elif name == "radar_no_ego_motion":
    orig = B._recur

    def recur(block, bevs, moves, state, fused, obs, residual=None):
        if residual is not None:
            moves = [[] for _ in moves]
        return orig(block, bevs, moves, state, fused, obs, residual)
    B._recur = staticmethod(recur)
elif name == "side_from_oldest":
    orig = B._radar_frames

    def frames(self, batch, like):
        r = orig(self, batch, like)
        seq = batch["radar_seq"]
        r["side"] = self._encode_radar(seq[-1], like.shape[0], like.device)[1]
        return r
    B._radar_frames = frames
elif name == "side_none":
    orig = B._radar_frames

    def frames(self, batch, like):
        r = orig(self, batch, like)
        r["side"] = None
        return r
    B._radar_frames = frames
elif name == "rad_mix_default_init":
    orig_init = B.__init__

    def init(self, cfg, *a, **k):
        orig_init(self, cfg, *a, **k)
        if self.rad_mix is not None:
            self.rad_mix.reset_parameters()
    B.__init__ = init
elif name == "mask_from_newest":
    orig = B._radar_frames

    def frames(self, batch, like):
        r = orig(self, batch, like)
        r["obs"] = [r["obs"][0]] * len(r["obs"])
        return r
    B._radar_frames = frames
elif name == "residual_dropped_in_past":
    # the carried radar past is the encoder feature of the previous frame only (no block information carried): residual form kept for the output
    orig = B._recur

    def recur(block, bevs, moves, state, fused, obs, residual=None):
        if residual is None:
            return orig(block, bevs, moves, state, fused, obs, residual)
        for j, k in enumerate(range(bevs.shape[1] - 1, -1, -1)):
            prev = None if j == 0 and fused is None else (bevs[:, k + 1] if j > 0 else fused)
            out, state, _ = block(bevs[:, k], [] if prev is None else [(prev, moves[j])], state, moves[j], obs[k])
            fused = bevs[:, k] + residual(out)
        return fused, state
    B._recur = staticmethod(recur)
elif name == "radar_moves_negated":
    orig = B._recur

    def recur(block, bevs, moves, state, fused, obs, residual=None):
        if residual is not None:
            moves = [[(dt, -v, w) for (dt, v, w) in mv] for mv in moves]
        return orig(block, bevs, moves, state, fused, obs, residual)
    B._recur = staticmethod(recur)
elif name == "radar_moves_omega_negated":
    orig = B._recur

    def recur(block, bevs, moves, state, fused, obs, residual=None):
        if residual is not None:
            moves = [[(dt, v, -w) for (dt, v, w) in mv] for mv in moves]
        return orig(block, bevs, moves, state, fused, obs, residual)
    B._recur = staticmethod(recur)
else:
    raise SystemExit("unknown mutant " + name)
print("MUTANT", name, flush=True)
sys.argv = ["tests/test_radar_recursive.py"]
os.chdir(ROOT)
try:
    runpy.run_path(os.path.join(ROOT, "tests", "test_radar_recursive.py"), run_name="__main__")
except SystemExit as e:
    print("EXIT", e.code)
