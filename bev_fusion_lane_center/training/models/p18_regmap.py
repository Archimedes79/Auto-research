"""Loop 18 / R2.1: the map input of the map-conditioned builder after rough localisation and trust gating.
RegMapWrap(MapWrap): like p17_mapnet.MapWrap in evaluation mode, but the SD raster is drawn with the pose correction of the registration (p18_regis_eval.py output: causal accumulation per scene)
and multiplied by the map trust (hard: 0 / 1 at a threshold, soft: the trust itself, none: no gating). The correction is applied to the VECTOR map (re-render with the composed pose, as a navigation system
that corrects its own pose estimate would do), not by resampling the raster. Wrong maps get no pose correction (the estimate would be meaningless), only the gate.
Usage: imported by p17_eval.py (--reg)."""
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc_lib as AL  # noqa: E402
import p17_mapnet as MN  # noqa: E402
import p18_regis as RG  # noqa: E402
import sd_lib as SL  # noqa: E402


class RegMapWrap(MN.MapWrap):
    def __init__(self, base, part, P, regdir, tag, sigma=0.0, mode="exact", trust_mode="hard", thr=0.62, correct=True, **kw):
        super().__init__(base, part, P, train=False, mode=mode, sigma=sigma, **kw)
        cond = "wrong" if mode == "wrong" else f"s{sigma:g}"
        z = np.load(f"{regdir}/{tag}_{cond}.npz")
        self.zr = {k: z[k] for k in ("rows", "th_acc", "t_acc", "c_acc", "trust_acc", "trust")}
        self.ridx = {int(r): i for i, r in enumerate(self.zr["rows"])}
        self.trust_mode, self.thr, self.correct = trust_mode, thr, correct

    def raster(self, row, rng, mode, sigma):
        if mode == "none":
            return super().raster(row, rng, mode, sigma)
        i = self.ridx[int(row)]
        if mode == "exact" and sigma > 0 and self.correct and self.zr["c_acc"][i] > 1e-6:
            sd = self.sdl[row]
            if sd is None:
                return np.zeros((MN.N_SD, 200, 200), np.float32)
            scene = int(self.P["scene"][row])
            w, th_e = SL.scene_pose_error(scene, sigma)
            d_e = self.P["R"][row].T @ w
            r2 = np.random.RandomState((zlib.crc32(f"sd{scene}".encode()) + int(row)) % (2 ** 32))
            dg = SL.degrade_sd(sd, r2)
            Rn, dn = RG.corrected_pose(AL.rot2(th_e), d_e, float(self.zr["th_acc"][i]), self.zr["t_acc"][i])
            ras = SL.raster_sd(dg, Rn, dn).astype(np.float32)
        else:
            ras = super().raster(row, rng, mode, sigma)
        tr = float(self.zr["trust_acc"][i])
        if self.trust_mode == "hard":
            ras = ras * (1.0 if tr >= self.thr else 0.0)
        elif self.trust_mode == "soft":
            ras = ras * tr
        return ras
