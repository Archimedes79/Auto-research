p = "evaluation/bev_selftest.py"
s = open(p, encoding="utf-8").read()
anchor = '@check("BEVSubject.ops: the GMACs of the cached chain'
assert s.count(anchor) == 1
new = '''@check("taps of the new modules: the hooks pick the delta-slot features and the lossless state out of the radar encoder's pair, the radar memory, the three motion levels (the finest "
       "last, any number of levels); a plain radar tensor gives no state (and is not mistaken for a pair)")
def _():
    import torch.nn as nn
    from evalsuite.torch_taps import TapRecorder

    class Radar(nn.Module):
        def __init__(self, pair):
            super().__init__()
            self.pair, self.net = pair, nn.Identity()

        def forward(self, det, with_state=True):
            f = torch.ones(2, 3, 4, 4) * 5
            return (f, torch.ones(2, 3, 4, 4) * 7) if self.pair else f

    class Memory(nn.Module):
        def forward(self, x):
            return x * 2, {"state": 0}, None

    class Motion(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def forward(self, x):
            return [torch.full((2, 2, 4, 4), float(i + 1)) for i in range(self.n)]

    class Net(nn.Module):
        def __init__(self, pair, levels):
            super().__init__()
            self.radar, self.rad_temporal, self.motion_block = Radar(pair), Memory(), Motion(levels)

        def forward(self, x):
            f = self.radar(x)
            self.rad_temporal(f[0] if isinstance(f, tuple) else f)
            self.motion_block(x)
    names = ["radar_feat", "radar_state", "radar_memory", "motion_l0", "motion_l1", "motion_l2", "radar_raster"]
    for pair, levels in ((True, 3), (False, 2)):
        net = Net(pair, levels)
        rec = TapRecorder(net, A.bev_hooks())
        rec.arm(names)
        net(torch.zeros(1))
        got = rec.read(0)
        assert float(got["radar_feat"].mean()) == 5.0 and float(got["radar_memory"].mean()) == 10.0, (pair, {k: float(v.mean()) for k, v in got.items()})
        assert ("radar_state" in got) == pair and (not pair or float(got["radar_state"].mean()) == 7.0), (pair, sorted(got))
        assert float(got["motion_l2"].mean()) == float(levels) and float(got["motion_l0"].mean()) == 1.0, (levels, {k: float(v.mean()) for k, v in got.items() if k.startswith("motion")})
        assert ("motion_l1" in got) and float(got["motion_l1"].mean()) == 2.0
        assert not rec.errors.get("radar_feat") and rec.errors.get("radar_state", 0) == (0 if pair else 1), rec.errors
        rec.close()
    taps = {t.name: t for t in A.bev_taps(type("G", (), {"y_min": -26.0, "y_max": 26.0, "x_min": 0.0, "x_max": 60.0})())}
    assert {"radar_state", "radar_memory", "motion_l0", "motion_l1", "motion_l2", "hurdle_mov", "radar_aux"} <= set(taps) and all(t.optional for t in taps.values() if t.name != "trunk")


'''
s = s.replace(anchor, new + anchor, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
