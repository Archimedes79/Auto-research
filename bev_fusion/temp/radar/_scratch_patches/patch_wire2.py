p = "training/model.py"
s = open(p, encoding="utf-8").read()
# check_config: the pass-through encoder needs C >= its input channels
old = '''    if cfg.lane_branch and cfg.camera_off:'''
new = '''    if cfg.radar_kind == "delta_slots" and cfg.c_rad < n_channels():
        errs.append(f"radar_kind='delta_slots' needs c_rad >= {n_channels()} (one channel per input channel: the pass-through start); c_rad = {cfg.c_rad}")
    if cfg.lane_branch and cfg.camera_off:'''
assert s.count(old) == 1
s = s.replace(old, new, 1)
# forward: the radar encoder also returns its lossless raster (the side loss and the kept-fraction monitor read it)
old = '''        rad = (torch.zeros(cam.shape[0], self.cfg.c_rad, self.grid.nx, self.grid.ny, device=cam.device)
               if self.cfg.radar_kind == "off" else self.radar(batch["radar"], cam.shape[0]))'''
new = '''        radar_side = None
        if self.cfg.radar_kind == "off":
            rad = torch.zeros(cam.shape[0], self.cfg.c_rad, self.grid.nx, self.grid.ny, device=cam.device)
        elif self.cfg.radar_kind == "delta_slots":               # features + the lossless raster: the side loss (train.py) reads both
            rad, rstate = self.radar(batch["radar"], cam.shape[0], with_state=True)
            radar_side = {"features": rad, "state": rstate}
        else:
            rad = self.radar(batch["radar"], cam.shape[0])'''
assert s.count(old) == 1
s = s.replace(old, new, 1)
old = '''        return {"out": out, "state": {"cam": cam_state, "rad": rad_state}, "features": S}'''
new = '''        res = {"out": out, "state": {"cam": cam_state, "rad": rad_state}, "features": S}
        if radar_side is not None:
            res["radar_side"] = radar_side
        return res'''
assert s.count(old) == 1
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)

p = "training/interface.py"
s = open(p, encoding="utf-8").read()
old = '''OUTPUTS = {"out": "{dt: per-dt dict}"'''
new = '''OPTIONAL_OUTPUTS = {"radar_side": "radar_kind 'delta_slots': {'features': (B, C, X, Y) encoder output, 'state': (B, 61, X, Y) lossless slot raster}, "
                                  "read by the radar side loss (train.py) and the kept-fraction monitor"}
OUTPUTS = {"out": "{dt: per-dt dict}"'''
assert s.count(old) == 1
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
