p = "training/train.py"
s = open(p, encoding="utf-8").read()
# imports
s = s.replace("    from .nuscenes_dataset import FrontRadarDataset, available_scenes, collate\n", "    from .nuscenes_dataset import FrontRadarDataset, available_scenes, collate\n    from .radar_encoder import aux_weight\n", 1)
s = s.replace("    from nuscenes_dataset import FrontRadarDataset, available_scenes, collate\n", "    from nuscenes_dataset import FrontRadarDataset, available_scenes, collate\n    from radar_encoder import aux_weight\n", 1)
# config
old = "    consistency_warmup: tuple = (0.2, 0.2)       # (hold, ramp) fractions: off for the first 20 %, then linear in\n"
new = old + ('''    radar_aux: float = 0.0                       # radar_kind "delta_slots": weight of the radar encoder's reconstruction SIDE loss at the start of
                                                 # the training (radar_encoder.aux_weight: floor + (start - floor) * 0.5 ** (progress / half)); it
                                                 # helps the first phase, the task losses are the goal; 0 = off (kept-fraction not computed either)
    radar_aux_floor: float = 0.05                # weight it decays to
    radar_aux_half: float = 0.2                  # half-life in fractions of the training
''')
assert s.count(old) == 1
s = s.replace(old, new, 1)
# the loss term + monitor
old = '''        if diag:
            groups = {k[len("group."):]: v for k, v in out.items() if k.startswith("group.") and torch.is_tensor(v) and v.requires_grad}'''
new = '''        side = res.get("radar_side")                                               # radar_kind "delta_slots" only
        if side is not None and cfg is not None and cfg.radar_aux > 0:
            enc = model.radar
            f_side, s_side = side["features"].float(), side["state"].float()
            if train_terms:                                                        # the SIDE loss: helps early, decays to a floor
                la = enc.aux_loss(f_side, s_side)
                out["radar_aux"] = la
                out["total"] = out["total"] + aux_weight(step_frac, cfg.radar_aux, cfg.radar_aux_floor, cfg.radar_aux_half) * la
            else:                                                                  # validation: how much of the detections the features hold
                out["radar_kept"] = enc.kept_fraction(batch["radar"], f_side.shape[0], features=f_side)[0]
        if diag:
            groups = {k[len("group."):]: v for k, v in out.items() if k.startswith("group.") and torch.is_tensor(v) and v.requires_grad}'''
assert s.count(old) == 1
s = s.replace(old, new, 1)
old = 'LOGGED_TERMS = ("objhead.vel@+0.0", "objhead.mov@+0.0", "motion@+0.0")'
new = 'LOGGED_TERMS = ("objhead.vel@+0.0", "objhead.mov@+0.0", "motion@+0.0", "radar_aux", "radar_kept")'
assert s.count(old) == 1
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
