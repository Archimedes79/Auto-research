import os
p = os.path.join(os.environ["BFLC_ROOT"], "temp/_session_scratch_d113/work3/make_pins.py")
s = open(p, encoding="utf-8").read()
s = s.replace('("shift_min", "shift_max", "yaw_std_deg", "exact_samples", "own_share_error", "own_share_S")', '("shift_min", "shift_max", "yaw_std_deg", "exact_samples", "own_share_error", "own_share_S", "connector_share_cut", "connector_share_uncut", "connector_samples")')
s = s.replace('''txt = json.dumps(pins, indent=1)''', '''for k in ("train_frames", "val_test_frames"):
    pins["check"][f"train_guard.{k}"] = chk["train_guard"][k]
txt = json.dumps(pins, indent=1)''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
