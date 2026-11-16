p = "temp/e299/heat_replay.py"
s = open(p, encoding="utf-8").read()
old = '''                    both = np.asarray(e["valid"]).reshape(-1) & val
                    if both.any():
                        sane["max_dpos"] = max(sane["max_dpos"], float(np.abs(pos[both] - np.asarray(e["kp_pos"])[both]).max()))'''
new = '''                    dv_ = np.asarray(e["valid"]).reshape(-1)
                    if val.any() and dv_.any():                                        # as sets: equal heat values (saturated peaks) tie in the top-K, so the ORDER of the keypoints may differ
                        A_, B_ = pos[val], np.asarray(e["kp_pos"])[dv_]
                        sane["max_dpos"] = max(sane["max_dpos"], float(np.linalg.norm(A_[:, None] - B_[None], axis=-1).min(1).max()))'''
assert old in s
open(p, "w", encoding="utf-8").write(s.replace(old, new))
