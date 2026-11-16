p = "temp/e299/heat_replay.py"
s = open(p, encoding="utf-8").read()
a = s.index("                    dv_ = np.asarray(e[\"valid\"]).reshape(-1)")
b = s.index("    print(\"baseline sanity against ends_val.pkl:\"")
new = '''                    dv_ = np.asarray(e["valid"]).reshape(-1)
                    A_, B_ = pos[val], np.asarray(e["kp_pos"])[dv_]
                    if len(A_) and len(B_):                                            # as SETS: equal heat values (saturated peaks) tie in the top-K, so the ORDER may differ
                        dd = np.linalg.norm(A_[:, None] - B_[None], axis=-1)
                        if (val == dv_).all():
                            sane["max_dpos"] = max(sane["max_dpos"], float(dd.min(1).max()))
                        else:                                                          # a frame with a different number of valid keypoints: how far from the threshold are the keypoints that only one side has?
                            pdump = np.asarray(e["kp_p"]).reshape(-1)
                            for j in np.nonzero(dv_)[0]:
                                if np.linalg.norm(B_ - np.asarray(e["kp_pos"])[j], axis=1).min() > 0 and (dd[:, list(np.nonzero(dv_)[0]).index(j)].min() > 0.05):
                                    sane.setdefault("only_dump_dp", []).append(round(abs(float(pdump[j]) - PEAK_MIN), 5))
                            for ii in np.nonzero(val)[0]:
                                if dd[list(np.nonzero(val)[0]).index(ii)].min() > 0.05:
                                    kk = 0 if ii < len(val) // 2 else 1
                                    cx, cy = int((pos[ii, 0] + HALF) // RES), int((pos[ii, 1] + HALF) // RES)
                                    sane.setdefault("only_mine_dp", []).append(round(abs(float(p[i][kk, cx, cy]) - PEAK_MIN), 5))
'''
s = s[:a] + new + s[b:]
s = s.replace('''    assert sane["frames"] and sane["same_count"] >= 0.99 * sane["frames"] and sane["max_dpos"] < 0.05, "the baseline extraction does not reproduce the stored keypoints: stop"''',
'''    extra = sane.get("only_dump_dp", []) + sane.get("only_mine_dp", [])
    sane["unequal_frames"] = sane["frames"] - sane["same_count"]
    sane["max_extra_keypoint_distance_to_threshold"] = max(extra) if extra else 0.0
    assert sane["frames"] and sane["same_count"] >= 0.99 * sane["frames"] and sane["max_dpos"] < 0.05, "the baseline extraction does not reproduce the stored keypoints: stop"''')
open(p, "w", encoding="utf-8").write(s)
