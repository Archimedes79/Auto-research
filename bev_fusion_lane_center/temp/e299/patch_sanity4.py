p = "temp/e299/heat_replay.py"
s = open(p, encoding="utf-8").read()
a = s.index("                    dv_ = np.asarray(e[\"valid\"]).reshape(-1)")
b = s.index("    print(\"baseline sanity against ends_val.pkl:\"")
new = '''                    dv_ = np.asarray(e["valid"]).reshape(-1)
                    A_, B_ = pos[val], np.asarray(e["kp_pos"])[dv_]
                    same_set = False
                    if len(A_) and len(B_):                                            # as SETS: saturated peaks tie in the top-K, so the ORDER and, at the K-cut, the members may differ between CPU and GPU
                        dd = np.linalg.norm(A_[:, None] - B_[None], axis=-1)
                        same_set = len(A_) == len(B_) and dd.min(1).max() <= 0.05 and dd.min(0).max() <= 0.05
                        if same_set:
                            sane["max_dpos"] = max(sane["max_dpos"], float(dd.min(1).max()))
                    sane["same_set"] = sane.get("same_set", 0) + int(same_set)
                    if not same_set:                                                   # a differing frame must have a tie at the K-cut of some kind: more than K peaks and the K-th and K+1-th score equal
                        tie = False
                        for kk in (0, 1):
                            pk = p[i][kk]
                            mx = F.max_pool2d(pk[None, None], 3, 1, 1)[0, 0]
                            sc = torch.where((pk >= mx) & (pk > PEAK_MIN), pk, torch.zeros_like(pk)).flatten().sort(descending=True).values
                            tie = tie or bool((sc[K] > 0) and abs(float(sc[K - 1] - sc[K])) < 1e-6)
                        sane.setdefault("differing_frames", []).append((t, tie))
'''
s = s[:a] + new + s[b:]
a = s.index("    extra = sane.get(\"only_dump_dp\", [])")
b = s.index("    maps_done = None") if "    maps_done = None" in s else s.index("    rng = np.random.RandomState(1234)")
new2 = '''    sane["unequal_frames"] = sane["frames"] - sane["same_set"]
    sane["differing_without_cut_tie"] = sum(1 for _, tie in sane.get("differing_frames", []) if not tie)
    sane["differing_frames"] = len(sane.get("differing_frames", []))
    print("baseline sanity against ends_val.pkl:", sane, flush=True)
    assert sane["frames"] and sane["same_set"] >= 0.98 * sane["frames"] and sane["max_dpos"] <= 0.05 and sane["differing_without_cut_tie"] == 0, "the baseline extraction does not reproduce the stored keypoints: stop"
'''
s = s[:a] + new2 + s[b:]
open(p, "w", encoding="utf-8").write(s)
