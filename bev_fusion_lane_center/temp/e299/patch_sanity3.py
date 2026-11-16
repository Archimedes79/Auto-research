p = "temp/e299/heat_replay.py"
s = open(p, encoding="utf-8").read()
a = s.index("                            pdump = np.asarray(e[\"kp_p\"]).reshape(-1)")
b = s.index("    print(\"baseline sanity against ends_val.pkl:\"")
new = '''                            pdump = np.asarray(e["kp_p"]).reshape(-1)
                            idx_d, idx_m = np.nonzero(dv_)[0], np.nonzero(val)[0]
                            for kd in range(len(idx_d)):
                                if dd[:, kd].min() > 0.05:
                                    sane.setdefault("only_dump_dp", []).append(round(abs(float(pdump[idx_d[kd]]) - PEAK_MIN), 5))
                            for km in range(len(idx_m)):
                                if dd[km].min() > 0.05:
                                    kk = 0 if idx_m[km] < len(val) // 2 else 1
                                    cx, cy = int((pos[idx_m[km], 0] + HALF) // RES), int((pos[idx_m[km], 1] + HALF) // RES)
                                    sane.setdefault("only_mine_dp", []).append(round(abs(float(p[i][kk, cx, cy]) - PEAK_MIN), 5))
'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
