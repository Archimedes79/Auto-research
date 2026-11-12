p = "temp/jr/maneuver.py"
s = open(p, encoding="utf-8").read()
a = s.index("        for m, r_ in zip(mans, real):")
b = s.index("        near = P[(np.abs(P) < BOX).all(1)]")
new = '''        if v == "plain":
            base_idx = len(rows)
            kp, kd = e["kp_pos"], e["kp_dir"]
            K = e["logit"].shape[0]
            ve = e["valid"][0] & (e["kp_p"][:K] >= 0.5)
            vx = e["valid"][1] & (e["kp_p"][K:] >= 0.5)

            def near_kp(kpos, kdir, valid, pt, tn):
                if not valid.any():
                    return False
                d_ = np.linalg.norm(kpos[valid] - pt, axis=1)
                c_ = (kdir[valid] * tn).sum(1)
                return bool(((d_ <= 4.0) & (c_ >= np.cos(np.radians(30)))).any())
            for m in mans:
                ang = float(np.degrees(np.arctan2(m[3][0][0] * m[3][-1][1] - m[3][0][1] * m[3][-1][0], (m[3][0] * m[3][-1]).sum())))
                rows.append(dict(token=f.token, length=float(np.linalg.norm(m[2][-1] - m[2][0])), run_len=float(len(m[2])), turn=ang, range=float(np.linalg.norm(m[2][0])),
                                 found_e=near_kp(kp[:K], kd[:K], ve, m[2][0], m[3][0]), found_x=near_kp(kp[K:], kd[K:], vx, m[2][-1], m[3][-1]), real={}))
        for i_, r_ in enumerate(real):
            rows[base_idx + i_]["real"][v] = bool(r_)
'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
print("ok")
