p = "temp/jr/maneuver.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


sub("per = {v: [] for v in VARIANTS}", "per = {v: [] for v in VARIANTS}\nrows = []                                  # one row per GT maneuver: ends found by the keypoints, length, turn, realised per variant")
sub("""        real = [realised(P, T, tree, m[2], m[3]) >= NEED for m in mans]
""", """        real = [realised(P, T, tree, m[2], m[3]) >= NEED for m in mans]
        for m, r_ in zip(mans, real):
            if v == "plain":
                kp, kd = e["kp_pos"], e["kp_dir"]
                K = e["logit"].shape[0]
                ve = e["valid"][0] & (e["kp_p"][:K] >= 0.5)
                vx = e["valid"][1] & (e["kp_p"][K:] >= 0.5)

                def near(kpos, kdir, valid, pt, tn):
                    d_ = np.linalg.norm(kpos[valid] - pt, axis=1) if valid.any() else np.zeros(0)
                    if not len(d_):
                        return False
                    c_ = (kdir[valid] * tn).sum(1)
                    return bool(((d_ <= 4.0) & (c_ >= np.cos(np.radians(30)))).any())
                ang = float(np.degrees(np.arctan2(m[3][0][0] * m[3][-1][1] - m[3][0][1] * m[3][-1][0], (m[3][0] * m[3][-1]).sum())))
                rows.append(dict(token=f.token, length=float(np.linalg.norm(m[2][-1] - m[2][0])), run_len=float(len(m[2])), turn=ang, range=float(np.linalg.norm(m[2][0])),
                                 found_e=near(kp[:K], kd[:K], ve, m[2][0], m[3][0]), found_x=near(kp[K:], kd[K:], vx, m[2][-1], m[3][-1]), real={}))
            rows[-len(mans) + mans.index(m)]["real"][v] = bool(r_)
""")
sub("json.dump(dict(run=run, seed=seed, tol=TOL, ang=ANG, need=NEED, variants=res,", "json.dump(dict(run=run, seed=seed, tol=TOL, ang=ANG, need=NEED, rows=rows, variants=res,")
open(p, "w", encoding="utf-8").write(s)
print("ok")
