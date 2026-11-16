p = "temp/jr/build_junction_report.py"
s = open(p, encoding="utf-8").read()
old = 'E307 = {s: J(f"runs/e307_n_s{s}/jmaps_val.json") for s in (0, 1)}'
assert old in s
s = s.replace(old, old + '\nE313 = {s: J(f"runs/e313_s_s{s}/jmaps_val.json") for s in (0, 1)}                                   # E313: heat sigma 0.7 m', 1)
old = '    return tbl(["arm", "entry R", "entry P", "exit R", "exit P", "kp_f1", "zone IoU", "core IoU", "stop line P / R"], rows, num=range(1, 9))'
assert old in s
add = '''    for s_ in (0, 1):
        r = E313[s_]
        if r:
            k = r["keypoints"]
            rows.append([f"<b>E313: heat sigma 0.7 m</b> seed {s_}", f3(k["kp_ours_tol3.0_noang_ent"]["recall"]), f3(k["kp_ours_tol3.0_noang_ent"]["precision"]), f3(k["kp_ours_tol3.0_noang_ext"]["recall"]), f3(k["kp_ours_tol3.0_noang_ext"]["precision"]),
                         f"<b>{f3(kp_f1(r))}</b>", f3(r["zone_iou"]), f3(r["core_iou"]), f"{f2(r['stop_prec'])} / {f2(r['stop_rec'])}"])
'''
s = s.replace(old, add + old, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
