p = "temp/jr/build_junction_report.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


# E307 maps loaded next to the E286 arms
sub('E286 = {(a, s): J(f"runs/e286_{a}_s{s}/jmaps_val.json") for a in ("j0", "j1", "j2", "jj") for s in (0, 1)}',
    'E286 = {(a, s): J(f"runs/e286_{a}_s{s}/jmaps_val.json") for a in ("j0", "j1", "j2", "jj") for s in (0, 1)}\nE307 = {s: J(f"runs/e307_n_s{s}/jmaps_val.json") for s in (0, 1)}                                   # E307: the dilated arm with heat sigma 1.0 m')
sub('''    return tbl(["arm", "entry R", "entry P", "exit R", "exit P", "kp_f1", "zone IoU", "core IoU", "stop line P / R"], rows, num=range(1, 9))''',
    '''    for s in (0, 1):
        r = E307[s]
        if r:
            k = r["keypoints"]
            rows.append([f"<b>E307: j1 with heat sigma 1.0 m</b> seed {s}", f3(k["kp_ours_tol3.0_noang_ent"]["recall"]), f3(k["kp_ours_tol3.0_noang_ent"]["precision"]), f3(k["kp_ours_tol3.0_noang_ext"]["recall"]), f3(k["kp_ours_tol3.0_noang_ext"]["precision"]),
                         f"<b>{f3(kp_f1(r))}</b>", f3(r["zone_iou"]), f3(r["core_iou"]), f"{f2(r['stop_prec'])} / {f2(r['stop_rec'])}"])
    return tbl(["arm", "entry R", "entry P", "exit R", "exit P", "kp_f1", "zone IoU", "core IoU", "stop line P / R"], rows, num=range(1, 9))''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
