import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/jwire_check.py", [
("""    t.add_argument("--thr", type=float, default=0.1)
    v = sub.add_parser("live")
    v.add_argument("--seed", type=int, required=True)
    v.add_argument("--pair-run", required=True)""", """    t.add_argument("--thr", type=float, default=0.1)
    v = sub.add_parser("live")
    v.add_argument("--seed", type=int, required=True)
    v.add_argument("--pair-run", required=True)"""),
("""    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        _selftest()
        return 0
    if a.cmd == "twin":
        ok, diff = twin_check(a.run, a.seed, a.tag, a.thr)
        print(f"twin {a.tag} of {a.run} (seed {a.seed}, thr {a.thr}) against the stored snapped card: " + ("IDENTICAL (numbers, curves, wired ends, per-frame counts)" if ok else "DIFFERENT: " + "; ".join(diff)))
        return 0 if ok else 1
    ok, n, diff = live_check(a.seed, a.pair_run, a.max, a.workers, a.device)
    print(f"live pass of {a.pair_run} (seed {a.seed}) on {n} val_clean frames against the stored-dump records: " + ("ACCEPTED" if ok else "REFUSED") + (f"; frames that differ: {diff}" if diff else "; all fields equal"))
    return 0 if ok else 1""", """    for s in (t, v):
        s.add_argument("--read-out", default=None, choices=tuple(READ_OUT_ENDS), help="snap (default): the snapped read-out, keypoint ends; primary (E358): the end-to-end read-out, attached ends")
        s.add_argument("--ends", default=None, choices=tuple(READ_OUT_ENDS.values()), help="the end kind of the read-out: kp = snap (default), att = primary")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        _selftest()
        return 0
    read_out, ends = read_out_ends(a.read_out, a.ends)
    if a.cmd == "twin":
        ok, diff = twin_check(a.run, a.seed, a.tag, a.thr, read_out)
        print(f"twin {a.tag} of {a.run} (seed {a.seed}, thr {a.thr}) against the stored {read_out} card: " + ("IDENTICAL (numbers, curves, wired ends, per-frame counts)" if ok else "DIFFERENT: " + "; ".join(diff)))
        return 0 if ok else 1
    ok, n, diff = live_check(a.seed, a.pair_run, a.max, a.workers, a.device, ends)
    print(f"live pass of {a.pair_run} (seed {a.seed}, {ends} ends) on {n} val_clean frames against the stored-dump records: " + ("ACCEPTED" if ok else "REFUSED") + (f"; frames that differ: {diff}" if diff else "; all fields equal"))
    return 0 if ok else 1"""),
])
print("ok")
