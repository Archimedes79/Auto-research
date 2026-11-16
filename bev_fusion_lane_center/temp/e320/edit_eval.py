p = "temp/e320/dev/jends_eval.pysrc"
s = open(p, encoding="utf-8").read()
def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old, s.count(old))
    s = s.replace(old, new)
rep('''ntok=bool(a.get("ntok", False)), unc=bool(a.get("unc", False)))
    m.load_state_dict(ck["state"])''', '''ntok=bool(a.get("ntok", False)), unc=bool(a.get("unc", False)), attach=bool(a.get("attach", False)))
    m.load_state_dict(ck["state"])''')
rep('''                    fr["logsig"] = out["kp_logsig"][i].float().cpu().numpy()
''', '''                    fr["logsig"] = out["kp_logsig"][i].float().cpu().numpy()
                if "att_pos" in out:                                                                                                           # E320: the attached ends (position, heading, probability of an attachment) of every keypoint
                    fr.update(att_pos=out["att_pos"][i].float().cpu().numpy(), att_dir=out["att_dir"][i].float().cpu().numpy(), att_p=out["att_p"][i].float().cpu().numpy())
''')
rep('''snap_both=bool(getattr(a, "snap_both", False))))''', '''snap_both=bool(getattr(a, "snap_both", False)), attach=bool(getattr(a, "attach", False))))''')
rep('''            for k_ in ("--snap-all",''', '''            s.add_argument("--attach", action="store_true", help="E320: connectors drawn from the ATTACHED ends of the dump (att_pos / att_dir; no snap, the default wiring of every piece in reach), writes ends_card_attach_thr<t>.json")
            for k_ in ("--snap-all",''')
open(p, "w", encoding="utf-8").write(s)

p = "temp/e320/dev/qends_eval.pysrc"
s = open(p, encoding="utf-8").read()
rep('''        pairs = [(e["kp_pos"][i], e["kp_dir"][i], e["kp_pos"][k + j], e["kp_dir"][k + j]) + (tuple(e["scale"][i, j]) if e.get("scale") is not None and not a.chord else ()) for i, j in zip(ia, ib)]''', '''        qp, qd = (e["att_pos"], e["att_dir"]) if attach else (e["kp_pos"], e["kp_dir"])                                  # E320: the attached ends of the dump instead of the keypoints
        pairs = [(qp[i], qd[i], qp[k + j], qd[k + j]) + (tuple(e["scale"][i, j]) if e.get("scale") is not None and not a.chord else ()) for i, j in zip(ia, ib)]''')
rep('''    rec = dict(base=[], head=[])
    n_curves = n_wired = 0''', '''    rec = dict(base=[], head=[])
    attach = bool(getattr(a, "attach", False))
    assert not attach or ("att_pos" in next(iter(ends.values()))), f"{a.run}: the dump has no attached ends (train with --attach, then jends_eval.py dump)"
    n_curves = n_wired = 0''')
rep('''    out["snap"] = snap
''', '''    out["snap"] = snap
    out["attach"] = attach
''')
rep('''("snap" + "".join("_" + k[5:] for k in opts) + "_" if snap else "") + f"thr{a.thr}.json"''', '''("snap" + "".join("_" + k[5:] for k in opts) + "_" if snap else "") + ("attach_" if attach else "") + f"thr{a.thr}.json"''')
open(p, "w", encoding="utf-8").write(s)

p = "temp/e320/dev/jmaneuver.pysrc"
s = open(p, encoding="utf-8").read()
rep('''    thr = {v: float(v[1:-1] if v in snapped else v[1:]) for v in names if v in snapped or (v.startswith("p") and v[1:].replace(".", "", 1).isdigit())}''', '''    attached = {v for v in names if v.startswith("p") and v.endswith("a") and v[1:-1].replace(".", "", 1).isdigit()}                    # p0.1a = connectors at p >= 0.1 from the ATTACHED ends of the dump, no snap (E320)
    thr = {v: float(v[1:-1] if (v in snapped or v in attached) else v[1:]) for v in names if v in snapped or v in attached or (v.startswith("p") and v[1:].replace(".", "", 1).isdigit())}''')
rep('''                prs = [(e["kp_pos"][i], e["kp_dir"][i], e["kp_pos"][K + j], e["kp_dir"][K + j]) +''', '''                qp, qd = (e["att_pos"], e["att_dir"]) if v in attached else (e["kp_pos"], e["kp_dir"])
                prs = [(qp[i], qd[i], qp[K + j], qd[K + j]) +''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
