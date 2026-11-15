p = "evaluation/jmaps_eval.py"
s = open(p, encoding="utf-8").read()
old = """            for i in range(x.shape[0]):
                ent_g = b["ent"][i][b["ent_m"][i]].numpy()
                ext_g = b["ext"][i][b["ext_m"][i]].numpy()
"""
new = """            for i in range(x.shape[0]):
                ent_g = b["ent"][i][b["ent_m"][i]].numpy()
                ext_g = b["ext"][i][b["ext_m"][i]].numpy()
                pe, px = painted_of(b["token"][i], len(ent_g), len(ext_g))
"""
assert old in s
s = s.replace(old, new, 1)
old2 = """                                if use_ang and who == "ours" and tol == 3.0:
                                    acc.lst(key + "_" + typ, "ang", [p[2] for p in pairs])
"""
new2 = """                                if use_ang and who == "ours" and tol == 3.0:
                                    acc.lst(key + "_" + typ, "ang", [p[2] for p in pairs])
                                if tol == 3.0 and who in ("ours", "B") and use_ang == (who == "ours"):                       # recall on painted / unpainted ends (head with heading, B distance only, thr 0.3)
                                    fl = pe if typ == "ent" else px
                                    if fl is not None:
                                        hit = np.zeros(len(gt), bool)
                                        for _, j, _ in pairs:
                                            hit[j] = True
                                        acc.add(f"split_{who}_{typ}", gt_p=int(fl.sum()), hit_p=int(hit[fl].sum()), gt_u=int((~fl).sum()), hit_u=int(hit[~fl].sum()))
"""
assert old2 in s
s = s.replace(old2, new2, 1)
old3 = "def _dil(m):"
new3 = '''def painted_of(token, n_ent, n_ext):
    """(painted flags of the GT entries, of the GT exits) of a frame, aligned with the batch's GT ends; None when the paint sidecar or the junction GT is missing or does not line up"""
    try:
        import jgt as JG
        import junction_paint as JP
        import qjunction as QJ
        from GtGenerator import lane_gt
        jg = JG.junction_gt(lane_gt.load(token))
        try:
            paint = JP.load(token)
        except FileNotFoundError:
            return None, None
        pe, px = QJ.painted_flags(jg, paint)
        return (pe, px) if len(pe) == n_ent and len(px) == n_ext else (None, None)
    except Exception:
        return None, None


def _dil(m):'''
assert old3 in s
s = s.replace(old3, new3, 1)
old4 = '    res["keypoints"] = kp\n'
new4 = '''    res["keypoints"] = kp
    split = {}
    for key, d in acc.k.items():
        if key.startswith("split_"):
            split[key[6:]] = dict(painted_recall=d["hit_p"] / max(d["gt_p"], 1), unpainted_recall=d["hit_u"] / max(d["gt_u"], 1), painted_gt=int(d["gt_p"]), unpainted_gt=int(d["gt_u"]))
    res["painted_split"] = split
'''
assert old4 in s
s = s.replace(old4, new4, 1)
old5 = '    print("written", out_path)'
new5 = '''    for k, v in split.items():
        print(f"  recall at {k}: painted {v['painted_recall']:.3f} (n {v['painted_gt']}), unpainted {v['unpainted_recall']:.3f} (n {v['unpainted_gt']})")
    print("written", out_path)'''
assert old5 in s
s = s.replace(old5, new5, 1)
s = s.replace("  heading            median angle error (degrees) at the matched predictions\n", "  heading            median angle error (degrees) at the matched predictions\n  painted split      recall of the entries / exits that are PAINTED (a stop line of another type than the turn stop within 3 m, or a crossing within 4 m: `qjunction.painted_flags`) and of the rest (thr 0.3, 3 m)\n")
open(p, "w", encoding="utf-8").write(s)
print("ok")
