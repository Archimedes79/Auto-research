p = "evaluation/jmaps_eval.py"
s = open(p, encoding="utf-8").read()
# 1 painted_of also returns the city
old = '''def painted_of(token, n_ent, n_ext):
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
'''
new = '''def painted_of(token, n_ent, n_ext):
    """(painted flags of the GT entries, of the GT exits, city) of a frame, aligned with the batch's GT ends; (None, None, city) when the paint sidecar or the junction GT is missing or does not line up"""
    city = None
    try:
        import jgt as JG
        import junction_paint as JP
        import qjunction as QJ
        from GtGenerator import lane_gt
        g = lane_gt.load(token)
        city = "boston" if "boston" in str(g.get("location", "")) else "singapore"
        jg = JG.junction_gt(g)
        try:
            paint = JP.load(token)
        except FileNotFoundError:
            return None, None, city
        pe, px = QJ.painted_flags(jg, paint)
        return (pe, px, city) if len(pe) == n_ent and len(px) == n_ext else (None, None, city)
    except Exception:
        return None, None, city


def _range_class(E):
    d = np.linalg.norm(np.asarray(E)[:, :2], axis=1)
    return np.where(d < 20, 0, np.where(d < 35, 1, 2))
'''
assert old in s
s = s.replace(old, new, 1)
s = s.replace("                pe, px = painted_of(b[\"token\"][i], len(ent_g), len(ext_g))\n", "                pe, px, city = painted_of(b[\"token\"][i], len(ent_g), len(ext_g))\n                rc_e, rc_x = _range_class(ent_g) if len(ent_g) else np.zeros(0, int), _range_class(ext_g) if len(ext_g) else np.zeros(0, int)\n", 1)
old = '''                                    fl = pe if typ == "ent" else px
                                    if fl is not None:
                                        hit = np.zeros(len(gt), bool)
                                        for _, j, _ in pairs:
                                            hit[j] = True
                                        acc.add(f"split_{who}_{typ}", gt_p=int(fl.sum()), hit_p=int(hit[fl].sum()), gt_u=int((~fl).sum()), hit_u=int(hit[~fl].sum()))
'''
new = '''                                    fl = pe if typ == "ent" else px
                                    hit = np.zeros(len(gt), bool)
                                    for _, j, _ in pairs:
                                        hit[j] = True
                                    if fl is not None:
                                        acc.add(f"split_{who}_{typ}", gt_p=int(fl.sum()), hit_p=int(hit[fl].sum()), gt_u=int((~fl).sum()), hit_u=int(hit[~fl].sum()))
                                    rc = rc_e if typ == "ent" else rc_x
                                    for k, nm in enumerate(("lt20", "20to35", "gt35")):                                  # range from the ego, E300: the far ends are the hole
                                        acc.add(f"range_{who}_{typ}_{nm}", n=int((rc == k).sum()), hit=int(hit[rc == k].sum()))
                                    if city:
                                        acc.add(f"city_{who}_{typ}_{city}", n=len(gt), hit=int(hit.sum()))
'''
assert old in s
s = s.replace(old, new, 1)
old = '''    res["painted_split"] = split
'''
new = '''    res["painted_split"] = split
    other = {}
    for key, d in acc.k.items():
        if key.startswith("range_") or key.startswith("city_"):
            other[key] = dict(n=int(d["n"]), recall=d["hit"] / max(d["n"], 1))
    res["range_city_split"] = other
'''
assert old in s
s = s.replace(old, new, 1)
old = '''    for k, v in split.items():
        print(f"  recall at {k}: painted {v['painted_recall']:.3f} (n {v['painted_gt']}), unpainted {v['unpainted_recall']:.3f} (n {v['unpainted_gt']})")
'''
new = old + '''    for k, v in sorted(other.items()):
        print(f"  recall {k}: {v['recall']:.3f} (n {v['n']})")
'''
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
