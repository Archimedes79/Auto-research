def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:80])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, count))


# ---- A: painted_flags: exits are painted by a crossing only (critic finding 7: the stop line of the arm's own entry side lies next to its exits)
sub("training/grid/qjunction.py", '''    def flag(E):
        out = np.zeros(len(E), bool)
        for i, e in enumerate(E):
            pt = Point(float(e[0]), float(e[1]))
            out[i] = any(p.distance(pt) <= stop_m for p in stops) or any(p.distance(pt) <= ped_m for p in peds)
        return out
    return flag(jg["ent"]), flag(jg["ext"])''', '''    def flag(E, use_stops):
        out = np.zeros(len(E), bool)
        for i, e in enumerate(E):
            pt = Point(float(e[0]), float(e[1]))
            out[i] = (use_stops and any(p.distance(pt) <= stop_m for p in stops)) or any(p.distance(pt) <= ped_m for p in peds)
        return out
    return flag(jg["ent"], True), flag(jg["ext"], False)''')
sub("training/grid/qjunction.py", "(turn stops lie inside the junction, they are waiting points, not entries) lies within `stop_m`\n    or a pedestrian crossing within `ped_m` (E285: only about half of the entries are painted: the rest has to be inferred from context)\"\"\"",
    "(turn stops lie inside the junction, they are waiting points, not entries) lies within `stop_m`\n    or a pedestrian crossing within `ped_m`; an EXIT is painted by a crossing only (critic 2026-11-11: the stop line of the arm's own entry side lies next to its exits) (E285: only part of the ends are painted: the rest has to be inferred from context)\"\"\"")

# ---- F: the paint sidecars are listed once at dataset init (a missing sidecar must not become an all-zero target silently)
sub("training/data/lane_gt_data.py", "        self.mm = None\n\n    def __len__(self):", "        if self.junction:                                                                                 # critic 2026-11-11: a missing sidecar counted in a worker copy never reached the main process\n            import junction_paint as JP\n            pdir = JP.paint_dir()\n            miss = [self.tokens[int(self.bd.rows[k])] for k in self.ks if not os.path.exists(os.path.join(pdir, self.tokens[int(self.bd.rows[k])] + \".pkl\"))]\n            assert not miss, f\"{len(miss)} frames have no paint sidecar in {pdir} (junction_paint.py make); first {miss[:3]}\"\n        self.mm = None\n\n    def __len__(self):")

# ---- B: jmaps_eval: narrow excepts, missing == 0, lane GT signature, threshold sweep, crowded ends
p = "evaluation/jmaps_eval.py"
s = open(p, encoding="utf-8").read()
a = s.index("def painted_of(")
b = s.index("def _range_class")
new_painted = '''def painted_of(token, n_ent, n_ext):
    """(painted flags of the GT entries, of the GT exits, city) of a frame, aligned with the batch's GT ends; (None, None, city) when the paint sidecar is missing or the junction GT does not line up with the batch (counted by the caller)"""
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


def crowded_of(G, tol=4.0, ang=30.0):
    """(n,) bool: a GT end with another end of the same kind within tol metres and ang degrees of heading (one keypoint can serve only one of them: the strict matching is harder there)"""
    G = np.asarray(G, float)
    out = np.zeros(len(G), bool)
    for j in range(len(G)):
        d = np.linalg.norm(G[:, :2] - G[j, :2], axis=1)
        a = np.degrees(np.arccos(np.clip(G[:, 2] * G[j, 2] + G[:, 3] * G[j, 3], -1, 1)))
        m = (d <= tol) & (a <= ang)
        m[j] = False
        out[j] = bool(m.any())
    return out


THR_SWEEP = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


'''
s = s[:a] + new_painted + s[b:]
# keypoint sweep per batch
s = s.replace('            kps = {"ours": JM.keypoints(out, thr=0.3), "ours5": JM.keypoints(out, thr=0.5), "B": JM.keypoints(fake, thr=0.3), "B5": JM.keypoints(fake, thr=0.5)}\n',
              '            kps = {"ours": JM.keypoints(out, thr=0.3), "ours5": JM.keypoints(out, thr=0.5), "B": JM.keypoints(fake, thr=0.3), "B5": JM.keypoints(fake, thr=0.5)}\n            sweep = {t: (JM.keypoints(out, thr=t), JM.keypoints(fake, thr=t)) for t in THR_SWEEP}                                   # critic 2026-11-11: both maps at the same fixed threshold is not a fair comparison of differently calibrated maps\n')
s = s.replace('                pe, px, city = painted_of(b["token"][i], len(ent_g), len(ext_g))\n', '                pe, px, city = painted_of(b["token"][i], len(ent_g), len(ext_g))\n                if pe is None and len(ent_g):\n                    acc.add("painted_unavailable", frames=1)\n                cr_e, cr_x = crowded_of(ent_g) if len(ent_g) else np.zeros(0, bool), crowded_of(ext_g) if len(ext_g) else np.zeros(0, bool)\n                for t, (ko, kb) in sweep.items():\n                    for typ, gt in (("ent", ent_g), ("ext", ext_g)):\n                        for who, kk in (("ours", ko), ("B", kb)):\n                            pairs_, npred_, ngt_ = match(kk[i][typ].numpy(), gt, 3.0, False)\n                            acc.add(f"sweep_{who}_{t}_{typ}", tp=len(pairs_), pred=npred_, gt=ngt_)\n')
s = s.replace('                                    rc = rc_e if typ == "ent" else rc_x\n', '                                    cr = cr_e if typ == "ent" else cr_x\n                                    acc.add(f"crowd_{who}_{typ}", gt_c=int(cr.sum()), hit_c=int(hit[cr].sum()), gt_n=int((~cr).sum()), hit_n=int(hit[~cr].sum()))\n                                    rc = rc_e if typ == "ent" else rc_x\n')
s = s.replace('    ds, dl = LD.loader("val_clean", bs=8, workers=a.workers, shuffle=False, map_mode="exact", pixels=False)\n    acc = Acc()', '    ds, dl = LD.loader("val_clean", bs=8, workers=a.workers, shuffle=False, map_mode="exact", pixels=False)\n    assert ds.missing == 0, f"{ds.missing} VAL frames have no stored lane GT"\n    acc = Acc()')
s = s.replace('    res = dict(run=a.run, base=base_path, frames=n, args=args)\n', '    from GtGenerator import lane_gt\n    res = dict(run=a.run, base=base_path, frames=n, args=args, lane_gt_signature=lane_gt.signature(), painted_unavailable_frames=int(acc.k.get("painted_unavailable", {}).get("frames", 0)))\n')
old = '    res["range_city_split"] = other\n'
new = '''    res["range_city_split"] = other
    crowd = {}
    for key, d in acc.k.items():
        if key.startswith("crowd_"):
            crowd[key[6:]] = dict(crowded_recall=d["hit_c"] / max(d["gt_c"], 1), crowded_n=int(d["gt_c"]), isolated_recall=d["hit_n"] / max(d["gt_n"], 1), isolated_n=int(d["gt_n"]))
    res["crowded_split"] = crowd
    sweep_res = {}
    for who in ("ours", "B"):
        sweep_res[who] = {}
        for typ in ("ent", "ext"):
            rows = []
            for t in THR_SWEEP:
                d = acc.k[f"sweep_{who}_{t}_{typ}"]
                r_, p_ = d["tp"] / max(d["gt"], 1), d["tp"] / max(d["pred"], 1)
                rows.append(dict(thr=t, recall=r_, precision=p_, f1=2 * r_ * p_ / max(r_ + p_, 1e-9)))
            sweep_res[who][typ] = dict(rows=rows, best_f1=max(r["f1"] for r in rows), best_thr=max(rows, key=lambda r: r["f1"])["thr"])
    res["thr_sweep"] = sweep_res
    res["kp_f1_best_threshold"] = {who: (sweep_res[who]["ent"]["best_f1"] + sweep_res[who]["ext"]["best_f1"]) / 2 for who in sweep_res}
'''
assert old in s
s = s.replace(old, new, 1)
s = s.replace('    for k, v in sorted(other.items()):\n', '    print("  best F1 over the thresholds (distance only, 3 m): ours %.3f (entry thr %s, exit thr %s), B %.3f (%s, %s)" % (res["kp_f1_best_threshold"]["ours"], sweep_res["ours"]["ent"]["best_thr"], sweep_res["ours"]["ext"]["best_thr"], res["kp_f1_best_threshold"]["B"], sweep_res["B"]["ent"]["best_thr"], sweep_res["B"]["ext"]["best_thr"]))\n    for k, v in sorted(crowd.items()):\n        print(f"  recall {k}: crowded {v[\'crowded_recall\']:.3f} (n {v[\'crowded_n\']}), isolated {v[\'isolated_recall\']:.3f} (n {v[\'isolated_n\']})")\n    for k, v in sorted(other.items()):\n', 1)
open(p, "w", encoding="utf-8").write(s)

# ---- G: read-out spec validation
sub("evaluation/qgraph_readout.py", "    if not spec.thr > 0.0:", "    if spec.kind == \"plain\" and spec.exempt:\n        raise ValueError(f\"read-out spec {text!r}: x (protect the special types) is for close / mnms, not plain\")\n    if spec.sigma is not None and not spec.sigma > 0.0:\n        raise ValueError(f\"read-out spec {text!r}: sigma must be above 0\")\n    if not spec.thr > 0.0:")
print("ok")
