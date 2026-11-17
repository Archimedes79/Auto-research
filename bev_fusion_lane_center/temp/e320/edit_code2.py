def edit(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)

edit("temp/e320/dev/qends_eval.pysrc", [
    ('''if (not snap and not chord and a.thr == 0.5) else''', '''if (not snap and not chord and not attach and a.thr == 0.5) else'''),
])
edit("temp/e320/dev/jends.pysrc", [
    ('''    if geom_w > 0 and "ppts" in batch:                                                                          # E317 / E318: the shape and the ends of the delivered connector
        lg, ig''', '''    if geom_w > 0:                                                                                              # E317 / E318 / E320: the shape and the ends of the delivered connector
        assert "ppts" in batch, "geom_w > 0 needs the GT connector polylines in the batch (`QData.DEFAULT_PPTS = True`, set by jends_train.py --geom-w): without them the whole geometry and attachment loss would be skipped silently"
        lg, ig'''),
    ('''so the curve starts / ends ON a node of B (a delivered piece end) by construction and falls back to the keypoint when nothing is in reach.''', '''a convex combination of the keypoint and the candidate nodes: when one weight dominates the curve starts / ends at that node, and it falls back to the keypoint when nothing is in reach. It is NOT guaranteed to be a delivered piece end (the candidates are the raw nodes with existence > 0.05, mid-section nodes of a lane included; the weighted mean lies inside their hull; the dustbin falls back to the keypoint), and the proximity wiring of the read-out (4 m / 45 degrees, `qends_eval.add_connectors`) still decides which pieces the connector is joined to.'''),
])
edit("temp/e320/dev/jends_train.pysrc", [
    ('''    def loss_fn(m, b):
        dev = next(m.parameters()).device''', '''    calls = [0]

    def loss_fn(m, b):
        dev = next(m.parameters()).device'''),
    ('''jends_loss(out, b, cross_w=a.cross_w, driven_w=a.driven_w, geom_w=a.geom_w)
''', '''jends_loss(out, b, cross_w=a.cross_w, driven_w=a.driven_w, geom_w=a.geom_w)
        calls[0] += 1
        if a.geom_w > 0 and calls[0] == 100:                                                              # critic 2026-11-12: a wrong pair alignment of the GT polylines shows as geom_n far below pos_pairs
            run.log(f"step 100 check of the geometry loss: geom_n {info.get('geom_n', 0.0):.0f} pairs with a GT polyline of pos_pairs {info.get('pos_pairs', 0.0):.0f} positives, geom_end_d {info.get('geom_end_d', float('nan')):.2f} m" + (f", att_acc {info['att_acc']:.2f}, att_ce {info['att_ce']:.2f}" if "att_ce" in info else ""))
'''),
])
edit("temp/e320/dev/jdiag.pysrc", [
    ('''    E = {r: dict(pos=[], ang=[]) for r in runs}
    n = 0
    for sid, (name, toks)''', '''    E = {r: dict(pos=[], ang=[], apos=[], aang=[]) for r in runs}
    n = 0
    for sid, (name, toks)'''),
    ('''                        E[r]["pos"].append(float(np.linalg.norm(f["kp_pos"][k] - gt[g][:2])))
                        E[r]["ang"].append(float(np.degrees(np.arccos(c))))''', '''                        E[r]["pos"].append(float(np.linalg.norm(f["kp_pos"][k] - gt[g][:2])))
                        E[r]["ang"].append(float(np.degrees(np.arccos(c))))
                        if "att_pos" in f:                                                                                     # E320: the ATTACHED end of the same keypoint against the same GT end
                            ca = float(np.clip(np.dot(f["att_dir"][k], gt[g][2:4]) / (np.linalg.norm(f["att_dir"][k]) * np.linalg.norm(gt[g][2:4]) + 1e-9), -1, 1))
                            E[r]["apos"].append(float(np.linalg.norm(f["att_pos"][k] - gt[g][:2])))
                            E[r]["aang"].append(float(np.degrees(np.arccos(ca))))'''),
    ('''        print(f"{r:12s} position error: mean {p.mean():.2f} m, median {np.median(p):.2f}, 90% {np.percentile(p, 90):.2f}; heading error: mean {a.mean():.1f} deg, median {np.median(a):.1f}, 90% {np.percentile(a, 90):.1f}")''', '''        print(f"{r:12s} position error: mean {p.mean():.2f} m, median {np.median(p):.2f}, 90% {np.percentile(p, 90):.2f}; heading error: mean {a.mean():.1f} deg, median {np.median(a):.1f}, 90% {np.percentile(a, 90):.1f}")
        if E[r]["apos"]:                                                                                                         # E320: the attached ends of the same matched keypoints
            p2, a2 = np.array(E[r]["apos"]), np.array(E[r]["aang"])
            print(f"{'  attached':12s} position error: mean {p2.mean():.2f} m, median {np.median(p2):.2f}, 90% {np.percentile(p2, 90):.2f}; heading error: mean {a2.mean():.1f} deg, median {np.median(a2):.1f}, 90% {np.percentile(a2, 90):.1f}")'''),
])
print("code edits ok")
