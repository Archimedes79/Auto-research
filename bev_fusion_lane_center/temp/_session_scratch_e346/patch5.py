p = "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e346/e346_lib.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a
    s = s.replace(a, b)
rep('''# ------------------------------------------------------------------------------------------------------------------------------ one frame
def frame_record''', '''def old_as_pieces(gt):
    """the old graph GT (pts, kind) in the shape of a lane GT for collision_counts: polylines every 0.5 m, one segment of the piece's kind (E273's 18.4 % / 3.7 % were counted on these pieces)"""
    pieces = []
    for p, k in zip(gt["pts"], gt["kind"]):
        q = GE.resample(np.asarray(p, float), 0.5)
        pieces.append(dict(pts=q, segments=[dict(kind=k, i0=0, i1=len(q) - 1, in_intersection=False)]))
    return dict(pieces=pieces)


# ------------------------------------------------------------------------------------------------------------------------------ one frame
def frame_record''')
rep('''out=output_counts(nt), coll=collision_counts(lane), time=''', '''out=output_counts(nt), coll=collision_counts(lane), coll_old=collision_counts(old_as_pieces(gt)), time=''')
rep('''        out["supp"], out["out"], out["coll"] = (sum(r[k] for r in recs) for k in ("supp", "out", "coll"))''', '''        out["supp"], out["out"], out["coll"], out["coll_old"] = (sum(r[k] for r in recs) for k in ("supp", "out", "coll", "coll_old"))''')
rep('''    c = pl["coll"]
    m.update(coll_occupied_cells_per_frame=c[0] / n, coll_raw_share=c[1] / max(c[0], 1), coll_crossing_share=c[2] / max(c[0], 1), coll_cell_share=c[3] / max(c[0], 1),
             coll_connectors_n=c[4], coll_connector_share=c[5] / max(c[4], 1), coll_inter_connectors_n=c[6], coll_inter_connector_share=c[7] / max(c[6], 1),
             coll_connector_cross_share=c[8] / max(c[4], 1))
    return {k: float(v) for k, v in m.items()}''', '''    for key, pre in (("coll", "coll"), ("coll_old", "coll_old")):                  # on the lane GT (the code's source) and on the old graph GT (E273's pieces)
        c = pl[key]
        m.update({f"{pre}_occupied_cells_per_frame": c[0] / n, f"{pre}_raw_share": c[1] / max(c[0], 1), f"{pre}_crossing_share": c[2] / max(c[0], 1), f"{pre}_cell_share": c[3] / max(c[0], 1),
                  f"{pre}_connectors_n": c[4], f"{pre}_connector_share": c[5] / max(c[4], 1), f"{pre}_inter_connectors_n": c[6], f"{pre}_inter_connector_share": c[7] / max(c[6], 1),
                  f"{pre}_connector_cross_share": c[8] / max(c[4], 1)})
    return {k: float(v) for k, v in m.items()}''')
open(p, "w", encoding="utf-8").write(s)

p = "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e346/e346_arm.py"
s = open(p, encoding="utf-8").read()
rep('''"coll_cell_share", "coll_connector_share", "coll_connector_cross_share", "coll_inter_connector_share")''', '''"coll_cell_share", "coll_connector_share", "coll_connector_cross_share", "coll_inter_connector_share", "coll_old_raw_share", "coll_old_crossing_share", "coll_old_connector_share",
         "coll_old_connector_cross_share")''')
rep('''            print(f"chain rule {a}: connectors with a collision cell {sh:.4f} (bar {CHAIN_MAX}): {'rejected' if sh > CHAIN_MAX else 'own card'}; of it by crossing alone {m['coll_connector_cross_share']:.4f}; "
                  f"raw / crossing share of occupied cells {m['coll_raw_share']:.4f} / {m['coll_crossing_share']:.4f} (E273: {'0.184 / 0.037' if a == 'l2' else '0.121 / 0.019'})")''', '''            print(f"chain rule {a}: connectors with a collision cell {sh:.4f} (bar {CHAIN_MAX}): {'rejected' if sh > CHAIN_MAX else 'own card'}; of it by crossing alone {m['coll_connector_cross_share']:.4f}; "
                  f"raw / crossing share of occupied cells {m['coll_raw_share']:.4f} / {m['coll_crossing_share']:.4f} on the lane GT, {m['coll_old_raw_share']:.4f} / {m['coll_old_crossing_share']:.4f} on the old graph GT "
                  f"(E273: {'0.184 / 0.037' if a == 'l2' else '0.121 / 0.019'})")''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
