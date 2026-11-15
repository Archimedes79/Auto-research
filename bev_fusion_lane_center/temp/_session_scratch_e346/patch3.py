p = "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e346/e346_lib.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a
    s = s.replace(a, b)
rep('''    segments of the lane GT), connectors with a collision cell along them (a cell of theirs in which their piece collides), the same two numbers for the connector segments inside an intersection]"""
    cells, keys, aff, pts, tans = {}, [], [], [], []''', '''    segments of the lane GT), connectors with a collision cell along them (a cell of theirs in which their piece collides), the same two numbers for the connector segments inside an intersection,
    connectors with a CROSSING cell along them (the part of the share that does not depend on COLL_DIST)]"""
    cells, keys, aff, affx, pts, tans = {}, [], [], [], [], []''')
rep('''        aff.append(set())
        c = QT.cell_of(q)''', '''        aff.append(set())
        affx.append(set())
        c = QT.cell_of(q)''')
rep('''    out = np.zeros(8)
    out[0] = len(cells)''', '''    out = np.zeros(9)
    out[0] = len(cells)''')
rep('''                if bad:
                    aff[ia].add(key)
                    aff[ib].add(key)''', '''                if bad:
                    aff[ia].add(key)
                    aff[ib].add(key)
                if ang > COLL_ANG:
                    affx[ia].add(key)
                    affx[ib].add(key)''')
rep('''            hit = bool(aff[i] & set(keys[i][s["i0"]: s["i1"] + 1].tolist()))
            out[4] += 1
            out[5] += hit''', '''            along = set(keys[i][s["i0"]: s["i1"] + 1].tolist())
            hit = bool(aff[i] & along)
            out[4] += 1
            out[5] += hit
            out[8] += bool(affx[i] & along)''')
rep('''coll_inter_connectors_n=c[6], coll_inter_connector_share=c[7] / max(c[6], 1))''', '''coll_inter_connectors_n=c[6], coll_inter_connector_share=c[7] / max(c[6], 1),
             coll_connector_cross_share=c[8] / max(c[4], 1))''')
rep('''    out = np.zeros(8)
''' if False else '''    out["split"]''' if False else '''def _f1(p, r):''', '''def _f1(p, r):''')
open(p, "w", encoding="utf-8").write(s)

p = "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e346/e346_arm.py"
s = open(p, encoding="utf-8").read()
rep('''assert c.tolist() == [occupied, 1, 1, 1, 1, 1, 1, 1], (res, c)''', '''assert c.tolist() == [occupied, 1, 1, 1, 1, 1, 1, 1, 1], (res, c)''')
rep('''assert same[1] == cells and same[2] == 0 and same[3] == 0 and same[5] == 0, (res, same)
        assert side[1] == cells and side[2] == 0 and side[3] == cells and side[5] == 1, (res, side)''', '''assert same[1] == cells and same[2] == 0 and same[3] == 0 and same[5] == 0, (res, same)
        assert side[1] == cells and side[2] == 0 and side[3] == cells and side[5] == 1 and side[8] == 0, (res, side)''')
rep('''"coll_cell_share", "coll_connector_share", "coll_inter_connector_share")''', '''"coll_cell_share", "coll_connector_share", "coll_connector_cross_share", "coll_inter_connector_share")''')
rep('''            sh = cards[a]["metrics"]["coll_connector_share"]
            print(f"chain rule {a}: connectors with a collision cell {sh:.4f} (bar {CHAIN_MAX}): {'rejected' if sh > CHAIN_MAX else 'own card'}; raw / crossing share {cards[a]['metrics']['coll_raw_share']:.4f} / {cards[a]['metrics']['coll_crossing_share']:.4f}"
                  f" (E273: {'0.184 / 0.037' if a == 'l2' else '0.121 / 0.019'})")''', '''            m = cards[a]["metrics"]
            sh = m["coll_connector_share"]
            print(f"chain rule {a}: connectors with a collision cell {sh:.4f} (bar {CHAIN_MAX}): {'rejected' if sh > CHAIN_MAX else 'own card'}; of it by crossing alone {m['coll_connector_cross_share']:.4f}; "
                  f"raw / crossing share of occupied cells {m['coll_raw_share']:.4f} / {m['coll_crossing_share']:.4f} (E273: {'0.184 / 0.037' if a == 'l2' else '0.121 / 0.019'})")''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
