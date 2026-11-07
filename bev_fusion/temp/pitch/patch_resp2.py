p = "evaluation/bev_response.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    i, kept, k = 0, [], 0
    for bs, _ in ann:
        o = []''', '''    from evaluation.bev_fusion_area import footprint_share
    shifted, n_sh = [], 0                                              # the detection of a neutral box nudged (<= 1.5 m) until at least half of it lies in the area: still on a neutral object
    for bs, area in ann:
        o = [dict(b, cls=None) for b in bs if b["cls"] == "care"]
        for b in bs:
            if b["cls"] == "care":
                continue
            cand = [dict(b, cx=b["cx"] + dx, cy=b["cy"] + dy) for dx in (-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5) for dy in (-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5) if math.hypot(dx, dy) <= 1.5]
            hit = [c for c in cand if footprint_share(c, area) >= 0.5 and in_front(c)]
            if hit:
                o.append(hit[0])
                n_sh += 1
        shifted.append(strip(o))
    m = run(shifted, "detf")
    lab = f"detf: the care boxes + {n_sh} detections of NEUTRAL boxes nudged up to 1.5 m into the area (still on a neutral object)"
    rows += [(lab, "n_det (neutral ones ignored)", m("n_det"), Nc, 0.0), (lab, "ap_all@2", m("ap_all@2"), 1.0, 1e-9)]
    i, kept, k = 0, [], 0
    for bs, _ in ann:
        o = []''', 1)
open(p, "w", encoding="utf-8").write(s)
