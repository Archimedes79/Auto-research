p = "evaluation/jdup.py"
s = open(p, encoding="utf-8").read()
a = s.index("Same counting as"); b = s.index('"""\nimport argparse')
new = '''Same counting as `sparse_lanes/topo_metrics.dup_counts` (dense points every 0.5 m; a point is a duplicate when another piece that is neither its graph neighbour nor its sibling runs parallel (|cos| > 0.9) within 0.75 m; the first and last
4 m of a piece never count), but every duplicate point is attributed, for six read-outs (all on B's delivered graph with the hard step `close:1.5@0.5`, connectors at p >= 0.1, snapped to the delivered pieces):
  c  E312 snapped (keypoint ends, the snap's geometry)      u  E312 snap-nogeom (same curves as c, keypoint geometry)      k  E320 own keypoints snapped      kn  E320 own keypoints snap-nogeom
  a  E320 PRIMARY (attached ends, snap-nogeom)              l  a third run read out like c (--l-run, e.g. E316)
  by kind        own piece / partner piece = connector (C) or B piece (B): CC (no B partner), CB (a B partner; `conn_dup_with_c_partner` of them also have a connector partner), BC, BB; for CB the arc length from the nearest end of the
                 connector (4 - 8, 8 - 16, 16+ m) and the kind of the B piece (section / link)
  by truth       a connector is TRUE when both of its ends match the two ends of one GT pair (strict one-to-one matching of `jwiring.match`, 4 m, 30 degrees), FALSE otherwise (wrong pairs, unmatched or duplicate keypoints, unannotated lanes
                 are not told apart): curves, curve points and duplicate points of each
  by probability the pair probability bins 0.1 - 0.3 / 0.3 - 0.5 / 0.5 - 0.8 / 0.8+ : curves, share true, duplicate share of the curve points
The totals reproduce the dup_frac of the cards (c 0.1747 / 0.1557, a 0.1931 / 0.1651 on VAL). Needs the GT ends of the VAL frames (`jwiring.py gt-ends`).
'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s); print("ok")
