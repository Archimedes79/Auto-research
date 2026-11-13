p = "centre_thickness.py"
s = open(p).read()
s = s.replace("acc = {z: dict(cells=0.0, gt_len=0.0, far075=0.0, far15=0.0, zone_cells=0.0, fill=0.0) for z in (\"near a junction\", \"away\")}",
              "acc = {z: dict(cells=0.0, gt_len=0.0, far075=0.0, far15=0.0, zone_cells=0.0, fill=0.0, ideal=0.0) for z in (\"near a junction\", \"away\")}")
s = s.replace("        a[\"fill\"] += float((on & (zc == zi)).sum())\n", "        a[\"fill\"] += float((on & (zc == zi)).sum())\n        a[\"ideal\"] += float(((d_gt <= 1.1774) & (zc == zi) & inbox).sum())          # a perfect prediction of the Gaussian target (sigma 1 m, heat >= 0.5 at 1.18 m)\n")
s = s.replace("cells=a[\"cells\"], gt_len_m=a[\"gt_len\"])", "ideal_fill=a[\"ideal\"] / max(a[\"zone_cells\"], 1), ideal_width_m=a[\"ideal\"] / max(a[\"gt_len\"], 1e-9), cells=a[\"cells\"], gt_len_m=a[\"gt_len\"])")
s = s.replace("fill of the zone {100 * v['fill']:.1f} %\")", "fill of the zone {100 * v['fill']:.1f} %   (a perfect prediction of the target: width {v['ideal_width_m']:.2f} m, fill {100 * v['ideal_fill']:.1f} %)\")")
open(p, "w").write(s)
