p = "training/grid/sections.py"
s = open(p, encoding="utf-8").read()
s = s.replace("        pl = pl + (F.binary_cross_entropy(p_same, same, reduction=\"none\") * w).sum() / w.sum()", "        bce = -(same * torch.log(p_same) + (1 - same) * torch.log(1 - p_same))\n        pl = pl + (bce * w).sum() / w.sum()")
open(p, "w", encoding="utf-8").write(s)
p = "training/data/lane_gt_data.py"
s = open(p, encoding="utf-8").read()
s = s.replace("def __init__(self, split=\"train\", ks=None, map_mode=\"train\", pixels=True, strict=False):", "def __init__(self, split=\"train\", ks=None, map_mode=\"train\", pixels=True, strict=False, dilate=2.0):")
s = s.replace("        self.map_mode, self.pixels = map_mode, pixels", "        self.map_mode, self.pixels, self.dilate = map_mode, pixels, dilate")
s = s.replace("pix = QR.pixel_targets(g)", "pix = QR.pixel_targets(g, dilate=self.dilate)")
s = s.replace("road2 (2, 50, 50)); the map", "road2 (2, 50, 50); the intersection polygons are grown by `dilate` = 2 m (Mirko: slightly enlarged; the GT itself is undilated: stop / exit events lie 1-3 m before the polygon edge)); the map")
open(p, "w", encoding="utf-8").write(s)
