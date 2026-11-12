p = "../../training/data/lane_gt_data.py"
s = open(p, encoding="utf-8").read()
s = s.replace("import qends as QEND  # noqa: E402\n", "import qends as QEND  # noqa: E402\nimport qjunction as QJ  # noqa: E402\n")
s = s.replace('ZONE_NAMES = (', 'JUNCTION_NAMES = ("jz", "jheat", "jdir", "jpaint")                                                                           # E285: the junction maps of `qjunction.junction_targets` (only with junction=True)\nZONE_NAMES = (', 1)
s = s.replace("    DEFAULT_ENDS = False  ", "    DEFAULT_JUNCTION = False                                       # E286: set by `jmaps_train.py`: the batch also carries the junction maps (zone, entry / exit heat + heading, paint) and the ends\n    DEFAULT_ENDS = False  ", 1)
s = s.replace("sparse=None, zone=None, ends=None):\n        self.bd", "sparse=None, zone=None, ends=None, junction=None):\n        self.bd", 1)
s = s.replace("        self.ends = QData.DEFAULT_ENDS if ends is None else bool(ends)\n", "        self.junction = QData.DEFAULT_JUNCTION if junction is None else bool(junction)\n        self.ends = (QData.DEFAULT_ENDS if ends is None else bool(ends)) or self.junction\n        self.paint_missing = 0\n", 1)
old = "        if self.pixels:\n            import qraster as QR"
new = """        if self.junction:                                                                              # E285: the junction GT of the frame (zone, ends heat + heading, painted stop lines / crossings)
            try:
                import junction_paint as JP
                paint = JP.load(token)
            except FileNotFoundError:
                paint, self.paint_missing = None, self.paint_missing + 1
            jt, _ = QJ.junction_targets(g, paint)
            out.update({n: torch.from_numpy(np.ascontiguousarray(v)) for n, v in jt.items()})
        if self.pixels:
            import qraster as QR"""
assert old in s
s = s.replace(old, new, 1)
s = s.replace('for n in ("lines_mask", "lines_dist", "events1", "road2") + ZONE_NAMES:', 'for n in ("lines_mask", "lines_dist", "events1", "road2") + ZONE_NAMES + JUNCTION_NAMES:', 1)
s = s.replace("sparse=None, zone=None, ends=None):\n    ds = QData(split, ks=ks, map_mode=map_mode, pixels=pixels, strict=strict, sparse=sparse, zone=zone, ends=ends)", "sparse=None, zone=None, ends=None, junction=None):\n    ds = QData(split, ks=ks, map_mode=map_mode, pixels=pixels, strict=strict, sparse=sparse, zone=zone, ends=ends, junction=junction)", 1)
open(p, "w", encoding="utf-8").write(s)
print("patched")
