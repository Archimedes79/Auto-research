def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:90])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, count))
p = "training/data/lane_gt_data.py"
sub(p, "    DEFAULT_ENDS = False  ", "    DEFAULT_PSCALE = False                                         # E303: set by `jends_train.py --scales`: the batch carries the fitted handle scales of the true pairs (a grid search, about 0.15 s per frame: NOT computed for the other runs)\n    DEFAULT_ENDS = False  ")
sub(p, "ends=None, junction=None):\n        ", "ends=None, junction=None, pscale=None):\n        ")
sub(p, "        self.paint_missing = 0\n", "        self.paint_missing = 0\n        self.pscale = QData.DEFAULT_PSCALE if pscale is None else bool(pscale)\n")
sub(p, "out[\"pscale\"] = torch.from_numpy(QJ.fit_pair_scales(jg)) if (jg is not None and len(jg[\"pairs\"]) == npair) else", "out[\"pscale\"] = torch.from_numpy(QJ.fit_pair_scales(jg)) if (self.pscale and jg is not None and len(jg[\"pairs\"]) == npair) else")
p = "training/train/jends_train.py"
sub(p, "    LD.QData.DEFAULT_JUNCTION = True\n    mp = ", "    LD.QData.DEFAULT_JUNCTION = True\n    LD.QData.DEFAULT_PSCALE = a.scales                                                                   # the fitted handle scales cost 0.15 s per frame: only for the scale head\n    mp = ")
print("ok")
