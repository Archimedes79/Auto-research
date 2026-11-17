def rd(p):
    return open(p, encoding="utf-8").read()
def wr(p, s):
    open(p + ".new_s292", "w", encoding="utf-8", newline="\n").write(s)
def sub(s, a, b):
    assert s.count(a) == 1, a[:70]
    return s.replace(a, b)

s = rd("training/grid/jwire_head.py")
s = sub(s, "    def forward(self, feat, mask):\n        return self.l2(F.relu(self.l1(feat))).squeeze(-1).masked_fill(~mask, float(\"-inf\"))\n",
  "    def forward(self, feat, mask):\n        return self.l2(F.relu(self.l1(feat))).squeeze(-1).masked_fill(~mask, float(\"-inf\"))\n\n"
  "    @classmethod\n    def rule_like(cls):\n"
  "        \"\"\"the head with hand-set weights that imitates the nearest-piece rule exactly (logit = -relu(distance feature)): the twin in the shape of the head, the fixture of the read-out path (`jends_eval.py score --wire head`\n"
  "        with it must reproduce the card of record)\"\"\"\n"
  "        w = cls()\n        with torch.no_grad():\n            for p in w.parameters():\n                p.zero_()\n            w.l1.weight[0, 0] = 1.0\n            w.l2.weight[0, 0] = -1.0\n        return w\n")
s = sub(s, "        got = wire_choice_head(RuleWire(), en, at, pairs, kp)\n        assert got == JW.twin_choice(en, pairs, REACH), (got, JW.twin_choice(en, pairs, REACH))\n",
  "        got = wire_choice_head(RuleWire(), en, at, pairs, kp)\n        assert got == JW.twin_choice(en, pairs, REACH), (got, JW.twin_choice(en, pairs, REACH))\n"
  "        assert wire_choice_head(Wire.rule_like(), en, at, pairs, kp) == got, \"the hand-set rule-like head must choose what the twin chooses\"\n")
wr("training/grid/jwire_head.py", s)

d = rd("evaluation/jwire_data.py")
i0 = d.index("def calibrate_jitter(")
i1 = d.index("def _selftest():")
new = '''def calibrate_jitter(records, target_median, seed=0, iters=12):
    """the jitter scale (m) at which the MEDIAN keypoint error of the matched ends of `records` (pair end to GT end) reaches `target_median` (the VAL value): the in-sample guard of the card, matched on the error quantile
    (TRAIN keypoints are better than VAL ones, the networks were trained on them); the agreement of the rule with the oracle label cannot be matched at the same time (see the card), it is reported -> (scale, median error)"""
    def med(scale):
        err = []
        for r in records:
            j = make_jitter(scale, seed)
            for n in range(len(r["pairs"])):
                for side, sl in ((0, slice(0, 2)), (1, slice(4, 6))):
                    g = r["gts"][n, side]
                    if not np.isnan(g).any():
                        p, _ = j(r["pairs"][n, sl], np.array([0.0, 1.0]))
                        err.append(np.linalg.norm(p - g[:2]))
        return float(np.median(err))
    lo, hi = 0.0, 3.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if med(mid) < target_median:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    return s, med(s)


'''
wr("evaluation/jwire_data.py", d[:i0] + new + d[i1:])
print("ok")
