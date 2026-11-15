def sub(path, pairs):
    s = open(path, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:70], s.count(a))
        s = s.replace(a, b)
    open(path + ".new_s292", "w", encoding="utf-8", newline="\n").write(s)

sub("training/grid/jwire_head.py", [
 ("    def forward(self, feat, mask):\n        return self.l2(F.relu(self.l1(feat))).squeeze(-1).masked_fill(~mask, float(\"-inf\"))\n",
  "    def forward(self, feat, mask):\n        return self.l2(F.relu(self.l1(feat))).squeeze(-1).masked_fill(~mask, float(\"-inf\"))\n\n"
  "    @classmethod\n    def rule_like(cls):\n"
  "        \"\"\"the head with hand-set weights that imitates the nearest-piece rule exactly (logit = -relu(distance feature)): the twin in the shape of the head, the fixture of the read-out path (`jends_eval.py score --wire head`\n"
  "        with it must reproduce the card of record)\"\"\"\n"
  "        w = cls()\n        with torch.no_grad():\n            for p in w.parameters():\n                p.zero_()\n            w.l1.weight[0, 0] = 1.0\n            w.l2.weight[0, 0] = -1.0\n        return w\n"),
 ("        got = wire_choice_head(RuleWire(), en, at, pairs, kp)\n        assert got == JW.twin_choice(en, pairs, REACH), (got, JW.twin_choice(en, pairs, REACH))\n",
  "        got = wire_choice_head(RuleWire(), en, at, pairs, kp)\n        assert got == JW.twin_choice(en, pairs, REACH), (got, JW.twin_choice(en, pairs, REACH))\n"
  "        assert wire_choice_head(Wire.rule_like(), en, at, pairs, kp) == got, \"the hand-set rule-like head must choose what the twin chooses\"\n"),
])

sub("evaluation/jwire_data.py", [
 ("def calibrate_jitter(records, target, seed=0, iters=10):\n", "def calibrate_jitter(records, target_median, seed=0, iters=12):\n"),
])
