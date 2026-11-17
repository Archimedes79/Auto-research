def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:90])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, count))


p = "training/grid/jends.py"
sub(p, "Run the self-test: python jends.py\n\"\"\"", '''E308 `ntok`: B's node tokens (the frozen node head of the E273 baseline on the same pyramid, 64 channels at 2 m) are sampled at the keypoints and enter the keypoint token: the E284 end head reads them and ranks the pairs between true ends better (AUC 0.974) than the head on the maps' own features alone (0.930, E287 p2).
Run the self-test: python jends.py
"""''')
sub(p, "    def __init__(self, d=64, hid=64, k=K, order=True, evidence=True, f1=64, scales=False, capacity=False):\n        super().__init__()\n        self.k, self.order, self.evidence, self.scales, self.capacity = k, order, evidence, scales, capacity\n        self.kp = nn.Linear(f1 + 6, d)\n",
    "    def __init__(self, d=64, hid=64, k=K, order=True, evidence=True, f1=64, scales=False, capacity=False, ntok=0):\n        super().__init__()\n        self.k, self.order, self.evidence, self.scales, self.capacity, self.ntok = k, order, evidence, scales, capacity, ntok\n        self.kp = nn.Linear(f1 + 6 + ntok, d)\n")
sub(p, "    def forward(self, jout, P, gt=None):\n        j1, f1 = jout[\"j1\"].float(), jout[\"f1\"]", "    def forward(self, jout, P, gt=None, tok=None):\n        j1, f1 = jout[\"j1\"].float(), jout[\"f1\"]")
sub(p, "        t = self.kp(torch.cat([feat, pos / HALF, D, pr[..., None], kind[..., None]], -1))\n", "        if self.ntok:                                                                                          # E308: B's node tokens at the keypoints (bilinear, 2 m grid over the same 100 m)\n            feat = torch.cat([feat, _sample(tok.float(), pos)], -1)\n        t = self.kp(torch.cat([feat, pos / HALF, D, pr[..., None], kind[..., None]], -1))\n")
# self-test: the node-token input changes the keypoint token and every parameter gets a gradient
sub(p, "    # E302: the capacitated plan. Two entries", '''    # E308: with node tokens the pair logits depend on the tokens (and the kp layer takes them)
    mt = JEnds(d=d, order=True, evidence=False, ntok=64)
    tok_a, tok_b = torch.randn(B, 64, 50, 50), torch.randn(B, 64, 50, 50)
    la = mt(jout, P, gt=gt, tok=tok_a)["logit"]
    lb = mt(jout, P, gt=gt, tok=tok_b)["logit"]
    assert float((la - lb).abs().max()) > 1e-4, "the node tokens must reach the pair logits"
    loss, info = jends_loss(mt(jout, P, gt=gt, tok=tok_a), batch)
    loss.backward()
    assert all(p_.grad is not None for p_ in mt.parameters()), [n for n, p_ in mt.named_parameters() if p_.grad is None]
    # E302: the capacitated plan. Two entries''')

p = "training/train/jends_train.py"
sub(p, "import jends as JE  # noqa: E402\n", "import jends as JE  # noqa: E402\nimport qnodes as QN  # noqa: E402\n")
sub(p, "    def __init__(self, maps_args, order=True, evidence=True, d=64, scales=False, e2e=False, capacity=False):", "    def __init__(self, maps_args, order=True, evidence=True, d=64, scales=False, e2e=False, capacity=False, ntok=False):")
sub(p, "scales=scales, capacity=capacity)", "scales=scales, capacity=capacity, ntok=64 if ntok else 0)\n        self.nodes = QN.NodeHead(c2=3 * 48, c1=48, c05=32, d=64) if ntok else None                       # E308: B's frozen node head: its tokens at the keypoints")
sub(p, "        return P, out, self.je(out, P, gt=gt)", "        tok = None\n        if self.nodes is not None:\n            with torch.no_grad():\n                tok = self.nodes(P[\"feat2\"], P[\"feat1\"], P[\"feat05\"], P[\"road2\"], P[\"centre1\"])[\"tok\"]\n        return P, out, self.je(out, P, gt=gt, tok=tok)")
sub(p, '    ap.add_argument("--e2e", action="store_true"', '    ap.add_argument("--ntok", action="store_true", help="E308: the pair head also reads the node tokens of B (frozen node head of the E273 baseline) at the keypoints")\n    ap.add_argument("--e2e", action="store_true"')
sub(p, "scales=a.scales, e2e=a.e2e, capacity=a.capacity)\n    model.jmodel.load_state_dict(ck[\"state\"])\n", "scales=a.scales, e2e=a.e2e, capacity=a.capacity, ntok=a.ntok)\n    model.jmodel.load_state_dict(ck[\"state\"])\n    if a.ntok:\n        base = ck[\"args\"].get(\"base\") or os.path.join(ROOT, \"runs\", f\"e273_p_s{a.seed}\", \"model.pt\")\n        bs_ = torch.load(base, map_location=\"cpu\")[\"state\"]\n        model.nodes.load_state_dict({k[6:]: v for k, v in bs_.items() if k.startswith(\"nodes.\")})\n")
sub(p, "evidence={not a.no_evidence} e2e={a.e2e} capacity={a.capacity}:", "evidence={not a.no_evidence} e2e={a.e2e} capacity={a.capacity} ntok={a.ntok}:")
sub(p, "e2e=a.e2e, capacity=a.capacity, maps=mp))", "e2e=a.e2e, capacity=a.capacity, ntok=a.ntok, maps=mp))")
p = "evaluation/jends_eval.py"
sub(p, 'capacity=bool(a.get("capacity", False)))', 'capacity=bool(a.get("capacity", False)), ntok=bool(a.get("ntok", False)))')
print("ok")
