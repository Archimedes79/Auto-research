def rd(p):
    return open(p, encoding="utf-8").read()
def wr(p, s):
    open(p + ".new_s292", "w", encoding="utf-8", newline="\n").write(s)
def sub(s, a, b):
    assert s.count(a) == 1, a[:80]
    return s.replace(a, b)

# ============================================================ jwire_head.py: blind columns, pin test, exit-side label
h = rd("training/grid/jwire_head.py")
h = sub(h, "FEAT = 15\nP_MAX = JW.MAX_CAND\nREACH = 4.0\n",
        "FEAT = 15\nP_MAX = JW.MAX_CAND\nREACH = 4.0\nBLIND = (9, 13)                                                                                      # the keypoint probabilities of the own and the partner end: the pilot of E325 found them shifted between TRAIN (mean 0.97) and VAL (0.90, p10 0.6), the head must not read them\n")
h = sub(h, '''    def __init__(self, hid=32):
        super().__init__()
        self.l1, self.l2 = nn.Linear(FEAT, hid), nn.Linear(hid, 1)

    def forward(self, feat, mask):
        return self.l2(F.relu(self.l1(feat))).squeeze(-1).masked_fill(~mask, float("-inf"))
''', '''    def __init__(self, hid=32):
        super().__init__()
        self.l1, self.l2 = nn.Linear(FEAT, hid), nn.Linear(hid, 1)
        keep = torch.ones(FEAT)
        keep[list(BLIND)] = 0.0
        self.register_buffer("keep", keep, persistent=False)                                              # the columns of BLIND are multiplied by 0 in training, in the read-out and everywhere else

    def forward(self, feat, mask):
        return self.l2(F.relu(self.l1(feat * self.keep))).squeeze(-1).masked_fill(~mask, float("-inf"))
''')
h = sub(h, '    print(f"jwire_head selftest ok (545 parameters;', '    _selftest_pins()\n    print(f"jwire_head selftest ok (545 parameters;')
h = sub(h, 'if __name__ == "__main__":\n    import os\n    import sys\n', '''def _selftest_pins():
    """every one of the 15 features pinned on a hand-computed scene (a mutant that flips a sign, swaps own and partner probability, drops the kind flag or takes a place's features from its last candidate must fail), the
    exit side and its label, and the blind columns"""
    up = np.array([0.0, 1.0])

    def line(x, y0, y1):
        return np.c_[np.full(6, float(x)), np.linspace(y0, y1, 6)]

    # piece 0 ends at (0, -1); pieces 1 (length 27) and 2 (length 22) end at the SAME place (0, -3); piece 3 (a link, length 13) starts at (0, 2); piece 4 starts at (0, 3.5)
    lines = [line(0, -20, -1), line(0, -30, -3), line(0, -25, -3), line(0, 2, 15), line(0, 3.5, 20)]
    kinds = ["section", "section", "section", "link", "section"]
    nodes = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    links = [[0], [1], [2], [3], [4]]
    node_e = np.arange(10) / 10.0 + 0.05
    link_p = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    keep, ends = JW.kept_ends(lines, kinds, links, np.zeros(5, bool), [], REACH)
    attrs = piece_attrs(lines, kinds, nodes, links, node_e, link_p, keep)
    pair = [(np.array([0.2, 0.0]), up, np.array([0.1, 1.0]), up)]
    kp_p = [(0.7, 0.6)]
    inst = build_instances(ends, attrs, pair, kp_p, None)
    ent = [i for i in inst if i["side"] == 0][0]
    ex = [i for i in inst if i["side"] == 1][0]
    chord = np.linalg.norm([0.1, 1.0]) - 0.0
    chord = float(np.linalg.norm(np.array([0.1, 1.0]) - np.array([0.2, 0.0])))
    want0 = [np.linalg.norm([0.2, 1.0]) / 4, -1.0 / 4, 0.2 / 4, 1.0, 0.0, 0.0, np.log1p(19) / 5, node_e[1], link_p[0], 0.7, chord / 50, 1.0, 0.0, 0.6, 1.0]
    assert np.allclose(ent["feat"][0], want0, atol=1e-5), (ent["feat"][0], want0)
    want1 = [np.linalg.norm([0.2, 3.0]) / 4, -3.0 / 4, 0.2 / 4, 1.0, 0.0, 0.0, np.log1p(27) / 5, node_e[3], link_p[1], 0.7, chord / 50, 1.0, 0.0, 0.6, 1.0]      # the place of pieces 1 and 2: the FIRST candidate (piece 1, length 27)
    assert np.allclose(ent["feat"][1], want1, atol=1e-5), (ent["feat"][1], want1)
    assert ent["piece"] == [0, 1]
    wantx = [np.linalg.norm([0.1, 1.0]) / 4, 1.0 / 4, 0.1 / 4, 1.0, 0.0, 1.0, np.log1p(13) / 5, node_e[6], link_p[3], 0.6, chord / 50, 1.0, 0.0, 0.7, 0.0]            # the exit joins the FIRST point of piece 3, the partner is the entry keypoint
    assert np.allclose(ex["feat"][0], wantx, atol=1e-5), (ex["feat"][0], wantx)
    assert ex["piece"] == [3, 4] and np.isclose(ex["feat"][1, 7], node_e[8]) and np.isclose(ex["feat"][1, 8], link_p[4])
    # the exit side's label: the oracle place is the second place (piece 4 at (0, 3.5)), the keypoint's nearest is piece 3 (a label taken at the keypoint is 0)
    gt = [(None, np.array([0.0, 3.45, 0.0, 1.0]))]
    lx = [i for i in build_instances(ends, attrs, pair, kp_p, gt) if i["side"] == 1][0]
    assert lx["label"] == 1 and lx["kind"] == "oracle", (lx["label"], lx["kind"])
    # the blind columns: the logits do not move when the keypoint probabilities change
    w = Wire()
    feat, mask, _ = pack(inst)
    f2 = feat.clone()
    f2[..., list(BLIND)] += 5.0
    assert torch.equal(w(feat, mask), w(f2, mask)), "the head reads a blind column"
    assert sum(p.numel() for p in w.parameters()) == 545 and "keep" not in w.state_dict()


if __name__ == "__main__":
    import os
    import sys
''')
wr("training/grid/jwire_head.py", h)
print("ok")
