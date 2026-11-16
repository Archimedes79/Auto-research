"""E325 step 2: the learned wiring head `Wire` - which delivered piece a connector end is joined to, chosen by a small softmax over the PLACES in reach instead of the nearest-piece rule (the twin, `jwire.py`).

    attrs = piece_attrs(lines, kinds, piece_nodes, piece_links, node_e, link_p, keep)           # per KEPT piece: (kind, log length, e / link p at its first and last end)  -> (n, 6)
    inst  = build_instances(ends, attrs, pairs, kp_p, gts, wire_pos)                            # one instance per connector end: candidate PLACES, 15 features each, the label (training) -> list of dicts
    X     = pack(inst)                                                                          # padded tensors feat (N, 8, 15), mask (N, 8), label (N,) for `Wire` / `wire_loss`
    wire  = Wire()                                                                              # 15 x 32 + 32 + 32 + 1 = 545 parameters
    choice = wire_choice_head(wire, ends, attrs, pairs, kp_p, wire_pos)                         # one (entry piece index or None, exit piece index or None) per pair: `add_connectors(wire_choice=choice)`

Places, not pieces: several candidate piece ends can sit at the SAME place (co-located ends of a fork or a merge; key = the position rounded to 6 decimals, as the G0 gate counts them). The head chooses among the PLACES, the
first candidate of a place in the list order is the piece that is wired (the rule's order inside a place), and the features of a place are those of that first candidate. At most `jwire.MAX_CAND` candidates enter (an index
choice, the gate of the rule: 4 m, 45 degrees). Features (15, all computed from the stored graph and the keypoint pair, NEVER from the GT): distance d, offsets along and across the end's heading (m / reach), the heading
difference (cos, sin), the kind of the piece (1 = connector link, 0 = section), log(1 + length) / 5, the existence of B's end node and the link probability at that end, the probability of the keypoint, the chord length of the pair
/ 50, the heading difference of the pair (cos, sin), the probability of the partner keypoint, the end kind (1 = entry). Label (training only): the place of the piece the rule picks at the matched GT END (the G0 oracle); an end that matches
no GT end or finds no piece at it gets the rule's choice at the keypoint (label 0); an end whose oracle place is not among the keypoint's places has NO label (-1: no gradient). The head is trained on labelled ends only.
Run the self-test: python jwire_head.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import jwire as JW

FEAT = 15
P_MAX = JW.MAX_CAND
REACH = 4.0
BLIND = (9, 13)                                                                                      # the keypoint probabilities of the own and the partner end: the pilot of E325 found them shifted between TRAIN (mean 0.97) and VAL (0.90, p10 0.6), the head must not read them


def piece_attrs(lines, kinds, piece_nodes, piece_links, node_e, link_p, keep):
    """-> (n_kept, 6) float32 per kept piece: kind (1 = connector link, 0 = section), log1p(arc length) / 5, existence of the first / last node, probability of the first / last link"""
    out = np.zeros((len(keep), 6), np.float32)
    for r, i in enumerate(keep):
        P = np.asarray(lines[i], float)
        length = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0
        nodes, links = piece_nodes[i], piece_links[i]
        out[r] = (float(kinds[i] == "link"), np.log1p(length) / 5.0, float(node_e[nodes[0]]), float(node_e[nodes[-1]]), float(link_p[links[0]]), float(link_p[links[-1]]))
    return out


def end_instance(ends, attrs, pos, tan, entry, kp_p, partner, wire_pos=REACH):
    """the candidate places of ONE connector end and their features -> dict(feat (P, 15) float32, piece [index of the first candidate of every place], keys [place keys]) or None (nothing in reach)
    partner = (partner position, partner heading, partner keypoint probability)"""
    pos, tan = np.asarray(pos, float), np.asarray(tan, float)
    cands = JW.candidates(ends, pos, tan, entry, wire_pos)
    if not cands:
        return None
    places, keys = [], []
    for c in cands:
        key = tuple(np.round(c[2], 6))
        if key in keys:
            places[keys.index(key)].append(c)
        else:
            keys.append(key)
            places.append([c])
    ppos, ptan, pp = np.asarray(partner[0], float), np.asarray(partner[1], float), float(partner[2])
    chord, pcos, psin = float(np.linalg.norm(ppos - pos)), float(tan @ ptan), float(tan[0] * ptan[1] - tan[1] * ptan[0])
    nrm = np.array([-tan[1], tan[0]])
    rows = []
    for pl in places:
        d, i, q, t = pl[0]
        rel = np.asarray(q, float) - pos
        a = attrs[i]
        e, p = (a[3], a[5]) if entry else (a[2], a[4])                                                   # an entry end joins the LAST point of a piece, an exit end the FIRST point
        rows.append([d / wire_pos, float(rel @ tan) / wire_pos, float(rel @ nrm) / wire_pos, float(np.asarray(t, float) @ tan), float(tan[0] * t[1] - tan[1] * t[0]), a[0], a[1], e, p, float(kp_p), chord / 50.0, pcos, psin, pp, float(entry)])
    return dict(feat=np.array(rows, np.float32), piece=[pl[0][1] for pl in places], keys=keys)


def build_instances(ends, attrs, pairs, kp_p, gts, wire_pos=REACH, jitter=None):
    """one instance per connector end of every pair (entry, exit) with at least one candidate: end_instance + `n` (pair index), `side` (0 entry / 1 exit), `label`, `kind` ('oracle' / 'rule' / 'outside').
    kp_p = [(entry probability, exit probability)] per pair; gts = [(GT entry end (4,) or None, GT exit end (4,) or None)] per pair or None (read-out: no labels); jitter(pos, tan) -> (pos, tan) perturbs the keypoint
    of an end before the candidates are gathered (training only: it emulates the keypoint error of unseen frames); the features never read `gts`"""
    out = []
    for n, pr in enumerate(pairs):
        pa, ta, pb, tb = pr[:4]
        for side, (pos, tan, partner) in enumerate(((pa, ta, (pb, tb, kp_p[n][1])), (pb, tb, (pa, ta, kp_p[n][0])))):
            if jitter is not None:
                pos, tan = jitter(np.asarray(pos, float), np.asarray(tan, float))
            ins = end_instance(ends, attrs, pos, tan, side == 0, kp_p[n][side], partner, wire_pos)
            if ins is None:
                continue
            g = None if gts is None else gts[n][side]
            lab, kind = 0, "rule"
            if g is not None:
                cg = JW.candidates(ends, g[:2], g[2:], side == 0, wire_pos)
                if cg:
                    key = tuple(np.round(cg[0][2], 6))
                    lab, kind = (ins["keys"].index(key), "oracle") if key in ins["keys"] else (-1, "outside")
            out.append(dict(ins, n=n, side=side, label=lab, kind=kind))
    return out


def pack(inst):
    """-> feat (N, P_MAX, 15) float32, mask (N, P_MAX) bool, label (N,) int64 (-1 = no label) as torch tensors"""
    N = len(inst)
    feat, mask, label = np.zeros((N, P_MAX, FEAT), np.float32), np.zeros((N, P_MAX), bool), np.full(N, -1, np.int64)
    for r, ins in enumerate(inst):
        k = len(ins["piece"])
        feat[r, :k], mask[r, :k], label[r] = ins["feat"], True, ins["label"]
    return torch.from_numpy(feat), torch.from_numpy(mask), torch.from_numpy(label)


class Wire(nn.Module):
    """the head: an MLP 15 -> 32 -> 1 on the features of every place; the softmax over the places of an end is the weight of each place; 545 parameters"""

    def __init__(self, hid=32):
        super().__init__()
        self.l1, self.l2 = nn.Linear(FEAT, hid), nn.Linear(hid, 1)
        keep = torch.ones(FEAT)
        keep[list(BLIND)] = 0.0
        self.register_buffer("keep", keep, persistent=False)                                              # the columns of BLIND are multiplied by 0 in training, in the read-out and everywhere else

    def forward(self, feat, mask):
        return self.l2(F.relu(self.l1(feat * self.keep))).squeeze(-1).masked_fill(~mask, float("-inf"))

    @classmethod
    def rule_like(cls):
        """the head with hand-set weights that imitates the nearest-piece rule exactly (logit = -relu(distance feature)): the twin in the shape of the head, the fixture of the read-out path (`jends_eval.py score --wire head`
        with it must reproduce the card of record)"""
        w = cls()
        with torch.no_grad():
            for p in w.parameters():
                p.zero_()
            w.l1.weight[0, 0] = 1.0
            w.l2.weight[0, 0] = -1.0
        return w


def wire_loss(logits, label):
    """cross entropy over the places of the LABELLED ends only (label >= 0); 0 when no end is labelled (no gradient)"""
    sel = label >= 0
    if not bool(sel.any()):
        return torch.nan_to_num(logits, neginf=0.0).sum() * 0.0
    return F.cross_entropy(logits[sel], label[sel])


def load_wire(run, name="model.pt"):
    """the head of a run (`runs/<run>/<name>`, written by `jwire_train.py run`) -> `Wire` in eval mode"""
    import os
    root = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ck = torch.load(os.path.join(root, "runs", run, name or "model.pt"), map_location="cpu")
    w = Wire()
    w.load_state_dict(ck["state"])
    w.meta = dict(ck.get("args") or {})                                                               # E358 staged: the provenance of the head (seed, pair_run, ends); E325's heads carry no pair_run / ends
    return w.eval()


def check_head(head, run, seed, attach):
    """E358 staged: a head is read out only on the kind of ends it was trained on (`ends`: 'kp' for E325 / E350, 'att' for E358; no field = 'kp'), on the pair run and the seed it was trained for (no `pair_run` field = E312's `e312_p_s<seed>`)"""
    m = getattr(head, "meta", None) or {}
    ends = m.get("ends", "kp")
    want = "att" if attach else "kp"
    assert ends == want, f"the head was trained on {ends!r} ends, the read-out uses {want!r} ends: refused"
    hs = m.get("seed", seed)
    assert hs == seed, f"the head was trained for seed {hs}, the read-out is seed {seed}: refused"
    hp = m.get("pair_run") or f"e312_p_s{hs}"
    assert hp == run, f"the head belongs to the pair run {hp!r}, the read-out is on {run!r}: refused"


@torch.no_grad()
def wire_choice_head(wire, ends, attrs, pairs, kp_p, wire_pos=REACH):
    """the read-out: one (entry piece index or None, exit piece index or None) per pair = the first candidate of the place with the largest logit (an index into the KEPT pieces); an end with nothing in reach is None"""
    inst = build_instances(ends, attrs, pairs, kp_p, None, wire_pos)
    choice = [[None, None] for _ in pairs]
    if inst:
        feat, mask, _ = pack(inst)
        best = wire(feat, mask).argmax(-1).tolist()
        for ins, b in zip(inst, best):
            choice[ins["n"]][ins["side"]] = int(ins["piece"][b])
    return [tuple(c) for c in choice]


def _selftest():
    rng = np.random.default_rng(0)
    up = np.array([0.0, 1.0])

    def line(x, y0, y1):
        return np.c_[np.full(6, float(x)), np.linspace(y0, y1, 6)]

    # a scene: piece 0 ends at (0, -1), pieces 1 and 2 end at the SAME place (0, -3) (a merge), piece 3 is a long link starting at (0, 2)
    lines = [line(0, -20, -1), line(0, -30, -3), line(0, -25, -3), line(0, 2, 15)]
    kinds = ["section", "section", "section", "link"]
    nodes = [[0, 1], [2, 3], [4, 5], [6, 7]]
    links = [[0], [1], [2], [3]]
    node_e = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    link_p = np.array([0.95, 0.85, 0.75, 0.65])
    keep, ends = JW.kept_ends(lines, kinds, links, np.zeros(4, bool), [], REACH)
    attrs = piece_attrs(lines, kinds, nodes, links, node_e, link_p, keep)
    assert attrs.shape == (4, 6) and attrs[3, 0] == 1.0 and attrs[0, 0] == 0.0 and np.isclose(attrs[0, 2], 0.9) and np.isclose(attrs[0, 3], 0.8) and np.isclose(attrs[0, 4], 0.95)
    pair = [(np.array([0.2, 0.0]), up, np.array([0.1, 1.0]), up)]                                       # entry keypoint at (0.2, 0), exit keypoint at (0.1, 1)
    kp_p = [(0.7, 0.6)]
    # 1. places and features: the merge is ONE place (the first of the list), the exit side reads the FIRST point of a piece
    inst = build_instances(ends, attrs, pair, kp_p, None)
    ent = [i for i in inst if i["side"] == 0][0]
    ex = [i for i in inst if i["side"] == 1][0]
    assert ent["piece"] == [0, 1] and len(ent["keys"]) == 2 and ent["feat"].shape == (2, FEAT), ent["piece"]            # place 0: piece 0 at (0, -1); place 1: pieces 1 and 2 at (0, -3), the first of them
    assert ex["piece"] == [3] and np.isclose(ex["feat"][0, 7], node_e[6]) and np.isclose(ex["feat"][0, 8], link_p[3]), ex["feat"][0]          # the exit joins the FIRST point of piece 3: e / p of its first node / link
    assert np.isclose(ent["feat"][0, 7], node_e[1]) and np.isclose(ent["feat"][0, 8], link_p[0]), ent["feat"][0]                         # the entry joins the LAST point of piece 0
    assert np.isclose(ent["feat"][0, 0], np.linalg.norm([0.2, 1.0]) / REACH, atol=1e-6) and ent["feat"][0, 14] == 1.0 and ex["feat"][0, 14] == 0.0
    # 2. labels: the oracle label is the place of the piece the rule picks at the matched GT END (here the merge place), not the keypoint's nearest; mutation: a label taken at the keypoint is 0
    gt = [(np.array([0.0, -3.05, 0.0, 1.0]), None)]
    li = [i for i in build_instances(ends, attrs, pair, kp_p, gt) if i["side"] == 0][0]
    assert li["label"] == 1 and li["kind"] == "oracle", (li["label"], li["kind"])
    assert [i for i in build_instances(ends, attrs, pair, kp_p, None) if i["side"] == 0][0]["label"] == 0                                    # no GT: the rule's choice at the keypoint
    assert [i for i in build_instances(ends, attrs, pair, kp_p, [(np.array([30.0, 30.0, 0.0, 1.0]), None)]) if i["side"] == 0][0]["kind"] == "rule"   # no piece at the GT end: the rule's choice
    far = [(np.array([0.2, 2.8]), up, np.array([0.1, 9.0]), up)]                                         # the keypoint at (0.2, 2.8): only piece 3 start is in reach... entry side sees piece 0's end at 3.8 m
    fi = [i for i in build_instances(ends, attrs, far, kp_p, [(np.array([0.0, -3.05, 0.0, 1.0]), None)]) if i["side"] == 0][0]
    assert fi["label"] == -1 and fi["kind"] == "outside", (fi["label"], fi["kind"])                       # the oracle place is not among the keypoint's places: NO label
    # 3. leak test: the features never read the GT (a mutant that took a feature from the GT end would change them)
    f1 = pack(build_instances(ends, attrs, pair, kp_p, gt))[0]
    f2 = pack(build_instances(ends, attrs, pair, kp_p, [(np.array([5.0, 5.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0, 1.0]))]))[0]
    f3 = pack(build_instances(ends, attrs, pair, kp_p, None))[0]
    assert torch.equal(f1, f3) and torch.equal(f1, f2), "a feature reads the GT"
    ins_a, ins_b, ins_c = (build_instances(ends, attrs, pair, kp_p, g) for g in (gt, [(np.array([5.0, 5.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0, 1.0]))], None))
    assert all(np.array_equal(a["feat"], b["feat"]) and np.array_equal(a["feat"], c["feat"]) for a, b, c in zip(ins_a, ins_b, ins_c)), "a feature reads the GT"
    # 4. float64 distances at large coordinates and the jitter hook
    big = [(np.array([49.9, 0.0]) + 0.0, up, np.array([49.9, 1.0]), up)]
    lb = [line(49.9, -20, -1)]
    k2, e2 = JW.kept_ends(lb, ["section"], [[0]], np.zeros(1, bool), [], REACH)
    a2 = piece_attrs(lb, ["section"], [[0, 1]], [[0]], node_e, link_p, k2)
    i2 = build_instances(e2, a2, big, kp_p, None)[0]
    assert abs(i2["feat"][0, 0] - 1.0 / REACH) < 1e-6, i2["feat"][0, 0]
    jit = lambda p, t: (p + np.array([0.0, 2.0]), t)  # noqa: E731
    assert build_instances(e2, a2, big, kp_p, None, jitter=jit)[0]["feat"][0, 0] > i2["feat"][0, 0] + 0.1                    # the keypoint was moved 2 m away from the piece end
    # 5. the head: 545 parameters, the softmax is over the valid places only, the gradient reaches both layers, an unlabelled end gives no loss and no gradient
    w = Wire()
    assert sum(p.numel() for p in w.parameters()) == 545
    feat, mask, label = pack(build_instances(ends, attrs, pair, kp_p, gt))
    lg = w(feat, mask)
    assert lg.shape == (feat.shape[0], P_MAX) and bool(torch.isinf(lg[~mask]).all()) and bool(torch.isfinite(lg[mask]).all())
    pr = lg.softmax(-1)
    assert torch.allclose(pr.sum(-1), torch.ones(len(pr))) and bool((pr[~mask] == 0).all())
    loss = wire_loss(lg, label)
    loss.backward()
    assert float(w.l1.weight.grad.abs().sum()) > 0 and float(w.l2.weight.grad.abs().sum()) > 0
    w.zero_grad()
    none_lab = torch.full_like(label, -1)
    assert float(wire_loss(w(feat, mask), none_lab)) == 0.0
    wire_loss(w(feat, mask), none_lab).backward()
    assert all(float(p.grad.abs().sum()) == 0.0 for p in w.parameters())
    # the gradient on labelled ends only: label one end, the loss of the other (unlabelled) end must not enter
    lab1 = label.clone()
    lab1[:] = -1
    lab1[0] = label[0] if label[0] >= 0 else 0
    assert abs(float(wire_loss(w(feat, mask), lab1)) - float(F.cross_entropy(w(feat, mask)[0:1], lab1[0:1]))) < 1e-6
    # 6. the plumbing: a head with the rule's logits (-d^2) chooses exactly what the twin chooses (random scenes with co-located ends); a trained-to-imitate head reaches the rule's agreement
    class RuleWire(nn.Module):
        def forward(self, feat, mask):
            return (-(feat[..., 0] ** 2)).masked_fill(~mask, float("-inf"))

    n_pairs = 0
    for _ in range(200):
        n_p = int(rng.integers(1, 10))
        ls = []
        for i in range(n_p):
            a0 = rng.uniform(-6, 6, 2)
            h = rng.uniform(0, 2 * np.pi)
            L = rng.uniform(3, 20)
            t = np.linspace(0, L, 8)
            ls.append(a0 + np.c_[t * np.cos(h), t * np.sin(h)])
        if n_p > 3 and rng.random() < 0.5:
            ls[2][-1] = ls[1][-1]
        kd = ["section" if rng.random() < 0.7 else "link" for _ in ls]
        nd = [[2 * i, 2 * i + 1] for i in range(n_p)]
        lk = [[i] for i in range(n_p)]
        ne, lp = rng.uniform(0, 1, 2 * n_p), rng.uniform(0, 1, n_p)
        pairs = []
        for _ in range(int(rng.integers(1, 5))):
            u1, u2 = (np.array([np.cos(a), np.sin(a)]) for a in rng.uniform(0, 2 * np.pi, 2))
            base = ls[int(rng.integers(n_p))]
            pairs.append((base[-1] + rng.normal(0, 1.0, 2), u1, base[0] + rng.normal(0, 1.0, 2), u2))
        kp = [(float(rng.uniform()), float(rng.uniform())) for _ in pairs]
        kp_k, en = JW.kept_ends(ls, kd, lk, np.array([k == "link" for k in kd]), pairs, REACH)
        at = piece_attrs(ls, kd, nd, lk, ne, lp, kp_k)
        got = wire_choice_head(RuleWire(), en, at, pairs, kp)
        assert got == JW.twin_choice(en, pairs, REACH), (got, JW.twin_choice(en, pairs, REACH))
        assert wire_choice_head(Wire.rule_like(), en, at, pairs, kp) == got, "the hand-set rule-like head must choose what the twin chooses"
        n_pairs += len(pairs)
    # 7. the head learns the oracle label where the rule is wrong: a synthetic set with the label on the second place when a feature says so
    torch.manual_seed(0)
    N = 600
    ft = torch.randn(N, P_MAX, FEAT)
    mk = torch.zeros(N, P_MAX, dtype=torch.bool)
    mk[:, :3] = True
    lb = ft[:, :3, 5].argmax(-1)                                                                        # the label = the place with the largest feature 5 among the first 3 (a rule that is not 'nearest')
    w = Wire()
    opt = torch.optim.Adam(w.parameters(), lr=1e-2)
    for _ in range(150):
        opt.zero_grad()
        wire_loss(w(ft, mk), lb).backward()
        opt.step()
    acc = float((w(ft, mk).argmax(-1) == lb).float().mean())
    assert acc > 0.85, acc
    _selftest_pins()
    print(f"jwire_head selftest ok (545 parameters; {n_pairs} pairs: a head with the rule's logits = the twin; synthetic accuracy {acc:.2f})")


def _selftest_pins():
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
    root = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, os.path.join(root, "evaluation"))
    _selftest()
