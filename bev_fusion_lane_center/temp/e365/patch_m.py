import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
rep('''def _frame_mean(x, m):
    """per frame the mean of x (B, N) over the entries of m, then the mean over the frames that have one (0 when none)"""
    per = (x * m).sum(1) / m.sum(1).clamp_min(1)''', '''def _frame_mean(x, m, w=None):
    """per frame the mean of x (B, N) over the entries of m, then the mean over the frames that have one (0 when none); w (B, N): the weight of every entry, the mean is then sum(w x) / sum(w) as torch's class-weighted cross-entropy
    (x is the already weighted term); w None = the plain mean"""
    per = (x * m).sum(1) / (m.sum(1) if w is None else (w * m).sum(1)).clamp_min(1)''')
rep('''def jtoken_loss(out, batch, type_w=0.5, scale_w=0.5):''', '''def _source_nll(pos, none, has_g, none_weight):
    """per source: the likelihood of its GT successors (-log sum_{G(i)} p) when G(i) is non-empty, `none_weight` x -log p(none|i) when it is empty (the sources with a GT successor are not touched by the weight)"""
    return torch.where(has_g, pos, none * none_weight)


def jtoken_loss(out, batch, type_w=0.5, scale_w=0.5, none_weight=1.0):''')
rep('''    of the long-class links (finite where the cache fitted one)"""''', '''    of the long-class links (finite where the cache fitted one). none_weight W (default 1: the loss of before): multiplies (i) the loss of the sources whose G(i) is empty (their target is the none slot), in both classes, and (ii) the type
    cross-entropy of the tokens whose type is none (class 0 weight W, the six others 1, normalised by the sum of the weights of the valid tokens of the frame, as torch's weighted mean)"""''')
rep('''    nll = torch.where(G.any(-1), -torch.logsumexp(lp[..., :K].masked_fill(~G, NEG), -1), -lp[..., K])''', '''    nll = _source_nll(-torch.logsumexp(lp[..., :K].masked_fill(~G, NEG), -1), -lp[..., K], G.any(-1), none_weight)''')
rep('''    ce1, ce3 = (_frame_mean(F.cross_entropy(t.float().flatten(0, 1), yt.flatten(), reduction="none").view_as(yt), valid) for t in (out["types1"], out["types"]))''', '''    wt = None
    if none_weight != 1.0:
        wt = torch.ones(NTYPE, device=dev)
        wt[0] = none_weight
        wt = wt[yt]
    ce1, ce3 = (_frame_mean(F.cross_entropy(t.float().flatten(0, 1), yt.flatten(), reduction="none").view_as(yt) * (1.0 if wt is None else wt), valid, wt) for t in (out["types1"], out["types"]))''')
open(p, "w", encoding="utf-8").write(s)

p = sys.argv[2]
s = open(p, encoding="utf-8").read()
rep('''ARCH_DEFAULTS = dict(d=64, layers=3, heads=4, ffn=None, token_drop=0.0)''', '''ARCH_DEFAULTS = dict(d=64, layers=3, heads=4, ffn=None, token_drop=0.0, none_weight=1.0)''')
rep('''"ffn", "token_drop")''', '''"ffn", "token_drop", "none_weight")''')
rep('''    ap.add_argument("--smoke"''', '''    ap.add_argument("--none-weight", type=float, default=ARCH_DEFAULTS["none_weight"], dest="none_weight", help="W: multiplies the loss of the sources with an empty G(i) (target: none) and the type cross-entropy of the none class (1 = off)")
    ap.add_argument("--smoke"''')
rep('''def make_loss(run, token_drop=0.0, seed=0):''', '''def make_loss(run, token_drop=0.0, seed=0, none_weight=1.0):''')
rep('''JK.jtoken_loss(m(b), b)''', '''JK.jtoken_loss(m(b), b, none_weight=none_weight)''')
rep('''make_loss(run, a.token_drop, a.seed)''', '''make_loss(run, a.token_drop, a.seed, a.none_weight)''')
rep('''            + (f"token drop {a.token_drop} per step, " if a.token_drop > 0 else "")''', '''            + (f"token drop {a.token_drop} per step, " if a.token_drop > 0 else "") + (f"none weight {a.none_weight}, " if a.none_weight != 1.0 else "")''')
rep('''The key-weight switch of the head counts''', '''`--none-weight W` (default 1 = the loss of before) multiplies the loss of the sources whose GT successor set is empty (target: the none slot) and the type cross-entropy of the none class (`jtoken.jtoken_loss`).
The key-weight switch of the head counts''')
open(p, "w", encoding="utf-8").write(s)
