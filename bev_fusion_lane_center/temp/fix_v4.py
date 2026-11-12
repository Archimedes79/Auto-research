"""one-off: E273 link head, attempt 4: set attention over the active nodes (context for which entry joins which exit), node scores (e, flags, u) as inputs, a smaller positive weight for the long links"""


def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:80])
    s = s.replace(old, new, count)
    open(path, "w", encoding="utf-8").write(s)


p = "training/grid/qlinks.py"
sub(p, "from qnodes import N, cell_centres, sample, unit", "from qassign import fourier_pts\nfrom qnodes import N, cell_centres, sample, unit")
sub(p, '''class LinkHead(nn.Module):
    def __init__(self, d=64, hidden=128):
        super().__init__()''', '''class SetBlock(nn.Module):
    """self-attention over the active nodes of a frame (all pairs, padding masked), pre-norm, residual, feed-forward"""

    def __init__(self, d, heads=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, pad):
        h = self.n1(x)
        a = self.att(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        x = x + torch.nan_to_num(a)
        return x + self.ffn(self.n2(x))


class LinkHead(nn.Module):
    def __init__(self, d=64, hidden=128, ctx_layers=2):
        super().__init__()
        self.nin = nn.Sequential(nn.Linear(16 + 2 + 11, d), nn.GELU(), nn.Linear(d, d))
        self.ctx = nn.ModuleList([SetBlock(d) for _ in range(ctx_layers)])''')
sub(p, '''        pos = gather(no["pos"], idx)
        dirv = unit(gather(no["dir_raw"], idx), dim=-1)
        tok = gather(no["tok"], idx)''', '''        pos = gather(no["pos"], idx)
        dirv = unit(gather(no["dir_raw"], idx), dim=-1)
        sc11 = torch.cat([no["e"][:, None], no["flags"], no["uout"][:, None], no["uin"][:, None]], 1).float().sigmoid()                 # the node scores of the cell: e, 8 flags, uout, uin
        tok = gather(no["tok"], idx) + self.nin(torch.cat([fourier_pts(pos.float()), dirv.float(), gather(sc11, idx)], -1)).to(no["tok"].dtype)
        for blk in self.ctx:                                                                    # context: every active node attends to all others of its frame
            tok = blk(tok, ~valid)''')
sub(p, "Link logit = MLP([token i, token j,", "The tokens first pass 2 layers of self-attention over the active nodes of the frame (with the node's position, direction and scores e / flags / u as extra input: which entry joins which exit is a question about the whole junction).\nLink logit = MLP([token i, token j,")
# the long-link positive weight
sub("training/grid/qgraph.py", "(nnl / npl.clamp_min(1.0)).clamp(1.0, 80.0)", "(nnl / npl.clamp_min(1.0)).clamp(1.0, 15.0)")
print("ok")
