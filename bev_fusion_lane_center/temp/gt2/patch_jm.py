p = "../../training/grid/jmaps.py"
s = open(p, encoding="utf-8").read()
old = "def _dice(logit, y, eps=1.0):"
new = '''class JModel(nn.Module):
    """the pixel stage of the lane head (pyramid) + the junction maps head; `pyr` is loaded from a baseline run (`runs/e273_p_s<k>/model.pt`, tensors `pyr.*`), so the junction maps read the SAME pixel features as B"""

    def __init__(self, width=48, f05=32, d=64, ctx="dil"):
        super().__init__()
        from pyramid import PyramidHead
        self.width, self.ctx = width, ctx
        self.pyr = PyramidHead(width=width, f05=f05)
        self.jm = JMaps(c2=3 * width, c1=width, c05=f05, d=d, ctx=ctx)

    def load_base(self, path):
        sd = torch.load(path, map_location="cpu")["state"]
        own = self.pyr.state_dict()
        fit = {k[4:]: v for k, v in sd.items() if k.startswith("pyr.") and k[4:] in own and own[k[4:]].shape == v.shape}
        self.pyr.load_state_dict(fit, strict=False)
        return len(fit), len(own)

    def forward(self, x, joint=False):
        if joint:
            P = self.pyr(x)
        else:
            with torch.no_grad():
                P = self.pyr(x)
        return P, self.jm(P)


def _dice(logit, y, eps=1.0):'''
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
