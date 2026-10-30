import re
# blocks_new.py: CtxAdapter
p = "training/blocks_new.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''Run the self-test: python training/blocks_new.py
"""''', '''CtxAdapter     trainable residual adapter on the cached (frozen, detection-trained) image context features before the lift:
               x + conv3x3(relu(norm(conv3x3(x)))), the last conv zero-initialised, so training starts exactly at the reference
               system (try 6 of t3_improve_loop: are the frozen features the bottleneck?).
Run the self-test: python training/blocks_new.py
"""''')
s = s.replace('''def _selftest():
    torch.manual_seed(0)''', '''class CtxAdapter(nn.Module):
    def __init__(self, c=80, hidden=96):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, hidden, 3, padding=1, bias=False), nn.GroupNorm(8, hidden), nn.ReLU(True),
                                  nn.Conv2d(hidden, c, 3, padding=1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):                                  # (N, c, H, W)
        return x + self.body(x)


def _selftest():
    torch.manual_seed(0)''')
s = s.replace('''    print(f"blocks_new selftest ok:''', '''    a = CtxAdapter(80)
    xa = torch.randn(3, 80, 16, 44)
    assert torch.equal(a(xa), xa), "zero-initialised adapter is the identity"
    a(xa).sum().backward()
    assert a.body[-1].weight.grad is not None and a.body[0].weight.grad is not None
    print(f"blocks_new selftest ok:''')
s = s.replace('''f"{sum(q.numel() for q in t.parameters()) / 1e6:.2f} M parameters)")''', '''f"{sum(q.numel() for q in t.parameters()) / 1e6:.2f} M parameters), CtxAdapter (identity at init, gradients)")''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
# model.py
p = "training/model.py"
s = open(p, encoding="utf-8").read()
s = s.replace("from .blocks_new import RadarPillars, SecondTrunk", "from .blocks_new import CtxAdapter, RadarPillars, SecondTrunk")
s = s.replace("from blocks_new import RadarPillars, SecondTrunk", "from blocks_new import CtxAdapter, RadarPillars, SecondTrunk")
s = s.replace('''    camera_off: bool = False           # study arm''', '''    cam_adapter: bool = False          # try 6 (t3_improve_loop): trainable residual adapter on the image context features before the lift
    camera_off: bool = False           # study arm''')
s = s.replace('''        self.cam_reduce = nn.Sequential(nn.Conv2d(nz * n_ctx''', '''        self.cam_adapter = CtxAdapter(n_ctx) if cfg.cam_adapter else None
        self.cam_reduce = nn.Sequential(nn.Conv2d(nz * n_ctx''')
s = s.replace('''        prob, ctx = prob.view(B, Fn, *prob.shape[1:]), ctx.view(B, Fn, *ctx.shape[1:])
        hw = img.shape[-2:]''', '''        prob, ctx = prob.view(B, Fn, *prob.shape[1:]), ctx.view(B, Fn, *ctx.shape[1:])
        if self.cam_adapter is not None:
            ctx = self.cam_adapter(ctx.flatten(0, 1)).view_as(ctx)
        hw = img.shape[-2:]''')
s = s.replace('''             {"camera_off": True}]
    tried = []''', '''             {"camera_off": True}, {"cam_adapter": True}]
    tried = []''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
# feature_cache.py cached override
p = "training/feature_cache.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''            prob, ctx = batch["prob"].float(), batch["ctx"].float()
            B, Fn = prob.shape[:2]''', '''            prob, ctx = batch["prob"].float(), batch["ctx"].float()
            B, Fn = prob.shape[:2]
            if self.cam_adapter is not None:
                ctx = self.cam_adapter(ctx.flatten(0, 1)).view_as(ctx)''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
