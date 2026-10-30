p = "tests/radar_codecs.py"
s = open(p, encoding="utf-8").read()

# 1 probe with selectable activation / initialisation
s = s.replace('''class Codec:''', '''class Probe(nn.Module):
    """encoder (two 3x3 convs) + 1x1 decoder; act 'relu' or 'leaky' (LeakyReLU 0.5: invertible, a ReLU deletes the negative half of
    every signed channel); ident = centre-tap identity start (the net starts as a near-lossless pass-through, training only
    has to keep it, instead of discovering the identity from noise)."""

    def __init__(self, cin, C, act="relu", ident=False):
        super().__init__()
        A = (lambda: nn.ReLU()) if act == "relu" else (lambda: nn.LeakyReLU(0.5))
        self.enc = nn.Sequential(nn.Conv2d(cin, C, 3, padding=1), A(), nn.Conv2d(C, C, 3, padding=1), A())
        self.dec = nn.Sequential(nn.Conv2d(C, C, 1), A(), nn.Conv2d(C, cin, 1))
        if ident:
            for conv in (self.enc[0], self.enc[2], self.dec[0], self.dec[2]):
                with torch.no_grad():
                    conv.weight.mul_(0.05)
                    for i in range(min(conv.weight.shape[0], conv.weight.shape[1])):
                        conv.weight[i, i, conv.weight.shape[2] // 2, conv.weight.shape[3] // 2] += 1.0
                    conv.bias.zero_()

    def forward(self, x):
        return self.dec(self.enc(x))


class Codec:''', 1)

# 2 train_codec takes variant strings 'ae', 'skip', 'leaky', 'leaky+id', 'relu+id'
s = s.replace('''    net = (Skip(cin, C) if variant == "skip" else rt.AE(cin, C, cin)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = np.random.RandomState(seed)
    for _ in range(steps):
        fs = [frames[i] for i in g.choice(len(frames), 8, replace=False)]
        tgt = codec_raster(codec, rt.batch(fs), 8)
        loss = codec_loss(codec, net(tgt), tgt)''', '''    if variant == "skip":
        net = Skip(cin, C)
    else:
        net = Probe(cin, C, act="leaky" if variant.startswith("leaky") else "relu", ident=variant.endswith("+id"))
    net = net.to(DEV)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = np.random.RandomState(seed)
    for _ in range(steps):
        fs = [frames[i] for i in g.choice(len(frames), 8, replace=False)]
        tgt = codec_raster(codec, rt.batch(fs), 8)
        loss = codec_loss(codec, net(tgt), tgt)''', 1)

# 3 level E + CLI
s = s.replace('''if __name__ == "__main__":
    level_d(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)''', '''def level_e(steps, names=("abs", "delta")):
    print(f"E  architecture of the probe (activation, identity start), codecs {names}, C = 64, {steps} steps")
    tr, te = rt.load(set(range(8))), rt.load({8, 9})
    print("  codec / variant     | channels in | C   | held-out | train | gap  | held-out median: " + ", ".join(rt.TOL))
    for name in names:
        for variant in ("ae", "leaky", "relu+id", "leaky+id"):
            c = Codec(name)
            net = train_codec(c, variant, 64, steps, tr)
            e, ok = eval_codec(c, net, te)
            _, ok_tr = eval_codec(c, net, tr)
            med = ", ".join(f"{np.median(e[k][torch.isfinite(e[k])].cpu().numpy()):.3f}" for k in rt.TOL)
            h, t = ok.float().mean().item(), ok_tr.float().mean().item()
            print(f"  {name + ' / ' + variant:19s} | {1 + (rt.K + 1) * (1 + c.P):11d} | {64:3d} | {100 * h:7.1f}% | {100 * t:4.1f}% | {100 * (t - h):+4.1f} | {med}", flush=True)


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 2000
    if "e" in sys.argv[2:]:
        level_e(steps)
    else:
        level_d(steps)''', 1)
open(p, "w", encoding="utf-8").write(s)
print("patched")
