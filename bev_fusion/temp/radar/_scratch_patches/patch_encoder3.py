p = "training/radar_encoder.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''        if identity_init:
            for conv in (self.net[0], self.net[2], self.decoder[0], self.decoder[2]):
                with torch.no_grad():
                    conv.weight.mul_(0.05)
                    for i in range(min(conv.weight.shape[0], conv.weight.shape[1])):
                        conv.weight[i, i, conv.weight.shape[2] // 2, conv.weight.shape[3] // 2] += 1.0
                    conv.bias.zero_()
''', '''        if identity_init:
            for conv in (self.net[0], self.net[2], self.decoder[0], self.decoder[2]):
                _pass_through_(conv)
''')
s = s.replace('''class RadarEncoder(nn.Module):''', '''def _pass_through_(conv):
    """centre tap = identity on the first min(out, in) channels, small noise elsewhere, zero bias."""
    with torch.no_grad():
        conv.weight.mul_(0.05)
        for i in range(min(conv.weight.shape[0], conv.weight.shape[1])):
            conv.weight[i, i, conv.weight.shape[2] // 2, conv.weight.shape[3] // 2] += 1.0
        conv.bias.zero_()


class RadarEncoder(nn.Module):''', 1)
s = s.replace('''    def new_decoder(self, C=None):
        """A fresh 1 x 1 decoder (default start) for probing frozen features: the honest way to measure what a task-trained encoder kept."""
        C = C or self.net[2].out_channels
        return nn.Sequential(nn.Conv2d(C, C, 1), nn.LeakyReLU(0.5), nn.Conv2d(C, self.cin, 1)).to(self.w_slot.device)''',
'''    def new_decoder(self, C=None, identity_init=True):
        """A fresh 1 x 1 decoder for probing frozen features (same start as the module's own decoder: a probe that has to find the
        identity from a random start under-measures): the honest way to measure what a task-trained encoder kept."""
        C = C or self.net[2].out_channels
        dec = nn.Sequential(nn.Conv2d(C, C, 1), nn.LeakyReLU(0.5), nn.Conv2d(C, self.cin, 1)).to(self.w_slot.device)
        if identity_init:
            _pass_through_(dec[0])
            _pass_through_(dec[2])
        return dec''')
open(p, "w", encoding="utf-8").write(s)

p = "tests/test_radar_encoder.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''FRESH 1 x 1 decoder trained on the frozen features (steps / 2, a lower bound: a longer probe can only keep more) - the module's own
decoder is meaningless when the side loss is off.''', '''FRESH 1 x 1 decoder (pass-through start, like the module's own) trained on the frozen features (steps / 2, a lower bound: a longer
probe can only keep more); where the side loss was on, the module's own jointly trained decoder is reported too (meaningless when the
side loss is off).''')
s = s.replace('''        res[mode] = (f1, k_te, k_tr)
        print(f"{mode:12s} | {f1:16.3f} | {100 * k_te:12.1f}% | {100 * k_tr:9.1f}% | {100 * (k_tr - k_te):+4.1f} | "
              + ", ".join(f"{100 * v:.0f}" for v in per), flush=True)''', '''        res[mode] = (f1, k_te, k_tr)
        own = "n/a" if mode == "task only" else f"{100 * kept(enc, te)[0]:.1f}%"
        print(f"{mode:12s} | {f1:16.3f} | {100 * k_te:12.1f}% | {100 * k_tr:9.1f}% | {100 * (k_tr - k_te):+4.1f} | {own:>9s} | "
              + ", ".join(f"{100 * v:.0f}" for v in per), flush=True)''')
s = s.replace('''print("mode         | task F1 held-out | kept held-out | kept train | gap  | violating''', '''print("mode         | task F1 held-out | kept held-out | kept train | gap  | own dec. | violating''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
