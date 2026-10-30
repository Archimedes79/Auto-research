p = "training/radar_encoder.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    def aux_loss(self, features, state):
        """Side loss: count (L1 on log1p), valid (BCE) and the tolerance-weighted L1 of the stored fields at valid slots."""
        out = self.decoder(features)''', '''    def new_decoder(self, C=None):
        """A fresh 1 x 1 decoder (default start) for probing frozen features: the honest way to measure what a task-trained encoder kept."""
        C = C or self.net[2].out_channels
        return nn.Sequential(nn.Conv2d(C, C, 1), nn.LeakyReLU(0.5), nn.Conv2d(C, self.cin, 1)).to(self.w_slot.device)

    def aux_loss(self, features, state, decoder=None):
        """Side loss: count (L1 on log1p), valid (BCE) and the tolerance-weighted L1 of the stored fields at valid slots."""
        out = (decoder or self.decoder)(features)''')
s = s.replace('''    def kept_fraction(self, det, B, features=None):''', '''    def kept_fraction(self, det, B, features=None, decoder=None):''')
s = s.replace('''        x = state if features is None else self.decoder(features)''', '''        x = state if features is None else (decoder or self.decoder)(features)''')
s = s.replace('''within tolerance after decoding the features (from the raster itself when None: the ceiling), and the''', '''within tolerance after decoding the features (from the raster itself when None: the ceiling; `decoder` = a probe, default the
        module's own side-loss decoder, which is only meaningful while the side loss trains it), and the''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
