p = "temp/_session_scratch_s292/mutants_distill.py"
s = open(p, encoding="utf-8").read()
a = s.index("MUT = {")
b = s.index("for name, (a, b) in MUT.items():")
new = '''MUT = {
    "no detach": ('tout["j1"].detach().float()', 'tout["j1"].float()'),
    "soft heading weight": ("m = pk[:, k]", "m = pt[:, k]"),
    "pos-weighted BCE": ('F.binary_cross_entropy_with_logits(s[:, 2:4], pt, reduction="sum") / pk.sum().clamp_min(1.0)', 'F.binary_cross_entropy_with_logits(s[:, 2:4], pt, pos_weight=torch.tensor(20.0))'),
    "reversed heading sign": ("l_d = l_d + ((1 - (ps * th).sum(1)) * m).sum()", "l_d = l_d + ((1 + (ps * th).sum(1)) * m).sum()"),
    "exit heat dropped from BCE": ("F.binary_cross_entropy_with_logits(s[:, 2:4], pt,", "F.binary_cross_entropy_with_logits(s[:, 2:3], pt[:, 0:1],"),
    "exit heading reads the entry channels": ("ps, th = s[:, 4 + 2 * k: 6 + 2 * k], t[:, 4 + 2 * k: 6 + 2 * k]", "ps, th = s[:, 4:6], t[:, 4:6]"),
    "peak 0.05": ("def distill_loss(out, tout, peak=0.3):", "def distill_loss(out, tout, peak=0.05):"),
    "peak 0.9": ("def distill_loss(out, tout, peak=0.3):", "def distill_loss(out, tout, peak=0.9):"),
    "mean instead of sum / peaks": ('reduction="sum") / pk.sum().clamp_min(1.0)', 'reduction="mean")'),
    "mass instead of peaks": ('reduction="sum") / pk.sum().clamp_min(1.0)', 'reduction="sum") / pt.sum().clamp_min(1.0)'),
    "peaks of the entry channel only": ('reduction="sum") / pk.sum().clamp_min(1.0)', 'reduction="sum") / pk[:, 0].sum().clamp_min(1.0)'),
}
'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
