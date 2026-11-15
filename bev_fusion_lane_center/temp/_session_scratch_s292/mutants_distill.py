"""Scratch: the jmaps selftest must FAIL for each mutant of distill_loss (no detach, a soft heading weight, the pos-weighted BCE)."""
import os
import sys

ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "training/grid"))
sys.path.insert(0, os.path.join(ROOT, "training"))
src = open(os.path.join(ROOT, "training/grid/jmaps.py"), encoding="utf-8").read()
MUT = {
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
for name, (a, b) in MUT.items():
    assert src.count(a) == 1, name
    ns = {"__name__": "mutant"}
    exec(compile(src.replace(a, b), "jmaps_mutant.py", "exec"), ns)
    try:
        ns["_selftest"]()
        print(f"MUTANT SURVIVES: {name}")
    except Exception:
        print(f"mutant killed: {name}")
