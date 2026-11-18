"""E339: the pos-weighted BCE puts the minimum above the teacher's logits (pos_weight 20 inflates a probability of 0.01 to 0.17); plain soft-target BCE has its minimum at the teacher's own logits,
normalised by the teacher's probability mass (the number of positives, as CenterNet / FCOS do) so that its gradient is not diluted by the 10 000 empty cells."""


def rd(p):
    return open(p, encoding="utf-8").read()


def sub(s, a, b):
    assert s.count(a) == 1, a[:80]
    return s.replace(a, b)


p = "training/grid/jmaps.py"
s = rd(p)
s = sub(s, "def distill_loss(out, tout, pos_w=20.0, peak=0.3):", "def distill_loss(out, tout, peak=0.3):")
s = sub(s, '''the entry / exit heat logits (channels 2:4) against the teacher's heat PROBABILITIES as soft targets, BCE with the positive weight of the ground-truth
    heat loss (a softmax over the cells would give a weak end next to a strong one a fraction of a percent of the mass; here every cell keeps its weight), + the cosine''', '''the entry / exit heat logits (channels 2:4) against the teacher's heat PROBABILITIES as soft targets: plain BCE (its minimum is the teacher's own
    logit; the positive weight 20 of the ground-truth loss would move the minimum from 0.01 to 0.17), summed over the cells and divided by the teacher's probability mass (the number of positives, as CenterNet / FCOS normalise: a mean over the 10 000 cells would
    dilute the gradient of the ends to nothing; a softmax over the cells, CWD, would give a weak end next to a strong one a fraction of a percent of the mass), + the cosine''')
s = sub(s, "    l_h = F.binary_cross_entropy_with_logits(s[:, 2:4], pt, pos_weight=torch.tensor(pos_w, device=s.device))\n",
        "    l_h = F.binary_cross_entropy_with_logits(s[:, 2:4], pt, reduction=\"sum\") / pt.sum().clamp_min(1.0)\n")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
