"""The loss of the camera student (net/model/student_encoder.py) against the cached teachers: per cell the KL of the depth distribution and the cosine and
squared error of the context, on both heads (stride 16 = DualBEV, stride 8 = BEVFusion). In the chain it is the anchor of a student that trains
(train_terms.py)."""
import torch.nn.functional as F


def cell_maps(out, tgt):
    """the per-cell terms, each (B, h, w): kl<k> = KL(teacher || student) over the 118 depth bins, cos<k> = cosine similarity of the 80 context channels, mse<k>
    = their mean squared error; k = 16 (DualBEV) and 8 (BEVFusion). The teacher probabilities are renormalised (they are stored as float16)."""
    m = {}
    for k in ("16", "8"):
        t = tgt["p" + k].float()
        t = t / t.sum(1, keepdim=True)
        m["kl" + k] = (t * (t.clamp_min(1e-8).log() - out["logit" + k].float().log_softmax(1))).sum(1)
        c, tc = out["ctx" + k].float(), tgt["c" + k].float()
        m["cos" + k] = F.cosine_similarity(c, tc, dim=1)
        m["mse" + k] = (c - tc).pow(2).mean(1)
    return m


def distill_terms(out, tgt):
    """the terms from the cell means: kl<k>, cosl<k> (= 1 - cosine), mse<k> and their sum over both heads, equal weights (`total`)"""
    mean = {n: v.mean() for n, v in cell_maps(out, tgt).items()}
    terms = {n: v for n, v in mean.items() if not n.startswith("cos")}
    terms.update({"cosl" + k: 1 - mean["cos" + k] for k in ("16", "8")})
    terms["total"] = sum(mean["kl" + k] + 1 - mean["cos" + k] + mean["mse" + k] for k in ("16", "8"))
    return terms


def anchor_terms(side):
    """distill_terms of the chain's `student_side` (the model result's entry): the student's logits and context of the current frame against the cached
    teacher's probabilities and context of the same frame; the anchor is its ["total"]"""
    out = {n + k: side[c + k] for k in ("16", "8") for n, c in (("logit", "l"), ("ctx", "c"))}
    tgt = {n + k: side["t_" + n + k] for k in ("16", "8") for n in ("p", "c")}
    return distill_terms(out, tgt)


def anchor_weight(step_frac, start, floor, until):
    """the weight of the anchor at the fraction `step_frac` of the steps: `start` at the beginning, falling linearly to start * floor at the fraction `until`,
    constant after"""
    return start * (1.0 - (1.0 - floor) * min(1.0, max(0.0, step_frac) / until))
