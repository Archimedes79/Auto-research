p = "temp/_session_scratch_s292/patch_distill5.py"
s = open(p, encoding="utf-8").read()
a = s.index("s = sub(s, '''summed over the cells and divided by the teacher's probability mass")
b = s.index("s = sub(s, '''    l_h = F.binary_cross_entropy_with_logits(s[:, 2:4], pt, reduction=\"sum\")")
new_first = """s = sub(s, "summed over the cells and divided by the teacher's probability mass (the number of positives, as CenterNet / FCOS normalise: a mean over the 10 000 cells would\n    dilute the gradient of the ends to nothing; a softmax over the cells, CWD, would give a weak end next to a strong one a fraction of a percent of the mass), + the cosine",
        "summed over the cells and divided by the number of peak cells of the teacher (probability > peak, the mask of the heading term: the number of positives, as CenterNet / FCOS normalise; a mean over the 10 000 cells\n    would dilute the gradient of the ends to nothing, the probability mass would make the scale depend on the teacher's background calibration; a softmax over the cells, CWD, would give a weak end next to a strong one a fraction of a percent of the mass), + the cosine")
"""
s = s[:a] + new_first + s[b:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched the patch")
