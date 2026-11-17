p = "evaluation/jpast_eval.py"
s = open(p, encoding="utf-8").read()
a = "sums d2 / x2 / n of the kill measure u = RMS(block(x, past) - block(x, none key only)) / RMS(x) (0 without a past) and state_ss / state_n"
b = ("the sums of the kill measures (`jpast.SUMS`: d2_none / x2 / n against the none key only, d2_other / x2_other / n_other against the past of ANOTHER SCENE = the donor of `donor_plan`, VAL and a block only; 0 without a past) and state_ss / state_n")
assert s.count(a) == 1
s = s.replace(a, b)
a = "`use` writes u and the RMS of the carried state per stream position"
b = ("`use` writes u = RMS(update with the past - update with another scene's past) / RMS(x) (the kill measure of the card; 0 when the block ignores the past content), u_none (against the none key only: any content-free attention satisfies it, reported next to u, no verdict) and the RMS of the carried state per stream position")
assert s.count(a) == 1
s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
