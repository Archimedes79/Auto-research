"""scratch: which distill mutant crashes"""
import sys
import traceback
sys.argv = ["x", "new", "none_such_test_name_xyz"]
try:
    exec(open(__file__.replace("e379f_dbg.py", "e379f_harness.py")).read().split("want = [")[0])
except SystemExit:
    pass
kl = "torch.where(pt > 0, pt * (lpt - lp), 0.0)"
tkl = 'out[k].float().log_softmax(-1)), out["valid"])'
mutants = {"KL(p_S || p_T)": [(kl, "torch.where(lp.exp() > 0, lp.exp() * (lp - lpt), 0.0)")], "no 0 x log 0 guard": [(kl, "pt * (lpt - lp)")],
           "the mean of the logits": [("torch.logsumexp(torch.stack(lps), 0) - math.log(len(lps))", "torch.stack(lps).mean(0).log_softmax(-1)")],
           "the none slot left out": [('_kl(lpt, out["logp"])', '_kl(lpt[..., :-1], out["logp"][..., :-1])')]}
for n, e in mutants.items():
    try:
        TJ._check_kd(TJ._mutant(*e))
        print(n, "PASSED")
    except AssertionError:
        print(n, "assert")
    except Exception as ex:
        print(n, "OTHER", repr(ex)[:200])
