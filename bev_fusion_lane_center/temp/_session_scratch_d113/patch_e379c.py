import os
p = os.path.join(os.environ["BFLC_ROOT"], "experiments", "E379_ensemble-distilled-head", "experiment.md")
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
rep("Its cost is measured by the job E379.time_e2 (GPU, night window, quiet host, batch 1, bf16, the 24-frame protocol of `jtoken_eval.py timing`):", "Its cost is measured by the job E379.time_e2 (`evaluation/token_time_ens.py`, committed 420a003f, tested with 5 mutants; GPU, a quiet gap, batch 1, bf16, the 24-frame protocol of `jtoken_eval.py timing`; the waiter `temp/e379/time_e2_waiter.sh` starts it as the LAST GPU job of the night, the lead's order):")
rep("`jtoken_eval.py ensemble` and the 2-head timing paths (`timing --heads A,B --stacked`);", "`jtoken_eval.py ensemble` (Fable; built, layer 1 exact); the 2-head timing is `evaluation/token_time_ens.py` (lane chat, committed 420a003f: not to be added to `jtoken_eval.py`);")
rep("| `jtoken_eval.py timing --heads` (to add), job `E379.time_e2`, GPU night window, quiet host |", "| `evaluation/token_time_ens.py` (committed 420a003f), job `E379.time_e2` via `temp/e379/time_e2_waiter.sh`, the last GPU job of the night, quiet host |")
open(p, "w", encoding="utf-8").write(s)
print("ok")
