import os
root = os.environ["BFLC_ROOT"]
p = os.path.join(root, "experiments", "E368_s2-cross-attention-to-the-past", "experiment.md")
s = open(p, encoding="utf-8").read()
old = "(scratch, listed here as how the numbers were made; to be promoted if S2 is revisited)."
new = "(scratch, listed here as how the numbers were made). **Promoted 2026-11-17** (a second use, E374): `choose`, `ap` and `every3` are now `evaluation/token_readout.py` (the cells and the AP of this section are reproduced by it on the E368 runs' tables); the paired interval of `paired.py` is `jtoken_eval.py score --ref-run`, the stream-mode controls stay in `jpast_eval.py`."
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
p = os.path.join(root, "experiments", "E374_s3-cross-attention-to-the-sd-map", "experiment.md")
s = open(p, encoding="utf-8").read()
old = "Prediction +0.006 (window 0.000 to +0.012): hit; outcome odds TIE 60 %: came true."
new = old + " **Promoted 2026-11-17:** `temp/e374/modes.py` and `temp/e368/{choose,ap,mk_every3}.py` are now `evaluation/token_readout.py` (`modes`, `choose`, `ap`, `every3`; tests with 13 mutants); on the E374 runs it reproduces this section's cells (0.1 / 0.6 / off) and AP (0.9539 / 0.9442 of m_s0 / z_s0); the paired intervals of `paired.py` are `jtoken_eval.py score --ref-run`."
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("cards patched")
