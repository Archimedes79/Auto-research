import io
p = "experiments/E343_test-e324s/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
a = "After the run: `results.md` / `card_s<S>.json` in `runs/eval/e324s_test/`"
assert s.count(a) == 1
b = ("**Device labels and the device effect (asked by the lane lead, 2026-11-15; no new comparison):** the report and the result section name the DEVICE of every arm and of the control: the E312 control and the plain graph are E321's stored TEST cards and dumps (CPU dumps); on the GPU branch the pair head of E324 s (arms w, p, k) and c2's B (arm k) are GPU dumps and B's E273 dump is E321's CPU dump; the scoring is CPU for all. "
     "Next to every difference that is smaller than the device effect the report states the bound of that effect (up to 3e-3 in a real number, a few integer counts: E321's measurement and the CPU rehearsal of E312 above) so that such a difference is read as inside it. ")
s = s.replace(a, b + a)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
