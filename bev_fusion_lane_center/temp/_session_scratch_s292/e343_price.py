import io
p = "experiments/E343_test-e324s/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
a = "**What the outcome decides.**"
if a not in s:
    a = "## 0. The card (what the tools run)"
assert s.count(a) >= 1
add = ("**The price of E324 s on one line (2026-11-15, for the report to Mirko):** +0.0156 / +0.0146 snapped topo F1 over E312 on VAL for **+7.5 ms per frame** (24.58 against 17.30 ms, paired interval [+6.16, +8.59]; the node head 3.7 ms and the attachment 2.8 ms of it; batch 1, bf16, idle GPU, loaded host: E342 card section 5); the TEST use measures the gain, not the ms.\n\n")
i = s.index(a)
s = s[:i] + add + s[i:]
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
