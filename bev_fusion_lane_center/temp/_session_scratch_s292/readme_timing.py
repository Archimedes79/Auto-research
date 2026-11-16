import io
p = "evaluation/README.md"
s = io.open(p, encoding="utf-8", newline="").read()
lines = s.split("\n")
i = [k for k, l in enumerate(lines) if l.startswith("| `jtest_eval.py` |")]
assert len(i) == 1
row = ("| `jtiming.py` | E342 part B / E341 / E348: ms per frame of the junction system on an IDLE GPU (batch 1, bf16 autocast, cached input features of 12 VAL frames, CUDA events): `run --runs <registry runs>` times all models in one process in alternating blocks (A B C D | D C B A), "
       "prints the median per model, the paired differences with a bootstrap interval over the rounds, and a SECTION pass (forward hooks around pyramid / maps head / node head / pair head and markers inside `JEnds.forward`, inserted into a COPY of its source by line number: the frozen file is not edited; "
       "per section the GPU-timeline ms and the host ms) plus the CUDA kernel count of a forward (`torch.profiler`); refuses on a GPU that is not idle (`--force` records it); never overwrites a measurement (`runs/eval/jtiming/<tag>.json`); `selftest` (CPU: statistics, schedule, the instrumented forward equals the original, a changed source line is caught). Replaces the scratch `temp/jr/timing_junction.py` |")
lines.insert(i[0] + 1, row)
io.open(p, "w", encoding="utf-8", newline="").write("\n".join(lines))
p2 = "tools/maintenance/run_gates.sh"
t = io.open(p2, encoding="utf-8", newline="").read()
a = [l for l in t.split("\n") if "jwire_verdict.py selftest" in l]
assert len(a) == 1
t = t.replace(a[0], a[0] + '\nrun "timing: junction system ms per frame, section markers on a toy JEnds (E342 part B, CPU)"  jtiming.py selftest')
io.open(p2, "w", encoding="utf-8", newline="").write(t)
print("ok")
