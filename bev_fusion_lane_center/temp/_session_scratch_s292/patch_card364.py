import io
p = "experiments/E364_what-r4-and-r6-cost/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
add = io.open("temp/_session_scratch_s292/e364_summary.md", encoding="utf-8").read().replace("\n", nl)
a = "## 6. Summary" + nl + "(empty until the runs exist)"
assert s.count(a) == 1
s = s.replace(a, "## 6. Summary" + nl + add.rstrip() + nl)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
