import io
p = "experiments/E358_wiring-head-on-attached-ends/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
add = io.open("temp/_session_scratch_s292/e358_summary.md", encoding="utf-8").read().replace("\n", nl)
a = "## 6. Summary" + nl + "(empty until the runs exist)"
assert s.count(a) == 1
s = s.replace(a, "## 6. Summary" + nl + add.rstrip() + nl)
s = s.replace("**Status: card v2, not yet passed by a second critic (section 4);", "**RESULT 2026-11-16 13:35: KEEP+ (section 6).** **Status (at the time of writing the card): card v2, not yet passed by a second critic (section 4);", 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
