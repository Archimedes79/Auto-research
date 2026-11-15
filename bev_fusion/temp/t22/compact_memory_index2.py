"""scratch: second pass over MEMORY.md: a line longer than 185 characters is cut after the last ';' or ',' before character 170 (the detail lives in the topic file the link names).
The lines of the t22 entry and of the ResNet-18 reminder are left as they are."""
import re

P = "/mnt/c/Users/mmeut/.claude/projects/C--Auto-research/memory/MEMORY.md"
KEEP = ("bev-fusion-t22-student-encoder.md", "reminder-resnet18-2026-11-09.md")
text = open(P, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in text else "\n"
lines = text.split(nl)
n = 0
for i, ln in enumerate(lines):
    m = re.match(r"- \[[^\]]*\]\(([^)]+)\)", ln)
    if not m or m.group(1) in KEEP or len(ln) <= 185:
        continue
    head = ln[:170]
    cut = max(head.rfind(";"), head.rfind(","))
    if cut < m.end() + 40:                       # nothing sensible to cut at
        continue
    lines[i] = head[:cut].rstrip()
    n += 1
out = nl.join(lines)
open(P, "w", encoding="utf-8", newline="").write(out)
print("shortened", n, "lines; bytes", len(text.encode("utf-8")), "->", len(out.encode("utf-8")))
