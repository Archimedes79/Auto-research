import io
p = "/mnt/c/Users/mmeut/.claude/projects/C--Auto-research/memory/lane-e2e-day-d109-d111-2026-11-16.md"
s = io.open(p, encoding="utf-8").read()
a = "the ends are the bigger limit.\n"
assert s.count(a) == 1
s = s.replace(a, a + "- **jstack.py** (`evaluation/`, promoted 15:55 from temp/e363 scratch: surgery of a pair run onto another graph's node head, layer-faithful dump, node check, array check; 22 hand mutants killed; its `dump` was not re-run after the move: start with the fixture dump). State message to the lead sent 16:00.\n")
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
