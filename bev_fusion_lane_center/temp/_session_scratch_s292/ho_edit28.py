import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
a = "E351 card (CUDA graph of the launch-bound forward, critic running);"
assert a in s
s = s.replace(a, "E351 v2 (probe: graph -23 % / -33 %, coder building jgraph.py); E291 v2 critic REVISE (v3 to write); report analyses in temp/report/ (own_lane);", 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
