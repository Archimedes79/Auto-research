import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("; E345 results.", ".")
rep("No mail / shutdown / install / push.", "No mail / shutdown / install / push.")
rep(" WSL restart kills chains: restart day14 / day16 / day21; day20 needs a new file.", " WSL restart kills chains: restart day14 / 16 / 21.")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
