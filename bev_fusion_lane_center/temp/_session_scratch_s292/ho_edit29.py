import io
p = "HANDOVER.md"
t = io.open(p, encoding="utf-8").read()
b = "E346 done;"
if b in t:
    t = t.replace(b, "", 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(t)
