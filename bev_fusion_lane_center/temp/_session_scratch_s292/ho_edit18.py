import io
p = "HANDOVER.md"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b, 1)
rep("E318.u_s1 done 06:44, post-processing day23, verdict after it;", "E318 CUT (Spearman 0.247 / 0.234 < 0.3, card section 4);")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
