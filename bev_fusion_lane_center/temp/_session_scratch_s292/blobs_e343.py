import io, subprocess
def head(p):
    return subprocess.run(["git", "show", "HEAD:bev_fusion_lane_center/" + p], capture_output=True).stdout.decode("utf-8")
def work(p):
    return io.open(p, encoding="utf-8", newline="").read()
nl = "\n"
# README: HEAD plus the two rows that E343 changes
h, w = head("evaluation/README.md"), work("evaluation/README.md")
hl, wl = h.split(nl), w.replace("\r\n", nl).split(nl)
for pre in ("| `jtest_eval.py` |", "| `tests/` |"):
    i = [k for k, l in enumerate(hl) if l.startswith(pre)]
    j = [k for k, l in enumerate(wl) if l.startswith(pre)]
    assert len(i) == 1 and len(j) == 1, (pre, i, j)
    hl[i[0]] = wl[j[0]]
io.open("temp/_session_scratch_s292/readme_e343_new.md", "w", encoding="utf-8", newline="").write(nl.join(hl))
# run_all_tests.sh: HEAD plus the two new run lines after the last E345 run line
h, w = head("tools/maintenance/run_all_tests.sh"), work("tools/maintenance/run_all_tests.sh")
hl, wl = h.split(nl), w.replace("\r\n", nl).split(nl)
new = [l for l in wl if "jtest_eval.py" in l and l.startswith("run ")]
assert len(new) == 2, new
last = max(k for k, l in enumerate(hl) if l.startswith("run ") and "jprune" in l)
hl[last + 1:last + 1] = new
io.open("temp/_session_scratch_s292/rat_e343_new.sh", "w", encoding="utf-8", newline="").write(nl.join(hl))
print("ok", len(new))
