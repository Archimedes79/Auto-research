import io, subprocess
src = io.open("evaluation/jwire_check.py", encoding="utf-8").read()
M = [("faithful exclusion removed", "if a.faithful and a.bs is not None:", "if False:"),
     ("bs default not 4", "a.bs = 4 if a.bs is None else a.bs", "a.bs = 5 if a.bs is None else a.bs"),
     ("deviation: min instead of max", "float(np.nanmax(np.abs(x - y)))", "float(np.nanmin(np.abs(x - y)))"),
     ("faithful not passed on", "ends, bs=a.bs, faithful=a.faithful)", "ends, bs=a.bs, faithful=False)")]
for name, a, b in M:
    assert src.count(a) == 1, name
    io.open("temp/_session_scratch_e358/mut_bs/jwire_check.py", "w", encoding="utf-8").write(src.replace(a, b))
    r = subprocess.run(["bash", "tools/env/_py14.sh", "temp/_session_scratch_e358/mut_bs/jwire_check.py", "selftest"], capture_output=True, text=True)
    print(name, "KILLED" if r.returncode != 0 else "SURVIVED")
