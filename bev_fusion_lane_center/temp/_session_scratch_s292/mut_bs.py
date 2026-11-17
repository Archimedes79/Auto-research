import io, subprocess, os
src = io.open("evaluation/jwire_check.py", encoding="utf-8").read()
M = [("live_records without bs=bs", "workers, bs=bs, device=dev", "workers, device=dev"),
     ("CLI: bs ignored", "ends, bs=a.bs)", "ends, bs=4)"),
     ("CLI: default 4 -> 5", 'v.add_argument("--bs", type=int, default=4', 'v.add_argument("--bs", type=int, default=5')]
for name, a, b in M:
    assert src.count(a) == 1, name
    io.open("temp/_session_scratch_e358/mut_bs/jwire_check.py", "w", encoding="utf-8").write(src.replace(a, b))
    r = subprocess.run(["bash", "tools/env/_py14.sh", "temp/_session_scratch_e358/mut_bs/jwire_check.py", "selftest"], capture_output=True, text=True)
    print(name, "KILLED" if r.returncode != 0 else "SURVIVED", (r.stderr.strip().splitlines() or [""])[-1][:100])
