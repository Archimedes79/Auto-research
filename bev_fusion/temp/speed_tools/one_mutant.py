"""run ONE mutant of mutate_host.py by a part of its name and print every FAIL line: python temp/speed_tools/one_mutant.py "loop path" """
import subprocess, sys, os
sys.argv_saved = sys.argv[:]
part = sys.argv[1]
src = open("temp/speed_tools/mutate_host.py", encoding="utf-8").read().split("survived = 0")[0]
ns = {}
exec(src, ns)
for name, path, old, new, test in ns["MUTANTS"]:
    if part in name and path:
        ns["fresh"]()
        p = os.path.join(ns["WORK"], path)
        s = open(p, encoding="utf-8").read()
        open(p, "w", encoding="utf-8").write(s.replace(old, new))
        r = subprocess.run([sys.executable, "tests/" + test], cwd=ns["WORK"], capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="2"))
        print(name, "exit", r.returncode)
        print("\n".join(ln[:160] for ln in r.stdout.splitlines() if "FAIL" in ln))
        print(r.stderr[-400:])
