import io
import subprocess

src = io.open("evaluation/jstack.py", encoding="utf-8").read()
M = [
    ("node_diff: missing key allowed", 'assert k in nd and k in nb, f"node output {k!r} missing (pair model: {sorted(nd)}, B\': {sorted(nb)}): nothing compared"', "if k not in nd or k not in nb:\n            continue"),
    ("node_diff: e_raw never required", '(("e_raw",) if layered else ())', "()"),
    ("node_diff: NaN passes", "ok = ok and v == v and v <= 1e-5", "ok = ok and v <= 1e-5 or v != v"),
    ("node_diff: tolerance 1e-3", "v <= 1e-5\n    return worst, ok", "v <= 1e-3\n    return worst, ok"),
    ("node_diff: worst dropped", "worst[k] = max(worst.get(k, 0.0), v) if v == v else float(\"nan\")", "worst[k] = 0.0"),
]
killed = 0
for name, a, b in M:
    if src.count(a) != 1:
        print(f"NOT APPLIED  {name}: {src.count(a)} matches")
        continue
    io.open("temp/_session_scratch_e358/mut_bs/jstack.py", "w", encoding="utf-8").write(src.replace(a, b))
    r = subprocess.run(["bash", "tools/env/_py14.sh", "temp/_session_scratch_e358/mut_bs/jstack.py", "selftest"], capture_output=True, text=True)
    killed += r.returncode != 0
    print(("killed       " if r.returncode != 0 else "SURVIVED     ") + name)
print(f"{killed} of {len(M)} killed")
