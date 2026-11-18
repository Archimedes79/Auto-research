"""Tests of `token_time_ens.py` (the cost of the plain ensemble of two token heads, E379 arm e2): CPU, no data: the selftest (the stacked `vmap` forward of two heads equals the two single forwards, the mean of the probabilities by hand, the mean of one head is the head) and textual
mutants of the module, each of which must make the selftest fail by an assertion. The timing itself is a GPU command (module docstring) read from its json.

    wsl bash tools/env/_py14.sh test_token_time_ens.py
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = os.path.join(HERE, "..", "token_time_ens.py")

MUTANTS = [
    ("the mean divides by one more", "return sum(p[\"p\"] for p in pr_list) / len(pr_list), sum(p[\"pt\"] for p in pr_list) / len(pr_list)", "return sum(p[\"p\"] for p in pr_list) / (len(pr_list) + 1), sum(p[\"pt\"] for p in pr_list) / (len(pr_list) + 1)"),
    ("the type probabilities not averaged", "sum(p[\"pt\"] for p in pr_list) / len(pr_list)", "pr_list[0][\"pt\"]"),
    ("the second head of the stack is the first", "torch.func.stack_module_state([m1, m2])", "torch.func.stack_module_state([m1, m1])"),
    ("the heads axis not sliced", "v[i] if torch.is_tensor(v) else v) for k, v in out.items()}) for i in range(n)]", "v[0] if torch.is_tensor(v) else v) for k, v in out.items()}) for i in range(n)]"),
    ("the heads are not different", "for seed in (1, 2):", "for seed in (1, 1):"),
    ("no wait before the after-sample", "        sleep(AGE_S)", "        pass"),
    ("the wait is too short", "AGE_S = 2.5 ", "AGE_S = 0.5 "),
    ("the CPU path waits too", "    if not cpu:\n        sleep(AGE_S)", "    if True:\n        sleep(AGE_S)"),
]


def run(src):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "token_time_ens.py")
        open(p, "w", encoding="utf-8").write(src)
        return subprocess.run([sys.executable, p, "selftest"], capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"), cwd=d, timeout=600)


def test_selftest_and_mutants():
    src = open(MODULE, encoding="utf-8").read()
    r = run(src)
    assert r.returncode == 0 and "selftest ok" in r.stdout, r.stdout + r.stderr
    survived = []
    for name, old, new in MUTANTS:
        assert src.count(old) == 1, f"mutant '{name}': the text to change occurs {src.count(old)} times"
        r = run(src.replace(old, new))
        if r.returncode == 0 or "AssertionError" not in r.stderr:
            survived.append((name, r.returncode, r.stderr[-200:]))
    assert not survived, f"mutants not killed by an assertion: {survived}"
    print(f"token_time_ens: selftest ok, {len(MUTANTS)} mutants killed")


if __name__ == "__main__":
    test_selftest_and_mutants()
