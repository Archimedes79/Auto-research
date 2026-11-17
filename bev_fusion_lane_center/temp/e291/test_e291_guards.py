"""E291 tests of the guards of the launch scripts (plain python, no torch; WSL system python3):  wsl python3 temp/e291/test_e291_guards.py [--start <start_arms.sh>] [--wrapper <run_staged_gpu.sh>]
Every case runs `start_arms.sh` with E291_DRY=1 (it prints the run.py line instead of starting it) or the GPU wrapper with a stage that is not its own; the options take mutated copies of the scripts (temp/e291/mutate_e291.py).
"""
import argparse
import hashlib
import os
import subprocess
import sys
import tempfile

ROOT = os.environ.get("E291_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--start", default=os.path.join(ROOT, "temp", "e291", "start_arms.sh"))
ap.add_argument("--wrapper", default=os.path.join(ROOT, "temp", "e291", "run_staged_gpu.sh"))
a = ap.parse_args()
STAGE = os.path.join(ROOT, "temp", "_session_scratch_e291", "stage")
E347 = os.path.join(ROOT, "temp", "_session_scratch_e347", "stage")


def sh(script, args, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("E291_") and k != "BFLC_STAGE"}
    e.update(E291_ROOT=ROOT, **{k: v for k, v in env.items() if v is not None})
    p = subprocess.run(["bash", script, *args], env=e, capture_output=True, text=True, cwd=ROOT)
    return p.returncode, p.stdout, p.stderr


def sha_list(path, bad=False):
    files = [os.path.join("temp", "_session_scratch_e291", "stage", f) for f in ("training/grid/jends.py", "training/data/lane_gt_data.py", "training/train/jends_train.py", "evaluation/jends_eval.py")] + ["training/grid/jprior.py", "evaluation/jdiag.py"]
    rows = [f"{hashlib.sha256(open(os.path.join(ROOT, f), 'rb').read()).hexdigest()}  {f}" for f in files]
    if bad:
        rows[0] = "0" * 64 + rows[0][64:]
    open(path, "w").write("\n".join(rows) + "\n")


with tempfile.TemporaryDirectory(prefix="e291_guards_") as tmp:
    good, bad, runs = os.path.join(tmp, "good.sha"), os.path.join(tmp, "bad.sha"), os.path.join(tmp, "runs")
    sha_list(good), sha_list(bad, bad=True)
    os.makedirs(os.path.join(runs, "e291_c_s0"))
    # (1) the stage is set by the script, whatever the shell exports (E347's stage in BFLC_STAGE); the right flags per arm; the staged wrapper and the twin without the prior flags
    for arm, flags in (("c", None), ("a", "--lane-prior --prior-noise 0"), ("b", "--lane-prior --prior-noise mix")):
        rc, out, err = sh(a.start, [arm, "0"], E291_DRY="1", E291_SHA_FILE=good, E291_RUNS_DIR=os.path.join(tmp, "norun"), BFLC_STAGE=E347)
        assert rc == 0 and f"BFLC_STAGE={STAGE} " in out and E347 not in out, (arm, rc, out[:300], err[:300])
        assert f"--id E291.{arm}_s0" in out and f"--name e291_{arm}_s0" in out and "temp/e291/run_staged_gpu.sh jends_train.py" in out and "BFLC_SIG=1.0" in out, out[:600]
        assert ("--lane-prior" in out) == (flags is not None) and (flags is None or out.strip().endswith(flags)), (arm, out[-120:])
    # (2) the freeze: a wrong hash, a missing list: refused (3); no run line printed
    for lst in (bad, os.path.join(tmp, "missing.sha")):
        rc, out, err = sh(a.start, ["c", "0"], E291_DRY="1", E291_SHA_FILE=lst, E291_RUNS_DIR=os.path.join(tmp, "norun"))
        assert rc == 3 and "REFUSED" in out and "DRY" not in out, (lst, rc, out[:200])
    # (3) an existing run folder: refused (4); with E291_RESUME=1 it goes on; another arm / seed is free
    rc, out, err = sh(a.start, ["c", "0"], E291_DRY="1", E291_SHA_FILE=good, E291_RUNS_DIR=runs)
    assert rc == 4 and "REFUSED" in out and "DRY" not in out, (rc, out[:200])
    rc, out, err = sh(a.start, ["c", "0"], E291_DRY="1", E291_SHA_FILE=good, E291_RUNS_DIR=runs, E291_RESUME="1")
    assert rc == 0 and "DRY" in out, (rc, out[:200])
    rc, out, err = sh(a.start, ["a", "0"], E291_DRY="1", E291_SHA_FILE=good, E291_RUNS_DIR=runs)
    assert rc == 0 and "DRY" in out, (rc, out[:200])
    # (4) the GPU wrapper accepts only its own stage: E347's, a made-up one and `none` are refused before anything runs (the tests have their own wrapper, run_staged.sh, which sets E291_TEST)
    for st in (E347, os.path.join(tmp, "stage"), "none"):
        rc, out, err = sh(a.wrapper, ["jprior.py"], BFLC_STAGE=st)
        assert rc == 2 and "REFUSED" in err, (st, rc, err[:200])
print("test_e291_guards ok")
